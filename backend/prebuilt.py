"""Install official prebuilt llama.cpp binaries - no compiler needed.

ggml-org publishes a build for every merge (`bNNNN`, marked prerelease) and,
every week or two, a curated `v0.x` release whose `nightly-tag.txt` names the
build it blessed. Every asset carries a GitHub `sha256:` digest, so a download
is verified against GitHub's own record before anything is extracted.

Installs live side by side in engines/llama.cpp/<tag>-<variant>/, each with an
lf-install.json manifest. Switching versions (update or rollback) is only a
change of `server_bin`; nothing is overwritten in place. CUDA builds need the
separate cudart bundle, which is cached once per digest and hard-linked into
each install, so hourly updates don't re-download ~400 MB of runtime.

The pure helpers (parse/choose/extract/list) are unit-tested; the Installer
thread composes them. Pure stdlib.
"""
import hashlib, json, os, re, shutil, stat, subprocess, tarfile, threading, time
import urllib.request, zipfile

import osplat

API = "https://api.github.com/repos/ggml-org/llama.cpp"
UA = {"User-Agent": "LlamaForge (+https://github.com/dadwritestech/LlamaForge)",
      "Accept": "application/vnd.github+json"}
MANIFEST = "lf-install.json"
CHANNELS = ("nightly", "stable")
KEEP_INSTALLS = 3
SERVER_NAMES = ("llama-server.exe", "llama-server")

# Generic variants in order of preference when no CUDA build applies.
_FALLBACK_ORDER = ("metal", "vulkan", "cpu")
# Blackwell (compute capability 10.x / 12.x) kernels first shipped in CUDA 12.8.
_BLACKWELL_MIN_CUDA = (12, 8)

_BIN_RE = re.compile(r"^llama-b\d+-bin-(win|ubuntu|macos)-(.+)-(x64|arm64)\.(?:zip|tar\.gz)$")
_BIN_NOVARIANT_RE = re.compile(r"^llama-b\d+-bin-(win|ubuntu|macos)-(x64|arm64)\.(?:zip|tar\.gz)$")
_CUDART_RE = re.compile(
    r"^cudart-llama-(?:b\d+-)?bin-(win|ubuntu)-cuda-(\d+)\.(\d+)-(x64|arm64)\.(?:zip|tar\.gz)$")
_CUDA_VARIANT_RE = re.compile(r"^cuda-(\d+)\.(\d+)$")


# ------------------------------------------------------------------ pure helpers

def platform_key(os_name=None, machine=None):
    """(release-os, arch) as llama.cpp names them: win/ubuntu/macos, x64/arm64."""
    os_name = os_name or osplat.current()
    if machine is None:
        import platform
        machine = platform.machine()
    rel_os = {"windows": "win", "linux": "ubuntu", "macos": "macos"}.get(os_name, os_name)
    m = (machine or "").lower()
    arch = "arm64" if m in ("arm64", "aarch64", "armv8", "armv8l") else "x64"
    return rel_os, arch


def parse_cuda_driver(nvidia_smi_text):
    """Highest CUDA version the installed driver supports, from nvidia-smi's
    header, as (major, minor) - or None. R600+ drivers print "CUDA UMD Version"
    where older ones print "CUDA Version"."""
    m = re.search(r"CUDA (?:UMD )?Version:\s*(\d+)\.(\d+)", nvidia_smi_text or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_asset(name):
    """Decode a release asset name. None for anything we don't install
    (UI tarball, xcframework, Android, Snapdragon, s390x, nightly-tag.txt)."""
    m = _CUDART_RE.match(name)
    if m:
        major, minor = int(m.group(2)), int(m.group(3))
        return {"kind": "cudart", "os": m.group(1), "arch": m.group(4),
                "variant": f"cuda-{major}.{minor}", "cuda": (major, minor)}
    m = _BIN_NOVARIANT_RE.match(name)
    if m:
        rel_os, arch = m.group(1), m.group(2)
        variant = "metal" if (rel_os, arch) == ("macos", "arm64") else "cpu"
        return {"kind": "bin", "os": rel_os, "arch": arch, "variant": variant, "cuda": None}
    m = _BIN_RE.match(name)
    if m:
        variant = m.group(2)
        cm = _CUDA_VARIANT_RE.match(variant)
        return {"kind": "bin", "os": m.group(1), "arch": m.group(3), "variant": variant,
                "cuda": (int(cm.group(1)), int(cm.group(2))) if cm else None}
    return None


def _uploaded(asset):
    return asset.get("state", "uploaded") == "uploaded"


def _is_blackwell(gpus):
    for g in gpus or []:
        cc = str(g.get("compute_cap") or "")
        try:
            if int(cc.split(".")[0]) >= 10:
                return True
        except ValueError:
            pass
    return False


def choose(assets, plat, gpus, driver, variant=None):
    """Pick the binary (and cudart bundle) for this machine.

    plat: (release-os, arch). gpus: NVIDIA GPUs from hardware.detect_gpus().
    driver: parse_cuda_driver() result. variant: explicit user override.
    Returns {bin, cudart, variant, reason, alternatives}; bin is None when
    nothing usable (or nothing fully uploaded) exists for the platform.
    """
    rel_os, arch = plat
    bins, cudarts = {}, {}
    for a in assets or []:
        info = parse_asset(a.get("name", ""))
        if not info or info["os"] != rel_os or info["arch"] != arch:
            continue
        (bins if info["kind"] == "bin" else cudarts)[info["variant"]] = (a, info)
    out = {"bin": None, "cudart": None, "variant": None, "reason": "",
           "alternatives": sorted(bins)}

    pick, reason = None, ""
    if variant:
        if variant in bins:
            pick, reason = variant, "chosen manually"
        else:
            out["reason"] = f"no {variant} build for {rel_os}-{arch}"
            return out
    elif gpus and driver:
        floor = _BLACKWELL_MIN_CUDA if _is_blackwell(gpus) else (0, 0)
        cudas = sorted((info["cuda"], v) for v, (_, info) in bins.items()
                       if info["cuda"] and info["cuda"][0] <= driver[0]
                       and info["cuda"] >= floor)
        if cudas:
            pick = cudas[-1][1]
            reason = (f"NVIDIA GPU, driver supports CUDA {driver[0]}.{driver[1]}")
        else:
            reason = (f"no CUDA build fits driver CUDA {driver[0]}.{driver[1]}"
                      + (" on Blackwell" if floor != (0, 0) else "")
                      + " - update the NVIDIA driver for CUDA speed")
    if not pick:
        for v in _FALLBACK_ORDER:
            if v in bins:
                pick = v
                reason = reason or {"metal": "Apple Silicon (Metal)",
                                    "vulkan": "Vulkan works on NVIDIA, AMD and Intel GPUs",
                                    "cpu": "CPU build"}[v]
                break
    if not pick:
        out["reason"] = f"no prebuilt llama.cpp for {rel_os}-{arch}"
        return out

    asset, info = bins[pick]
    cudart = cudarts[pick][0] if info["cuda"] and pick in cudarts else None
    out.update(variant=pick, reason=reason)
    # Pick first, then insist it's uploaded: falling back to CPU just because
    # CI hasn't attached the CUDA zip yet would be a silent downgrade.
    # resolve() moves on to the previous build instead.
    if not _uploaded(asset) or (cudart and not _uploaded(cudart)):
        out["reason"] = f"{pick} build is still uploading"
        return out
    out.update(bin=asset, cudart=cudart)
    return out


def _within(root, path):
    root = os.path.normcase(os.path.realpath(root))
    path = os.path.normcase(os.path.realpath(path))
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def _check_name(dest, name):
    if not name or name.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", name):
        raise ValueError(f"absolute path in archive: {name}")
    if not _within(dest, os.path.join(dest, name)):
        raise ValueError(f"path escapes archive root: {name}")


def safe_extract(archive, dest):
    """Extract a .zip or .tar.gz into dest, refusing any member (or link
    target) that would land outside it. Everything is checked before the first
    byte is written, so a hostile archive leaves nothing behind."""
    os.makedirs(dest, exist_ok=True)
    if archive.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            for n in z.namelist():
                _check_name(dest, n)
            z.extractall(dest)
        return
    with tarfile.open(archive, "r:*") as t:
        members = []
        for m in t.getmembers():
            _check_name(dest, m.name)
            if m.issym():
                _check_name(dest, os.path.join(os.path.dirname(m.name), m.linkname)
                            if not os.path.isabs(m.linkname) else m.linkname)
            elif m.islnk():
                _check_name(dest, m.linkname)
            elif not (m.isfile() or m.isdir()):
                continue                       # devices / fifos: never needed
            members.append(m)
        if hasattr(tarfile, "data_filter"):
            t.extractall(dest, members=members, filter="data")
        else:                                  # Python < 3.10.12: checked above
            t.extractall(dest, members=members)


def find_server_bin(root):
    """Shallowest llama-server(.exe) under root, or None."""
    best = None
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for n in SERVER_NAMES:
            if n in filenames:
                p = os.path.join(dirpath, n)
                if best is None or p.count(os.sep) < best.count(os.sep):
                    best = p
    return best


def list_installs(root, active_bin=None):
    """Installed builds (dirs with a manifest), newest first."""
    out = []
    if not os.path.isdir(root):
        return out
    for n in os.listdir(root):
        d = os.path.join(root, n)
        mp = os.path.join(d, MANIFEST)
        if not os.path.isfile(mp):
            continue
        try:
            with open(mp, encoding="utf-8") as f:
                man = json.load(f)
        except (OSError, ValueError):
            continue
        sbin = find_server_bin(d)
        out.append(dict(man, dir=d, server_bin=sbin,
                        active=bool(active_bin and _within(d, active_bin))))
    out.sort(key=lambda i: i.get("installed_at") or 0, reverse=True)
    return out


def prune_installs(root, keep=KEEP_INSTALLS, active_bin=None, protect=()):
    """Delete all but the newest `keep` installs, never the active one or a
    `protect`ed dir (builds a launch profile pins)."""
    removed = []
    spare = {os.path.normcase(os.path.abspath(d)) for d in protect}
    for i, inst in enumerate(list_installs(root, active_bin)):
        if i < keep or inst["active"] or os.path.normcase(os.path.abspath(inst["dir"])) in spare:
            continue
        shutil.rmtree(inst["dir"], ignore_errors=True)
        removed.append(inst["dir"])
    return removed


# ------------------------------------------------------------------ network

def _get_json(url, timeout=25):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _get_text(url, timeout=25):
    req = urllib.request.Request(url, headers={"User-Agent": UA["User-Agent"]})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def _norm(rel, channel, label=None):
    return {"tag": rel.get("tag_name", ""), "label": label or rel.get("tag_name", ""),
            "published": rel.get("published_at", ""), "channel": channel,
            "assets": rel.get("assets") or []}


def resolve(channel, get_json=_get_json, get_text=_get_text,
            plat=None, gpus=None, driver=None):
    """The release a channel currently points at.

    stable  -> releases/latest (a curated v0.x) -> its nightly-tag.txt -> that build.
    nightly -> newest build whose binary for this machine is fully uploaded
               (CI attaches assets over ~an hour, so the very newest release
               can be listed before its files exist).
    """
    if channel == "stable":
        latest = get_json(API + "/releases/latest")
        ptr = next((a for a in latest.get("assets") or []
                    if a.get("name") == "nightly-tag.txt"), None)
        if not ptr:
            return _norm(latest, channel)
        tag = get_text(ptr["browser_download_url"]).strip()
        if not re.match(r"^b\d+$", tag):
            raise RuntimeError(f"unexpected nightly-tag.txt contents: {tag[:40]!r}")
        return _norm(get_json(f"{API}/releases/tags/{tag}"), channel,
                     label=latest.get("tag_name"))
    if channel != "nightly":
        raise ValueError(f"unknown channel: {channel}")
    plat = plat or platform_key()
    for rel in get_json(API + "/releases?per_page=10"):
        if not str(rel.get("tag_name", "")).startswith("b"):
            continue
        if choose(rel.get("assets"), plat, gpus or [], driver)["bin"]:
            return _norm(rel, channel)
    raise RuntimeError("no recent llama.cpp build has binaries for this platform yet")


def detect_driver():
    out = osplat.run_text(["nvidia-smi"], timeout=10)
    return parse_cuda_driver(out)


# ------------------------------------------------------------------ installer

class Cancelled(Exception):
    pass


class Installer:
    """One install at a time on a background thread, with pollable state."""

    def __init__(self, root, log_dir, on_installed=None, detect=None):
        # on_installed(server_bin) points config at the new binary; injected so
        # this module stays free of config/router imports.
        self.engines = os.path.join(root, "engines", "llama.cpp")
        self.cudart_cache = os.path.join(root, "engines", "cudart")
        self.dl_dir = os.path.join(root, "engines", "_dl")
        self.on_installed = on_installed
        self.detect = detect            # () -> (plat, gpus, driver); tests stub it
        self.protected = lambda: []     # () -> install dirs pruning must keep
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, "engine-install.log")
        self.lock = threading.Lock()
        self.state = self._idle()

    @staticmethod
    def _idle():
        return {"running": False, "phase": "idle", "file": "", "downloaded": 0,
                "total": 0, "error": "", "tag": "", "variant": "", "server_bin": "",
                "cancel": False, "started": None, "finished": None}

    def _log(self, msg):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(msg.rstrip("\n") + "\n")

    def tail(self, n=80):
        try:
            with open(self.log_path, encoding="utf-8", errors="replace") as f:
                return "".join(f.readlines()[-n:])
        except OSError:
            return ""

    def progress(self):
        return dict(self.state)

    def cancel(self):
        with self.lock:
            if not self.state["running"]:
                return False
            self.state["cancel"] = True
            return True

    def start(self, channel="nightly", variant=None):
        if channel not in CHANNELS:
            raise ValueError(f"unknown channel: {channel}")
        with self.lock:
            if self.state["running"]:
                return False
            self.state = self._idle()
            self.state.update(running=True, phase="resolving", started=time.time())
        open(self.log_path, "w").close()
        threading.Thread(target=self._run, args=(channel, variant), daemon=True).start()
        return True

    # ---- steps ----
    def _download(self, asset):
        os.makedirs(self.dl_dir, exist_ok=True)
        dest = os.path.join(self.dl_dir, asset["name"])
        tmp = dest + ".part"
        have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        req = urllib.request.Request(asset["browser_download_url"],
                                     headers={"User-Agent": UA["User-Agent"]})
        if have:
            req.add_header("Range", f"bytes={have}-")
        self.state.update(phase="downloading", file=asset["name"],
                          total=int(asset.get("size") or 0), downloaded=0)
        self._log(f"downloading {asset['name']} ({int(asset.get('size') or 0)/1e6:.0f} MB)")
        with urllib.request.urlopen(req, timeout=60) as r:
            mode = "ab" if have and getattr(r, "status", 200) == 206 else "wb"
            self.state["downloaded"] = have if mode == "ab" else 0
            with open(tmp, mode) as f:
                while True:
                    if self.state["cancel"]:
                        raise Cancelled()
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    self.state["downloaded"] += len(chunk)
        self._verify(tmp, asset)
        os.replace(tmp, dest)
        return dest

    def _verify(self, path, asset):
        want = (asset.get("digest") or "").lower()
        self.state["phase"] = "verifying"
        if not want.startswith("sha256:"):
            self._log("[warn] GitHub published no digest for this asset; size check only")
            if asset.get("size") and os.path.getsize(path) != int(asset["size"]):
                os.remove(path)
                raise RuntimeError("download size mismatch")
            return
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != want.split(":", 1)[1]:
            os.remove(path)            # never resume from a corrupt .part
            raise RuntimeError(f"checksum mismatch for {asset['name']}")
        self._log(f"verified sha256 {want[7:19]}...")

    def _cudart_dir(self, asset):
        key = (asset.get("digest") or asset["name"]).split(":")[-1][:16]
        d = os.path.join(self.cudart_cache, key)
        if os.path.isdir(d) and os.listdir(d):
            self._log("cudart runtime already cached")
            return d
        arc = self._download(asset)
        self.state["phase"] = "extracting"
        safe_extract(arc, d + ".partial")
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
        os.replace(d + ".partial", d)
        os.remove(arc)
        return d

    @staticmethod
    def _link_tree(src, dst_dir):
        """Hard-link every runtime file from src into dst_dir (flattened next to
        llama-server, where the loader looks). Falls back to copying across
        volumes. Existing files win: the build's own libs are never replaced."""
        for dirpath, _, files in os.walk(src):
            for n in files:
                target = os.path.join(dst_dir, n)
                if os.path.exists(target):
                    continue
                s = os.path.join(dirpath, n)
                try:
                    os.link(s, target)
                except OSError:
                    shutil.copy2(s, target)

    def _smoke(self, sbin):
        self.state["phase"] = "testing"
        if not osplat.IS_WIN:
            os.chmod(sbin, os.stat(sbin).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        r = subprocess.run([sbin, "--version"], capture_output=True, text=True, timeout=120,
                           cwd=os.path.dirname(sbin))
        out = (r.stdout + r.stderr).strip()
        self._log(out[-1500:])
        if r.returncode != 0:
            raise RuntimeError(f"llama-server --version exited {r.returncode} - "
                               "this build does not run here (see log)")

    def _run(self, channel, variant):
        try:
            plat, gpus, driver = (self.detect() if self.detect else _detect())
            self._log(f"platform {plat[0]}-{plat[1]}, {len(gpus)} NVIDIA GPU(s), "
                      f"driver CUDA {'.'.join(map(str, driver)) if driver else 'n/a'}")
            rel = resolve(channel, plat=plat, gpus=gpus, driver=driver)
            pick = choose(rel["assets"], plat, gpus, driver, variant=variant)
            if not pick["bin"]:
                raise RuntimeError(pick["reason"] or "no matching build")
            self.state.update(tag=rel["tag"], variant=pick["variant"])
            self._log(f"{channel}: {rel['label']} -> {rel['tag']}, {pick['variant']} ({pick['reason']})")

            final = os.path.join(self.engines, f"{rel['tag']}-{pick['variant']}")
            sbin = find_server_bin(final) if os.path.isfile(os.path.join(final, MANIFEST)) else None
            if sbin:
                self._log("already installed - switching to it")
            else:
                arc = self._download(pick["bin"])
                self.state["phase"] = "extracting"
                partial = final + ".partial"
                shutil.rmtree(partial, ignore_errors=True)
                safe_extract(arc, partial)
                os.remove(arc)
                sbin = find_server_bin(partial)
                if not sbin:
                    raise RuntimeError("archive contains no llama-server")
                if pick["cudart"]:
                    self._link_tree(self._cudart_dir(pick["cudart"]), os.path.dirname(sbin))
                self._smoke(sbin)
                with open(os.path.join(partial, MANIFEST), "w", encoding="utf-8") as f:
                    json.dump({"tag": rel["tag"], "label": rel["label"], "channel": channel,
                               "variant": pick["variant"], "asset": pick["bin"]["name"],
                               "sha256": pick["bin"].get("digest", ""),
                               "published": rel["published"],
                               "installed_at": time.time()}, f, indent=1)
                shutil.rmtree(final, ignore_errors=True)
                os.replace(partial, final)
                sbin = find_server_bin(final)
            self.state["phase"] = "activating"
            if self.on_installed:
                self.on_installed(sbin)
            removed = prune_installs(self.engines, active_bin=sbin, protect=self.protected())
            for d in removed:
                self._log(f"pruned old install {os.path.basename(d)}")
            self.state.update(phase="done", server_bin=sbin)
            self._log(f"=== INSTALLED {rel['tag']} ({pick['variant']}) -> {sbin} ===")
        except Cancelled:
            self.state.update(phase="cancelled")
            self._log("=== CANCELLED ===")
        except Exception as e:
            self.state.update(phase="failed", error=str(e))
            self._log(f"=== FAILED: {e} ===")
        finally:
            for n in os.listdir(self.engines) if os.path.isdir(self.engines) else []:
                if n.endswith(".partial"):
                    shutil.rmtree(os.path.join(self.engines, n), ignore_errors=True)
            self.state.update(running=False, finished=time.time())


def _detect():
    import hardware
    gpus = hardware.detect_gpus()
    return platform_key(), gpus, (detect_driver() if gpus else None)
