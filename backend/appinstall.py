"""Install or update LlamaForge from an extracted release, in place.

The one-line installers (install.ps1 / install.sh) download a release archive,
extract it to a temp dir and hand it to this module, so the part that can
destroy data is plain stdlib Python with tests instead of shell.

Rules:
* Every file in the release is copied over the install dir.
* Files the *previous* release installed but this one doesn't ship are
  removed; the manifest (`.lf-files.json`) is how we know which those are.
  Anything not in a manifest (the user's own files) is never deleted.
* User data names (config, registry, engines, logs, the private Python) are
  never written or deleted, even if a release archive contains them.
"""
import argparse, json, os, shutil, sys

MANIFEST = ".lf-files.json"
# Top-level names that belong to the user, not to a release.
USER_TOP = {"config.json", "models.ini", "models-ikllama.ini", "stats.json",
            "vllm_models.json", "engines", "logs", "models", "wiki", "llama.cpp",
            "python", MANIFEST}
# Paths in config.example.json that are placeholders, not real installs.
_PLACEHOLDER_KEYS = ("llama_src", "build_dir", "server_bin")
SKIP_DIRS = {"__pycache__", ".git"}


def archive_root(extracted):
    """GitHub archives wrap everything in one `LlamaForge-<ref>/` folder."""
    if os.path.isfile(os.path.join(extracted, "backend", "server.py")):
        return extracted
    entries = os.listdir(extracted)
    if len(entries) == 1 and os.path.isdir(os.path.join(extracted, entries[0])):
        return os.path.join(extracted, entries[0])
    return extracted


def _safe_rel(rel):
    """A manifest/release path as a normalised relative posix path, or None
    if it is absolute, escapes the install dir, or names user data."""
    if not isinstance(rel, str) or not rel or os.path.isabs(rel) or "\\" in rel \
            or rel.startswith("/") or ":" in rel:
        return None
    parts = [p for p in rel.split("/") if p not in ("", ".")]
    if not parts or ".." in parts or parts[0] in USER_TOP \
            or any(p in SKIP_DIRS for p in parts):
        return None
    return "/".join(parts)


def release_files(src):
    out = []
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            rel = os.path.relpath(os.path.join(dirpath, name), src).replace(os.sep, "/")
            rel = _safe_rel(rel)
            if rel:
                out.append(rel)
    return sorted(out)


def _read_manifest(dest):
    try:
        with open(os.path.join(dest, MANIFEST), encoding="utf-8") as f:
            m = json.load(f)
        return m if isinstance(m, dict) else {}
    except (OSError, ValueError):
        return {}


def installed_version(dest):
    return _read_manifest(dest).get("version")


def _prune_empty(dest, rel):
    d = os.path.dirname(os.path.join(dest, *rel.split("/")))
    root = os.path.abspath(dest)
    while os.path.abspath(d) != root:
        try:
            os.rmdir(d)          # only succeeds when empty
        except OSError:
            return
        d = os.path.dirname(d)


def install(src, dest, version=None):
    """Lay release dir `src` over install dir `dest`. Returns what changed."""
    src = archive_root(src)
    if not os.path.isfile(os.path.join(src, "backend", "server.py")):
        raise ValueError(f"{src} is not a LlamaForge release (no backend/server.py)")
    files = release_files(src)
    old = {r for r in (_safe_rel(x) for x in _read_manifest(dest).get("files", [])) if r}
    os.makedirs(dest, exist_ok=True)
    added, updated = [], []
    for rel in files:
        target = os.path.join(dest, *rel.split("/"))
        (updated if os.path.exists(target) else added).append(rel)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        tmp = target + ".lf-new"
        shutil.copy2(os.path.join(src, *rel.split("/")), tmp)
        os.replace(tmp, target)          # never a half-written file in place
    removed = []
    for rel in sorted(old - set(files)):
        target = os.path.join(dest, *rel.split("/"))
        if os.path.isfile(target):
            os.remove(target)
            removed.append(rel)
        _prune_empty(dest, rel)
    with open(os.path.join(dest, MANIFEST), "w", encoding="utf-8") as f:
        json.dump({"version": version, "files": files}, f, indent=1)
    return {"added": added, "updated": updated, "removed": removed}


def ensure_config(dest):
    """First install: config.json from the example, minus its placeholder
    llama.cpp paths, so the panel offers the one-click engine instead of
    pointing at C:/path/to/llama.cpp. Returns True if it wrote one."""
    path = os.path.join(dest, "config.json")
    if os.path.exists(path):
        return False
    with open(os.path.join(dest, "config.example.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    for k in _PLACEHOLDER_KEYS:
        cfg[k] = ""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description="Install/update LlamaForge from an extracted release")
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--to", dest="dest", required=True)
    ap.add_argument("--version", default=None)
    a = ap.parse_args(argv)
    try:
        r = install(a.src, a.dest, a.version)
        ensure_config(a.dest)
    except (ValueError, OSError) as e:
        print(f"install failed: {e}", file=sys.stderr)
        return 1
    print(f"LlamaForge {a.version or ''} -> {a.dest}: "
          f"{len(r['added'])} added, {len(r['updated'])} updated, {len(r['removed'])} removed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
