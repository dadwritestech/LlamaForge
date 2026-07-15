"""Discover and download GGUF models from huggingface.co.

Search the HF Hub, list a repo's GGUF files with sizes, rate each against the
machine's VRAM, and stream downloads in a background thread with progress.
Downloads go through Python (this works even though the llama.cpp build has
no SSL support).
"""
import json, os, re, threading, time, urllib.request, urllib.parse

HF = "https://huggingface.co"
UA = {"User-Agent": "LlamaForge/1.0 (+local model manager)"}

def _get_json(url, timeout=25):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

# GGUF runs everywhere llama.cpp builds.
PLATFORMS = ["windows", "linux", "macos"]

def search(query="", sort="downloads", limit=50):
    """Search GGUF repos. sort: downloads | lastModified | likes."""
    params = {"filter": "gguf", "limit": str(limit), "direction": "-1", "sort": sort,
              # the list API omits lastModified/gated unless asked explicitly
              "expand[]": ["downloads", "likes", "lastModified", "gated"]}
    if query:
        params["search"] = query
    url = f"{HF}/api/models?{urllib.parse.urlencode(params, doseq=True)}"
    out = []
    for m in _get_json(url):
        out.append({
            "repo": m.get("id", ""),
            "downloads": m.get("downloads", 0),
            "likes": m.get("likes", 0),
            "updated": (m.get("lastModified") or "")[:10],
            "gated": bool(m.get("gated")),   # "auto"/"manual" -> needs an HF token
            "platforms": PLATFORMS,
        })
    return out

def _fit(size_bytes, vram_mib):
    """Rate a file against total VRAM: fits / tight / cpu-offload."""
    if not vram_mib:
        return "unknown"
    vram = vram_mib * 1024 * 1024
    if size_bytes * 1.15 <= vram:       # weights + KV/compute headroom
        return "fits"
    if size_bytes <= vram:
        return "tight"
    return "offload"

def files(repo, vram_mib=0):
    """List a repo's GGUF files with size + fit rating. Collapses shard sets."""
    tree = _get_json(f"{HF}/api/models/{repo}/tree/main")
    ggufs = [f for f in tree if f.get("path", "").lower().endswith(".gguf")]
    shard_totals, singles, mmproj = {}, [], []
    for f in ggufs:
        p, size = f["path"], f.get("size", 0)
        if os.path.basename(p).lower().startswith("mmproj"):
            mmproj.append({"path": p, "size": size})
            continue
        m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", p, re.I)
        if m:
            key = re.sub(r"-\d{5}-of-\d{5}\.gguf$", "", p, flags=re.I)
            agg = shard_totals.setdefault(key, {"size": 0, "first": None, "n": int(m.group(2))})
            agg["size"] += size
            if m.group(1) == "00001":
                agg["first"] = p
        else:
            singles.append({"path": p, "size": size})
    out = []
    for f in singles:
        out.append({"path": f["path"], "size": f["size"], "shards": 1,
                    "fit": _fit(f["size"], vram_mib)})
    for key, agg in shard_totals.items():
        if agg["first"]:
            out.append({"path": agg["first"], "size": agg["size"], "shards": agg["n"],
                        "fit": _fit(agg["size"], vram_mib)})
    out.sort(key=lambda x: x["size"])
    return {"files": out, "mmproj": sorted(mmproj, key=lambda x: x["size"])}

def shard_paths(first_path, n):
    """All shard file paths given the first shard's path."""
    if n <= 1:
        return [first_path]
    base = re.sub(r"-\d{5}-of-\d{5}\.gguf$", "", first_path, flags=re.I)
    return [f"{base}-{i:05d}-of-{n:05d}.gguf" for i in range(1, n + 1)]

class Cancelled(Exception):
    """User pressed Cancel; unwinds the download thread cleanly (drops .part)."""


class Paused(Exception):
    """User pressed Pause; unwinds the thread but KEEPS the .part for resume."""


class DownloadManager:
    """One download job at a time, streamed with progress. Cancel and pause are
    cooperative flags the chunk loop checks. Pause leaves the partial `.part`
    file in place; a later start() with the same paths resumes it via an HTTP
    Range request, so a 25GB transfer never restarts from zero."""
    def __init__(self):
        self.lock = threading.Lock()
        self._job = None       # (repo, paths, dest_dir) - kept so resume() can rerun
        self.state = self._idle_state()

    @staticmethod
    def _idle_state():
        return {"running": False, "repo": "", "file": "", "done_files": 0,
                "total_files": 0, "downloaded": 0, "total": 0, "cancel": False,
                "paused": False, "error": "", "finished_path": "", "phase": "idle"}

    def progress(self):
        return dict(self.state)

    def cancel(self):
        """Request cancellation of the running job. Returns whether one ran."""
        with self.lock:
            if not self.state["running"]:
                return False
            self.state["cancel"] = True
            return True

    def pause(self):
        """Request a pause of the running job (partial file is kept). Returns
        whether one ran."""
        with self.lock:
            if not self.state["running"]:
                return False
            self.state["paused"] = True
            return True

    def resume(self):
        """Restart a paused job from where it left off. Returns whether one
        was resumed."""
        with self.lock:
            if self.state["running"] or self.state.get("phase") != "paused" or not self._job:
                return False
            repo, paths, dest_dir = self._job
            self.state.update(running=True, cancel=False, paused=False, phase="starting")
        threading.Thread(target=self._run, args=(repo, paths, dest_dir), daemon=True).start()
        return True

    def _check_cancel(self):
        if self.state.get("cancel"):
            raise Cancelled()

    def _check_signals(self):
        if self.state.get("cancel"):
            raise Cancelled()
        if self.state.get("paused"):
            raise Paused()

    def _fetch(self, url, dest):
        tmp = dest + ".part"
        have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        req = urllib.request.Request(url, headers=UA)
        if have:
            req.add_header("Range", f"bytes={have}-")
        with urllib.request.urlopen(req, timeout=60) as r:
            resumed = have and getattr(r, "status", 200) == 206
            clen = int(r.headers.get("Content-Length") or 0)
            if resumed:                       # server continues from `have`
                self.state["total"] = have + clen
                self.state["downloaded"] = have
                mode = "ab"
            else:                             # range ignored/absent -> start over
                have = 0
                self.state["total"] = clen
                self.state["downloaded"] = 0
                mode = "wb"
            try:
                with open(tmp, mode) as f:
                    while True:
                        self._check_signals()
                        chunk = r.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        self.state["downloaded"] += len(chunk)
            except Paused:
                raise                          # keep .part so resume() can continue
            except Cancelled:
                try: os.remove(tmp)            # never leave a poisoned .part behind
                except OSError: pass
                raise
            os.replace(tmp, dest)

    def _run(self, repo, paths, dest_dir):
        try:
            os.makedirs(dest_dir, exist_ok=True)
            self.state.update(total_files=len(paths), phase="downloading")
            final = self.state.get("finished_path") or ""
            for i, p in enumerate(paths):
                self._check_signals()
                self.state.update(file=os.path.basename(p), done_files=i)
                url = f"{HF}/{repo}/resolve/main/{urllib.parse.quote(p)}"
                dest = os.path.join(dest_dir, os.path.basename(p))
                if os.path.exists(dest):   # already downloaded; skip
                    if not final: final = dest
                    continue
                self._fetch(url, dest)
                if not final: final = dest
            self.state.update(done_files=len(paths), phase="done",
                              finished_path=final)
        except Paused:
            self.state.update(phase="paused", error="")
        except Cancelled:
            self.state.update(phase="cancelled", error="")
        except Exception as e:
            self.state.update(phase="failed", error=str(e))
        finally:
            self.state["running"] = False

    def start(self, repo, paths, dest_dir):
        with self.lock:
            if self.state["running"]:
                return False
            self._job = (repo, paths, dest_dir)
            self.state = self._idle_state()
            self.state.update(running=True, repo=repo, total_files=len(paths),
                              phase="starting")
        threading.Thread(target=self._run, args=(repo, paths, dest_dir), daemon=True).start()
        return True

if __name__ == "__main__":
    print(json.dumps(search("qwen coder", limit=5), indent=1))
