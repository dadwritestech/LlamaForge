"""Try a llama.cpp pull request: build ggml-org/llama.cpp PR #N beside the
user's own build and install it as a pin target.

The build lands in ENGINES_DIR like a prebuilt install (lf-install.json with
channel "pr"), so builds.options / pinned_bin already let a model run on it in
its own process. It never touches llama_src, build_dir or the router's binary:
the work dir is a fresh shallow clone of just that PR's head, and the build is
pinned to the SHA the user was shown (a force-push in between aborts). The
functions are pure; PrBuilder at the bottom does the I/O on a thread.
"""
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.request

import logfiles
import prebuilt
from builder import BuildManager, toolset_args  # noqa: F401 - prtry.toolset_args

REPO = "https://github.com/ggml-org/llama.cpp"
API = "https://api.github.com/repos/ggml-org/llama.cpp/pulls/"
CHANNEL = "pr"
TARGETS = ("llama-server", "llama-bench")

_NUM = re.compile(r"#?\s*(\d{1,7})")
# ggerganov/llama.cpp redirects to ggml-org/llama.cpp, so old links still work
_URL = re.compile(r"(?:https?://)?(?:www\.)?github\.com/(?:ggml-org|ggerganov)/llama\.cpp/pull/(\d{1,7})(?:[/?#].*)?",
                  re.IGNORECASE)
_SHA = re.compile(r"[0-9a-f]{40}")


def parse_pr(text):
    """The PR number in "1234", "#1234" or a ggml-org/llama.cpp pull URL.
    ValueError for anything else, including other repos."""
    if not isinstance(text, str):
        raise ValueError("type a PR number or a github.com/ggml-org/llama.cpp/pull/... link")
    s = text.strip()
    m = _URL.fullmatch(s) or _NUM.fullmatch(s)
    if not m or int(m.group(1)) <= 0:
        raise ValueError("type a PR number or a github.com/ggml-org/llama.cpp/pull/... link")
    return int(m.group(1))


def pr_info(d):
    """GitHub's pull JSON -> the fields the panel shows. ValueError when it
    isn't a PR (e.g. {"message": "Not Found"}) or the SHA looks wrong."""
    try:
        n, sha = int(d["number"]), d["head"]["sha"]
    except (KeyError, TypeError, ValueError):
        raise ValueError(d.get("message") if isinstance(d, dict) and d.get("message")
                         else "that isn't a llama.cpp pull request")
    if not isinstance(sha, str) or not _SHA.fullmatch(sha):
        raise ValueError("GitHub sent an odd commit id for that PR")
    state = "merged" if d.get("merged_at") else ("open" if d.get("state") == "open" else "closed")
    return {"number": n, "title": str(d.get("title") or "").strip()[:200],
            "author": str((d.get("user") or {}).get("login") or ""),
            "sha": sha, "state": state, "draft": bool(d.get("draft")),
            "url": f"{REPO}/pull/{n}"}


def install_name(n, sha):
    return f"pr-{int(n)}-{sha[:7]}"


def commands(n, sha, work, build_dir, flags, jobs=8, toolset=()):
    """argv lists, run in order (never through a shell). check_head() goes
    between the checkout and the configure."""
    cfg = ["cmake", "-B", build_dir, "-S", work, "-DCMAKE_BUILD_TYPE=Release",
           "-DLLAMA_CURL=OFF"]
    cfg += [f"-D{k}={v}" for k, v in (flags or {}).items()]
    cfg += list(toolset)
    return [
        ["git", "init", "-q", work],
        ["git", "-C", work, "fetch", "-q", "--depth", "1", REPO, f"pull/{int(n)}/head"],
        ["git", "-C", work, "checkout", "-q", "--detach", "FETCH_HEAD"],
        cfg,
        ["cmake", "--build", build_dir, "--config", "Release", "--parallel", str(int(jobs)),
         "--target", *TARGETS],
    ]


def check_head(rev_parse_out, sha):
    """'' when the fetched head is the commit the user looked at."""
    got = (rev_parse_out or "").strip()
    return "" if got == sha else (f"the PR changed since you looked (now {got[:7] or '?'}, "
                                  f"was {sha[:7]}); look it up again")


def manifest(info, flags, now=None):
    n = info["number"]
    return {"tag": f"PR #{n}", "label": f"PR #{n}: {info['title']}".rstrip(": "),
            "channel": CHANNEL, "pr": n, "sha": info["sha"], "author": info["author"],
            "url": info["url"], "flags": dict(flags or {}),
            "installed_at": time.time() if now is None else now}


def is_pr(install):
    return (install or {}).get("channel") == CHANNEL


# what failed, by commands() index
_STEP = ("git init failed", "git fetch failed: is that an open llama.cpp PR?",
         "git checkout failed", "cmake configure failed", "cmake build failed")
_PHASE = ("fetch", "fetch", "fetch", "configure", "build")


def _fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "LlamaForge",
                                               "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))


class PrBuilder:
    """One PR build at a time: look() it up, start() the build on a thread,
    poll status(), cancel(), remove() an installed PR build. run, capture and
    fetch_json are injectable so tests never touch git, cmake or GitHub."""

    def __init__(self, engines, work_root, log_dir, run=None, capture=None, fetch_json=None):
        self.engines, self.work_root = engines, work_root
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, "build-pr.log")
        self._run_cmd = run or self._stream
        self._capture = capture or (lambda cmd: subprocess.run(
            cmd, capture_output=True, text=True, timeout=60).stdout)
        self._fetch = fetch_json or _fetch_json
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._proc = None
        self._thread = None
        self._st = {"running": False, "phase": "idle", "pr": None, "title": "",
                    "error": "", "install": "", "started": None, "finished": None}

    def look(self, text):
        n = parse_pr(text)
        try:
            d = self._fetch(API + str(n))
        except Exception as e:
            raise ValueError(f"couldn't reach GitHub for PR #{n}: {e}")
        return pr_info(d)

    def status(self):
        with self._lock:
            return dict(self._st)

    def tail(self, n=200):
        return "".join(logfiles.tail_lines(self.log_path, n))

    def start(self, info, flags, jobs=None):
        with self._lock:
            if self._st["running"]:
                return False
            self._cancel.clear()
            self._st = {"running": True, "phase": "fetch", "pr": info["number"],
                        "title": info["title"], "error": "", "install": "",
                        "started": time.time(), "finished": None}
            self._thread = threading.Thread(target=self._build, daemon=True, name="prtry",
                                            args=(info, dict(flags or {}), jobs or os.cpu_count() or 8))
            self._thread.start()
        return True

    def cancel(self):
        self._cancel.set()
        p = self._proc
        if p and p.poll() is None:
            p.terminate()

    def join(self, timeout=None):
        t = self._thread
        if t:
            t.join(timeout)

    def remove(self, name):
        """Delete an installed PR build by its dir name. ValueError for
        anything that isn't one (released builds, paths)."""
        inst = next((i for i in prebuilt.list_installs(self.engines)
                     if os.path.basename(i["dir"]) == name), None) if isinstance(name, str) else None
        if not inst or not is_pr(inst):
            raise ValueError("no such PR build")
        shutil.rmtree(inst["dir"], ignore_errors=True)
        return inst["dir"]

    # ---- the thread ----
    def _log(self, msg):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(msg.rstrip("\n") + "\n")

    def _stream(self, cmd, cwd=None):
        self._log("$ " + " ".join(cmd))
        self._proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, bufsize=1,
                                      errors="replace")
        for line in self._proc.stdout:
            self._log(line)
        return self._proc.wait()

    def _set(self, **kw):
        with self._lock:
            self._st.update(kw)

    def _build(self, info, flags, jobs):
        n, sha = info["number"], info["sha"]
        work = os.path.join(self.work_root, f"pr-{n}")
        name = install_name(n, sha)
        open(self.log_path, "w").close()
        self._log(f"=== PR #{n}: {info['title']} by {info['author']} @ {sha[:7]} ===")
        try:
            shutil.rmtree(work, ignore_errors=True)
            os.makedirs(self.work_root, exist_ok=True)
            bdir = os.path.join(work, "build")
            ts = toolset_args(flags, os.environ, os.name)
            for i, cmd in enumerate(commands(n, sha, work, bdir, flags, jobs, ts)):
                if self._cancel.is_set():
                    return self._set(phase="cancelled")
                if i == 3:               # between checkout and configure
                    bad = check_head(self._capture(["git", "-C", work, "rev-parse", "HEAD"]), sha)
                    if bad:
                        raise RuntimeError(bad)
                self._set(phase=_PHASE[i])
                if self._run_cmd(cmd) != 0:
                    if self._cancel.is_set():
                        return self._set(phase="cancelled")
                    raise RuntimeError(_STEP[i])
            if self._cancel.is_set():
                return self._set(phase="cancelled")
            self._set(phase="install")
            src = BuildManager.binaries_dir(bdir)
            if not src or not prebuilt.find_server_bin(src):
                raise RuntimeError("the build made no llama-server")
            final = os.path.join(self.engines, name)
            part = final + ".partial"
            shutil.rmtree(part, ignore_errors=True)
            shutil.copytree(src, part)
            with open(os.path.join(part, prebuilt.MANIFEST), "w", encoding="utf-8") as f:
                json.dump(manifest(info, flags), f, indent=1)
            shutil.rmtree(final, ignore_errors=True)
            os.replace(part, final)
            self._log(f"=== installed as {name} ===")
            self._set(phase="done", install=name)
        except Exception as e:
            self._log(f"=== FAILED: {e} ===")
            self._set(phase="failed", error=str(e))
        finally:
            self._proc = None
            shutil.rmtree(work, ignore_errors=True)     # sources + objects are GBs
            self._set(running=False, finished=time.time())
