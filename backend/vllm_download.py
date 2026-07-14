"""Download safetensors repos into the WSL-side HuggingFace cache and manage
them from the dashboard (the 'hybrid' storage model: files live in WSL ext4 for
native load speed, but sizes/deletes are driven from the UI so the user never
opens a WSL shell).

Progress is polled by du -sb on the repo's cache dir vs the expected total.
Pure stdlib; all filesystem work happens inside WSL via wsl.py.
"""
import threading, time

import wsl

VENV = "~/.llamaforge/vllm-venv"
HF_CACHE = "~/.cache/huggingface/hub"


def cache_dirname(repo):
    """HF cache dir name for a repo, e.g. Qwen/Qwen3-8B -> models--Qwen--Qwen3-8B."""
    return "models--" + repo.replace("/", "--")


def download_cmd(repo):
    return f"{VENV}/bin/hf download {repo}"


def delete_cmd(repo):
    return f"rm -rf {HF_CACHE}/{cache_dirname(repo)}"


def _du_bytes(distro, repo):
    code, out, _err = wsl.run(
        f"du -sb {HF_CACHE}/{cache_dirname(repo)} 2>/dev/null | cut -f1",
        distro=distro, timeout=15)
    try:
        return int(out.strip()) if code == 0 and out.strip() else 0
    except ValueError:
        return 0


class Manager:
    """One download at a time, streamed to a log; progress via du polling."""
    def __init__(self, distro):
        self.distro = distro
        self.lock = threading.Lock()
        self.state = {"running": False, "repo": "", "downloaded": 0,
                      "total": 0, "phase": "idle", "error": ""}

    def progress(self):
        with self.lock:
            if self.state["running"]:
                self.state["downloaded"] = _du_bytes(self.distro, self.state["repo"])
            return dict(self.state)

    def start(self, repo, expected_bytes):
        with self.lock:
            if self.state["running"]:
                return False
            self.state = {"running": True, "repo": repo, "downloaded": 0,
                          "total": expected_bytes, "phase": "downloading", "error": ""}
        threading.Thread(target=self._run, args=(repo,), daemon=True).start()
        return True

    def _run(self, repo):
        import os
        try:
            logdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
            os.makedirs(logdir, exist_ok=True)
            with open(os.path.join(logdir, "vllm-download.log"), "w",
                      encoding="utf-8", errors="replace") as log:
                p = wsl.popen(download_cmd(repo), stdout=log, stderr=log, distro=self.distro)
                rc = p.wait()
            self.state["phase"] = "done" if rc == 0 else "failed"
            if rc != 0:
                self.state["error"] = "hf download failed - see log"
        except Exception as e:
            self.state.update(phase="failed", error=str(e))
        finally:
            self.state["running"] = False

    def delete(self, repo):
        code, _out, err = wsl.run(delete_cmd(repo), distro=self.distro, timeout=30)
        return code == 0, (err or "").strip()

    def wsl_path(self, repo):
        """The snapshot path vLLM can serve directly (skips a re-resolve)."""
        return f"{HF_CACHE}/{cache_dirname(repo)}"
