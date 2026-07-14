"""Run a long WSL script (vLLM install/update) as a background job that streams
combined stdout+stderr to a log file the UI polls — same UX as builder.py.
"""
import os, threading, time

import wsl


class WslJob:
    def __init__(self, logdir, logname):
        self.logdir = logdir
        self.logname = logname
        self.log_path = os.path.join(logdir, logname)
        self.lock = threading.Lock()
        self.state = {"running": False, "phase": "idle", "returncode": None}

    def progress(self):
        return dict(self.state)

    def tail(self, n=300):
        if not os.path.exists(self.log_path):
            return ""
        with open(self.log_path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])

    def start(self, script, distro):
        with self.lock:
            if self.state["running"]:
                return False
            self.state.update(running=True, phase="running", returncode=None)
        threading.Thread(target=self._run, args=(script, distro), daemon=True).start()
        return True

    def _run(self, script, distro):
        os.makedirs(self.logdir, exist_ok=True)
        try:
            with open(self.log_path, "w", encoding="utf-8", errors="replace") as log:
                p = wsl.popen(script, stdout=log, stderr=log, distro=distro)
                rc = p.wait()
            self.state.update(phase="done" if rc == 0 else "failed", returncode=rc)
        except Exception as e:
            with open(self.log_path, "a", encoding="utf-8") as log:
                log.write(f"\n=== JOB FAILED: {e} ===\n")
            self.state.update(phase="failed", returncode=1)
        finally:
            self.state["running"] = False
