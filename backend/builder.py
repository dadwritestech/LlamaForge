"""Track the llama.cpp GitHub repo and (re)build it via CMake.

Runs the build in a background thread, streaming output to a log file the UI
polls. Backs up prior binaries before overwriting so a bad build is reversible.
"""
import os, shutil, subprocess, threading, time, datetime, re

class BuildManager:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, "build.log")
        self.lock = threading.Lock()
        self.state = {"running": False, "phase": "idle", "returncode": None,
                      "started": None, "finished": None}

    # ---- git introspection ----
    def _git(self, src, *args, timeout=60):
        return subprocess.run(["git", "-C", src, *args],
                              capture_output=True, text=True, timeout=timeout)

    def current_commit(self, src):
        if not src or not os.path.isdir(os.path.join(src, ".git")):
            return {"ok": False, "error": "not a git checkout"}
        r = self._git(src, "log", "-1", "--pretty=%h|%s|%ci")
        if r.returncode != 0:
            return {"ok": False, "error": r.stderr.strip()}
        h, s, d = (r.stdout.strip().split("|", 2) + ["", "", ""])[:3]
        br = self._git(src, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        return {"ok": True, "hash": h, "subject": s, "date": d, "branch": br}

    def check_updates(self, src, remote_branch="origin/master"):
        if not src:
            return {"ok": False, "error": "no source dir"}
        try:
            self._git(src, "fetch", "--quiet", "origin", timeout=120)
        except Exception as e:
            return {"ok": False, "error": f"fetch failed: {e}"}
        cnt = self._git(src, "rev-list", "--count", f"HEAD..{remote_branch}").stdout.strip()
        latest = self._git(src, "log", "-1", "--pretty=%h|%s", remote_branch).stdout.strip()
        lh, ls = (latest.split("|", 1) + ["", ""])[:2]
        try: behind = int(cnt)
        except ValueError: behind = 0
        return {"ok": True, "behind": behind,
                "latest": {"hash": lh, "subject": ls}, "up_to_date": behind == 0}

    # ---- build ----
    def _log(self, msg):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(msg if msg.endswith("\n") else msg + "\n")

    def tail(self, n=200):
        if not os.path.exists(self.log_path):
            return ""
        with open(self.log_path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])

    @staticmethod
    def binaries_dir(build_dir, isdir=os.path.isdir):
        """Where built binaries land: bin/Release with MSVC's multi-config
        generator, plain bin/ with Ninja/Make (Linux, macOS)."""
        rel = os.path.join(build_dir, "bin", "Release")
        if isdir(rel):
            return rel
        flat = os.path.join(build_dir, "bin")
        return flat if isdir(flat) else None

    def backup_binaries(self, build_dir):
        src = self.binaries_dir(build_dir)
        if src:
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            dst = src.rstrip("/\\") + f"-backup-{stamp}"
            try:
                shutil.copytree(src, dst)
                self._log(f"[backup] prior binaries -> {dst}")
            except Exception as e:
                self._log(f"[backup] skipped: {e}")

    def _stream(self, cmd, cwd=None):
        self._log(f"\n$ {' '.join(cmd)}")
        p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            self._log(line.rstrip("\n"))
        p.wait()
        return p.returncode

    def run_build(self, src, build_dir, flags, pull=True, jobs=None):
        """Blocking build; call inside a thread. flags: {CMAKE_VAR: value}."""
        with self.lock:
            if self.state["running"]:
                return
            self.state.update(running=True, phase="starting", returncode=None,
                              started=time.time(), finished=None)
        open(self.log_path, "w").close()  # fresh log
        try:
            if pull:
                self.state["phase"] = "pull"
                self._log("=== git pull (origin) ===")
                rc = self._stream(["git", "-C", src, "pull", "--ff-only", "origin"])
                if rc != 0:
                    self._log("[warn] git pull failed or non-fast-forward; building current checkout")

            self.state["phase"] = "backup"
            self.backup_binaries(build_dir)

            self.state["phase"] = "configure"
            self._log("\n=== cmake configure ===")
            cfg = ["cmake", "-B", build_dir, "-S", src,
                   "-DCMAKE_BUILD_TYPE=Release"]
            for k, v in (flags or {}).items():
                cfg.append(f"-D{k}={v}")
            rc = self._stream(cfg)
            if rc != 0:
                raise RuntimeError("cmake configure failed")

            self.state["phase"] = "build"
            self._log("\n=== cmake build ===")
            jobs = jobs or os.cpu_count() or 8
            rc = self._stream(["cmake", "--build", build_dir, "--config", "Release",
                               "--parallel", str(jobs)])
            if rc != 0:
                raise RuntimeError("cmake build failed")

            self.state.update(phase="done", returncode=0)
            self._log("\n=== BUILD OK ===")
        except Exception as e:
            self.state.update(phase="failed", returncode=1)
            self._log(f"\n=== BUILD FAILED: {e} ===")
        finally:
            self.state.update(running=False, finished=time.time())

    def start(self, src, build_dir, flags, pull=True, jobs=None):
        if self.state["running"]:
            return False
        t = threading.Thread(target=self.run_build,
                             args=(src, build_dir, flags, pull, jobs), daemon=True)
        t.start()
        return True
