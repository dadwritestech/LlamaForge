"""Track the llama.cpp GitHub repo and (re)build it via CMake.

Runs the build in a background thread, streaming output to a log file the UI
polls. Backs up prior binaries before overwriting so a bad build is reversible.
"""
import os, shutil, subprocess, threading, time, datetime, re
import logfiles

UPDATE_TTL      = 900   # seconds a successful upstream check stays cached
UPDATE_TTL_FAIL = 60    # failed fetches retry sooner, but never per-click


def toolset_args(flags, env, os_name, isdir=os.path.isdir):
    """["-T", "cuda=<CUDA_PATH>", "-DCUDAToolkit_ROOT=<CUDA_PATH>"] for a CUDA build with a Visual Studio
    generator (CMake's default on Windows). With two CUDA toolkits installed,
    a fresh build dir got nvcc from PATH but MSBuild's CUDA targets from the
    other one, mixed their headers, and failed. -T pins nvcc; CUDAToolkit_ROOT
    pins the headers ggml finds before CUDA is enabled. Used for PR builds and for
    a source build into a dir with no CMake cache yet."""
    gen = (env.get("CMAKE_GENERATOR") or "").strip()
    cuda = (env.get("CUDA_PATH") or "").strip()
    if (os_name != "nt" or str((flags or {}).get("GGML_CUDA", "")).upper() != "ON"
            or (gen and not gen.startswith("Visual Studio")) or not cuda or not isdir(cuda)):
        return []
    return ["-T", f"cuda={cuda}", f"-DCUDAToolkit_ROOT={cuda}"]


def _toolset(flags):
    return toolset_args(flags, os.environ, os.name)


class BuildManager:
    def __init__(self, log_dir, log_name="build", on_built=None):
        # on_built(path) fires once, after a successful build, with the located
        # llama-server. Injected rather than imported so builder.py stays free of
        # config.py: one BuildManager exists per engine and only the caller knows
        # which config key a given instance owns.
        self.on_built = on_built
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"{log_name}.log")
        self.lock = threading.Lock()
        self.state = {"running": False, "phase": "idle", "returncode": None,
                      "started": None, "finished": None}
        self._upd_cache = {}   # (src, remote_branch) -> (checked_at, result)

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

    def check_updates(self, src, remote_branch=None, force=False, now=None):
        """Upstream check with a TTL cache: opening the Build tab must not
        `git fetch` GitHub on every click. force=True bypasses the cache
        (the UI's Refresh button); a finished build clears it. remote_branch
        None means the checkout's own default (see resolve_remote_branch)."""
        if not src:
            return {"ok": False, "error": "no source dir"}
        now = time.time() if now is None else now
        key = (src, remote_branch)
        hit = self._upd_cache.get(key)
        if not force and hit:
            age = now - hit[0]
            if age < (UPDATE_TTL if hit[1].get("ok") else UPDATE_TTL_FAIL):
                return dict(hit[1], cached=True, checked_secs_ago=int(age))
        res = self._check_updates_fresh(src, remote_branch)
        self._upd_cache[key] = (now, res)
        return dict(res, cached=False, checked_secs_ago=0)

    def resolve_remote_branch(self, src):
        """The remote's default branch as "origin/<name>", or None.

        llama.cpp's default is master but ik_llama.cpp's is main, so a hardcoded
        origin/master made `rev-list` fail on the latter. Ask the checkout first
        (origin/HEAD), then probe main and master for clones that lack it.
        """
        r = self._git(src, "symbolic-ref", "--short", "refs/remotes/origin/HEAD")
        cands = ([r.stdout.strip()] if r.returncode == 0 else []) + ["origin/main", "origin/master"]
        for c in cands:
            if self._git(src, "rev-parse", "--verify", "--quiet", f"refs/remotes/{c}").returncode == 0:
                return c
        return None

    def _check_updates_fresh(self, src, remote_branch=None):
        try:
            self._git(src, "fetch", "--quiet", "origin", timeout=120)
            remote_branch = remote_branch or self.resolve_remote_branch(src)
            if not remote_branch:
                return {"ok": False, "error": "no origin/HEAD, origin/main or origin/master "
                                              "in this checkout; cannot tell what upstream is"}
            cnt = self._git(src, "rev-list", "--count", f"HEAD..{remote_branch}")
            latest = self._git(src, "log", "-1", "--pretty=%h|%s", remote_branch)
        except Exception as e:
            return {"ok": False, "error": f"fetch failed: {e}"}
        for r in (cnt, latest):
            if r.returncode != 0:
                # An unreadable count is not "0 behind": say so instead of
                # reporting up to date forever.
                return {"ok": False, "error": r.stderr.strip() or f"git failed against {remote_branch}"}
        lh, ls = (latest.stdout.strip().split("|", 1) + ["", ""])[:2]
        behind = int(cnt.stdout.strip())
        return {"ok": True, "behind": behind, "branch": remote_branch,
                "latest": {"hash": lh, "subject": ls}, "up_to_date": behind == 0}

    # ---- build ----
    def _log(self, msg):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(msg if msg.endswith("\n") else msg + "\n")

    def tail(self, n=200):
        return "".join(logfiles.tail_lines(self.log_path, n))

    @staticmethod
    def binaries_dir(build_dir, isdir=os.path.isdir):
        """Where built binaries land: bin/Release with MSVC's multi-config
        generator, plain bin/ with Ninja/Make (Linux, macOS)."""
        rel = os.path.join(build_dir, "bin", "Release")
        if isdir(rel):
            return rel
        flat = os.path.join(build_dir, "bin")
        return flat if isdir(flat) else None

    @staticmethod
    def locate_server_bin(build_dir, isdir=os.path.isdir, isfile=os.path.isfile):
        """Where this build actually put llama-server, or None.

        bootstrap has to guess this path before anything is built, and it guesses
        the Ninja/Make layout (bin/llama-server). MSVC's multi-config generator
        puts it in bin/Release with an .exe suffix instead, so on Windows the
        guess was always wrong and had to be corrected by hand in config.json
        after every build. Both names are probed rather than keying off the host
        OS, so a cross-build or a WSL-built tree resolves the same way.
        """
        if not build_dir:
            return None
        d = BuildManager.binaries_dir(build_dir, isdir=isdir)
        if not d:
            return None
        for name in ("llama-server.exe", "llama-server"):
            p = os.path.join(d, name)
            if isfile(p):
                return p
        return None

    @staticmethod
    def validate_paths(src, build_dir, isdir=os.path.isdir):
        """Why the build can't run, or "" if it can.

        On a fresh install llama_src/build_dir are deliberately blank, so Rebuild
        used to run `cmake -B  -S ` and surface CMake's own "No build directory
        specified for -B" - an error about our bug, phrased as if the user had
        typed something wrong. Check here and say what's actually missing.

        build_dir is not required to exist: cmake creates it. Only the source
        tree has to be there already.
        """
        if not (src or "").strip():
            return "llama.cpp source directory is not set - set it in Setup first."
        if not (build_dir or "").strip():
            return "Build directory is not set - set it in Setup first."
        if not isdir(src):
            return f"llama.cpp source directory does not exist: {src}"
        return ""

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
                              started=time.time(), finished=None, warning=None)
        open(self.log_path, "w").close()  # fresh log
        try:
            bad = self.validate_paths(src, build_dir)
            if bad:
                raise RuntimeError(bad)
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
            # a cached dir keeps its toolset: CMake refuses to switch it
            if not os.path.isfile(os.path.join(build_dir, "CMakeCache.txt")):
                cfg += _toolset(flags)
            rc = self._stream(cfg)
            if rc != 0:
                raise RuntimeError("cmake configure failed")

            self.state["phase"] = "build"
            self._log("\n=== cmake build ===")
            jobs = jobs or os.cpu_count() or 8
            rc = self._stream(["cmake", "--build", build_dir, "--config", "Release",
                               "--parallel", str(jobs)])
            if rc != 0:
                # cmake returns one code for the whole build. A non-essential
                # later target (npm/sharp UI assets) can fail after llama-server
                # already built - reporting BUILD FAILED then is wrong. But only
                # a binary from THIS run counts: a stale one from a previous
                # success would hide a real compile failure, so freshness is
                # proven by mtime, not assumed from mere presence.
                fresh = self._fresh_binary(build_dir, self.state.get("started") or 0)
                if not fresh:
                    raise RuntimeError("cmake build failed")
                self.state.update(phase="done_warnings", returncode=0,
                                  warning="llama-server built, but a later build "
                                          "step failed - see the log.")
                self._log("\n=== BUILD OK (with warnings) ===")
                self._record_built(fresh)
                return

            self.state.update(phase="done", returncode=0)
            self._log("\n=== BUILD OK ===")
            found = self.locate_server_bin(build_dir)
            if found:
                self._record_built(found)
        except Exception as e:
            self.state.update(phase="failed", returncode=1)
            self._log(f"\n=== BUILD FAILED: {e} ===")
        finally:
            self.state.update(running=False, finished=time.time())
            self._upd_cache.clear()   # a pull may have moved HEAD; re-check next open

    # Filesystem mtime granularity is coarse (up to ~2s on FAT; truncation can
    # even put a just-written file's recorded mtime a hair below the wall-clock
    # time captured moments earlier). A stale binary is from a *previous* build -
    # minutes or more old - so this tolerance separates "made this run" from
    # "left over" without ever misreading a genuinely fresh binary as stale.
    _MTIME_TOLERANCE = 3.0

    def _fresh_binary(self, build_dir, since):
        """The built llama-server if it exists AND was (re)written by this run,
        else None. Distinguishes a binary this build produced from a stale one
        left by a previous successful build."""
        p = self.locate_server_bin(build_dir)
        if not p:
            return None
        try:
            return p if os.path.getmtime(p) >= since - self._MTIME_TOLERANCE else None
        except OSError:
            return None

    def _record_built(self, found):
        """Note the built binary in state and hand it to on_built. A raising
        callback (e.g. read-only config.json) must never fail a good build."""
        self.state["server_bin"] = found
        self._log(f"[binaries] llama-server -> {found}")
        try:
            if self.on_built:
                self.on_built(found)
        except Exception as e:
            self._log(f"[binaries] could not record server_bin: {e}")

    def start(self, src, build_dir, flags, pull=True, jobs=None):
        if self.state["running"]:
            return False
        t = threading.Thread(target=self.run_build,
                             args=(src, build_dir, flags, pull, jobs), daemon=True)
        t.start()
        return True
