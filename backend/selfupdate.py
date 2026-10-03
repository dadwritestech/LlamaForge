"""In-app update for installer-managed copies.

Same path as the one-line installer, without the shell: download the release
zip, extract it, lay it over the install dir with appinstall (user data and
the private Python are never touched), then relaunch through run.ps1/run.sh.
The router is a separate process and the runner skips it when its port is
busy, so loaded models keep serving through the restart.

A git checkout has no manifest and is refused: it updates with `git pull`.
"""
import os, re, shutil, subprocess, sys, tempfile, threading, time, urllib.request, zipfile

import appinstall

REPO = "dadwritestech/LlamaForge"
UA = {"User-Agent": "LlamaForge/1.0 (+local model manager)"}


def _download(url, dest):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


class UpdateJob:
    def __init__(self, root, download=_download):
        self.root, self._download = root, download
        self._lock = threading.Lock()
        self._state = {"state": "idle", "tag": None, "error": None}

    def progress(self):
        with self._lock:
            return dict(self._state)

    def _set(self, **kw):
        with self._lock:
            self._state.update(kw)

    def start(self, tag):
        with self._lock:
            if self._state["state"] in ("downloading", "installing"):
                return False
            self._state = {"state": "downloading", "tag": tag, "error": None}
        threading.Thread(target=self.run, args=(tag,), daemon=True).start()
        return True

    def run(self, tag):
        self._set(state="downloading", tag=tag, error=None)
        if not re.fullmatch(r"v\d+\.\d+\.\d+", tag or ""):
            return self._set(state="error", error=f"not a release tag: {tag!r}")
        if appinstall.installed_version(self.root) is None:
            return self._set(state="error", error="this copy is a git checkout: update it with git pull")
        tmp = tempfile.mkdtemp(prefix="lf-update-")
        try:
            archive = os.path.join(tmp, "release.zip")
            self._download(f"https://github.com/{REPO}/archive/refs/tags/{tag}.zip", archive)
            self._set(state="installing")
            src = os.path.join(tmp, "src")
            with zipfile.ZipFile(archive) as z:
                z.extractall(src)          # zipfile drops absolute and ".." parts
            appinstall.install(src, self.root, tag)
            self._set(state="done")
        except (OSError, ValueError, zipfile.BadZipFile) as e:
            self._set(state="error", error=str(e))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def restart_cmd(root, platform):
    if platform == "windows":
        script = os.path.join(root, "run.ps1").replace("'", "''")
        return ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden",
                "-Command", f"Start-Sleep -Seconds 2; & '{script}'"]
    # root is passed as $0 so no path ever needs shell quoting
    return ["sh", "-c", 'sleep 2; exec bash "$0/run.sh"', root]


def restart(root, platform, exit_after=0.5):
    """Spawn the runner detached, then exit so it can take the panel port."""
    env = dict(os.environ, LLAMAFORGE_NO_BROWSER="1")
    kw = {"cwd": root, "env": env, "stdin": subprocess.DEVNULL,
          "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if platform == "windows":
        # CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP. Not DETACHED_PROCESS:
        # powershell started without a console exits before running anything.
        kw["creationflags"] = 0x08000000 | 0x00000200
    else:
        kw["start_new_session"] = True
    subprocess.Popen(restart_cmd(root, platform), **kw)
    threading.Timer(exit_after, lambda: os._exit(0)).start()
