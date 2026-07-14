"""Single choke point for everything WSL2. vLLM is Linux-only, so LlamaForge
drives it through `wsl.exe`. Centralizing here means the rest of the backend
never spells out wsl.exe, and tests mock exactly one module.

Windows-only. No third-party deps.
"""
import re, subprocess

CREATE_NO_WINDOW = 0x08000000


def win_to_wsl(path):
    """C:\\a\\b or C:/a/b -> /mnt/c/a/b. Leaves already-POSIX paths alone."""
    p = path.replace("\\", "/")
    m = re.match(r"^([A-Za-z]):/(.*)$", p)
    if m:
        return f"/mnt/{m.group(1).lower()}/{m.group(2)}"
    return p


def _run_text(args, timeout=15):
    """Run a wsl.exe management command and return decoded stdout.
    `wsl -l -v` emits UTF-16-LE; decode leniently and strip NULs."""
    out = subprocess.run(args, capture_output=True, timeout=timeout,
                         creationflags=CREATE_NO_WINDOW).stdout
    try:
        text = out.decode("utf-16-le")
    except Exception:
        text = out.decode("utf-8", errors="replace")
    return text.replace("\x00", "")


def list_distros():
    """[{name, state, version, default}] or [] if WSL isn't installed."""
    try:
        text = _run_text(["wsl.exe", "-l", "-v"])
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return []
    distros = []
    for line in text.splitlines()[1:]:            # skip the header row
        s = line.rstrip()
        if not s.strip():
            continue
        default = s.lstrip().startswith("*")
        parts = s.replace("*", " ", 1).split()
        if len(parts) >= 3:
            distros.append({"name": parts[0], "state": parts[1],
                            "version": parts[2], "default": default})
    return distros


def default_distro():
    for d in list_distros():
        if d["default"]:
            return d["name"]
    ds = list_distros()
    return ds[0]["name"] if ds else ""


def run(cmd, distro=None, timeout=60):
    """Run a shell command inside the distro. Returns (returncode, stdout, stderr).
    Uses `bash -lc` so the login environment (PATH from ~/.profile) is present."""
    args = ["wsl.exe"]
    if distro:
        args += ["-d", distro]
    args += ["--", "bash", "-lc", cmd]
    r = subprocess.run(args, capture_output=True, text=True, timeout=timeout,
                       creationflags=CREATE_NO_WINDOW)
    return r.returncode, r.stdout, r.stderr


def popen(cmd, stdout, stderr, distro=None):
    """Long-running command (a server or an install). Caller supplies open file
    objects for stdout/stderr; returns the Popen handle."""
    args = ["wsl.exe"]
    if distro:
        args += ["-d", distro]
    args += ["--", "bash", "-lc", cmd]
    return subprocess.Popen(args, stdout=stdout, stderr=stderr,
                            stdin=subprocess.DEVNULL,
                            creationflags=CREATE_NO_WINDOW, close_fds=True)
