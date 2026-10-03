"""Find and stop the processes this LlamaForge copy started - and nothing else.

stop.ps1 / stop.sh used to kill every llama-server on the machine and whatever
held the panel and router ports. That took down a user's own llama-server, or
an unrelated app that happened to sit on 8080. Now:

* The launchers record what they start: `logs/router.pid` (run.ps1, run.sh,
  router_ctl.start) and `logs/panel.pid` (server.py writes its own).
* A port owner is stopped only if it is the recorded PID and its image looks
  right (llama-server for the router, python for the panel). A PID that was
  reused by some other program after a crash fails one of those checks.
* With no pidfile at all (a copy started by an older version) the port owner
  is stopped only if its image looks right.
* Model instances are the router's own children; only those are swept.
* vLLM is stopped only when it has been set up, and only the server on the
  configured port.

    python procs.py --stop <install dir>
"""
import json, os, signal, subprocess, sys, time

IS_WIN = os.name == "nt"
_NO_WINDOW = 0x08000000 if IS_WIN else 0


def pidfile(logdir, name):
    return os.path.join(logdir, name + ".pid")


def write_pid(logdir, name, pid):
    try:
        os.makedirs(logdir, exist_ok=True)
        with open(pidfile(logdir, name), "w", encoding="utf-8") as f:
            f.write(str(int(pid)))
    except (OSError, ValueError):
        pass


def read_pid(logdir, name):
    """The recorded PID; None when there is no pidfile; 0 when it is garbage."""
    try:
        with open(pidfile(logdir, name), encoding="utf-8-sig") as f:
            text = f.read().strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else 0


def _image_name(image):
    base = os.path.basename((image or "").replace("\\", "/")).lower()
    return base[:-4] if base.endswith(".exe") else base


def is_llama(image):
    return "llama" in _image_name(image)


def is_python(image):
    return _image_name(image).startswith("python")


# ---------- process table ----------

def parse_ps(text):
    """`ps -A -o pid=,ppid=,comm=` -> {pid: (ppid, image)}."""
    out = {}
    for line in text.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            out[int(parts[0])] = (int(parts[1]), parts[2].strip())
    return out


def parse_cim(text):
    """Win32_Process rows as JSON -> {pid: (ppid, image)}."""
    try:
        rows = json.loads(text or "[]")
    except ValueError:
        return {}
    if isinstance(rows, dict):
        rows = [rows]
    out = {}
    for r in rows if isinstance(rows, list) else []:
        try:
            out[int(r["ProcessId"])] = (int(r.get("ParentProcessId") or 0),
                                        r.get("ExecutablePath") or r.get("Name") or "")
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _run(cmd, timeout=20):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              creationflags=_NO_WINDOW).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def table():
    if IS_WIN:
        return parse_cim(_run(["powershell", "-NoProfile", "-Command",
            "Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,"
            "ExecutablePath,Name | ConvertTo-Json -Compress"]))
    return parse_ps(_run(["ps", "-A", "-o", "pid=,ppid=,comm="]))


def pid_on_port(port):
    if IS_WIN:
        out = _run(["powershell", "-NoProfile", "-Command",
                    f"(Get-NetTCPConnection -LocalPort {int(port)} -State Listen "
                    f"-ErrorAction SilentlyContinue | Select-Object -First 1 "
                    f"-ExpandProperty OwningProcess)"]).strip()
    else:
        import osplat
        return osplat.pid_on_port_posix(port)
    return int(out) if out.isdigit() and int(out) > 0 else None


def kill(pid):
    if IS_WIN:
        _run(["taskkill", "/PID", str(pid), "/F"])
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    for _ in range(20):
        time.sleep(0.25)
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            return
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


# ---------- what to stop ----------

def owned(port_owner, recorded, procs, looks_right):
    """The port owner, if it is ours. `recorded` is read_pid()'s answer."""
    if not port_owner or port_owner == os.getpid() or port_owner not in procs:
        return None
    if not looks_right(procs[port_owner][1]):
        return None
    if recorded is not None and recorded != port_owner:
        return None
    return port_owner


def descendants(root, procs, looks_right):
    """Every process below `root` whose image passes `looks_right`, deepest first."""
    kids = {}
    for pid, (ppid, _img) in procs.items():
        kids.setdefault(ppid, []).append(pid)
    out, stack = [], [(c, 1) for c in kids.get(root, [])]
    seen = {root}
    while stack:
        pid, depth = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        if looks_right(procs[pid][1]):
            out.append((depth, pid))
        stack += [(c, depth + 1) for c in kids.get(pid, [])]
    return [pid for _d, pid in sorted(out, reverse=True)]


def plan(cfg, logdir, procs, port_owner):
    """[(label, pid), ...] in the order to stop them. Pure: no side effects."""
    steps = []
    router = owned(port_owner(cfg.get("router_port")), read_pid(logdir, "router"),
                   procs, is_llama)
    if router:
        steps += [("model instance", p) for p in descendants(router, procs, is_llama)]
        steps.append(("llama.cpp router", router))
    panel = owned(port_owner(cfg.get("panel_port")), read_pid(logdir, "panel"),
                  procs, is_python)
    if panel:
        steps.append(("LlamaForge dashboard", panel))
    return steps


def vllm_command(cfg, root):
    """The WSL command that stops our vLLM server, or None when vLLM was never set up."""
    if not IS_WIN or not os.path.isfile(os.path.join(root, "vllm_models.json")):
        return None
    port = int(cfg.get("vllm_port") or 8081)
    distro = cfg.get("wsl_distro") or ""
    pattern = f"vllm serve .*--port {port}( |$)"
    cmd = ["wsl.exe"] + (["-d", distro] if distro else [])
    return cmd + ["--", "bash", "-lc", f"pkill -f '{pattern}' 2>/dev/null; true"]


def stop(root):
    try:
        with open(os.path.join(root, "config.json"), encoding="utf-8-sig") as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        cfg = {}
    if not isinstance(cfg, dict):
        cfg = {}
    cfg.setdefault("router_port", 8080)
    cfg.setdefault("panel_port", 8090)
    logdir = os.path.join(root, "logs")
    if not IS_WIN:
        import osplat
        if not osplat.port_tool():
            print("can't tell what is listening on LlamaForge's ports: install lsof, "
                  "or iproute2 (ss), or psmisc (fuser), then run stop again", file=sys.stderr)
            return 1
    steps = plan(cfg, logdir, table(), pid_on_port)
    for label, pid in steps:
        kill(pid)
        print(f"stopped {label} (pid {pid})")
    for name, pid in (("router", dict(steps).get("llama.cpp router")),
                      ("panel", dict(steps).get("LlamaForge dashboard"))):
        if pid:
            try:
                os.remove(pidfile(logdir, name))
            except OSError:
                pass
    vllm = vllm_command(cfg, root)
    if vllm:
        _run(vllm, timeout=30)
        print("stopped LlamaForge's vLLM server, if one was running")
    if not steps:
        print("nothing of LlamaForge's was running.")
    print("LlamaForge stopped.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--stop":
        sys.exit(stop(os.path.abspath(sys.argv[2])))
    print("usage: procs.py --stop <install dir>", file=sys.stderr)
    sys.exit(2)
