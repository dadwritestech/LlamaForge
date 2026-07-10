"""Start/stop/restart the llama.cpp router process from the dashboard,
so changing network settings (host, API key) never requires the user
to touch a terminal. Windows-only (uses Get-NetTCPConnection to find
the process bound to a port).
"""
import os, subprocess, time, socket

CREATE_NO_WINDOW = 0x08000000

def lan_ip():
    """Best-effort local-network IP (no traffic sent; just picks the
    interface the OS would use to reach the internet)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return None
    finally:
        s.close()

def _pid_on_port(port):
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             f"(Get-NetTCPConnection -LocalPort {port} -State Listen "
             f"-ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty OwningProcess)"],
            text=True, timeout=10).strip()
        return int(out) if out.isdigit() else None
    except Exception:
        return None

def is_running(port):
    return _pid_on_port(port) is not None

def stop(port, timeout=10):
    pid = _pid_on_port(port)
    if not pid:
        return True
    subprocess.run(["powershell", "-NoProfile", "-Command", f"Stop-Process -Id {pid} -Force"],
                   timeout=10, capture_output=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _pid_on_port(port) is None:
            return True
        time.sleep(0.5)
    return _pid_on_port(port) is None

def start(server_bin, models_ini, port, host, api_key, logdir):
    if not server_bin or not os.path.exists(server_bin):
        return False, "server_bin not found - build llama.cpp first"
    os.makedirs(logdir, exist_ok=True)
    args = [server_bin, "--models-preset", models_ini, "--models-max", "1", "--offline",
            "--host", host, "--port", str(port), "--metrics"]
    if api_key:
        args += ["--api-key", api_key]
    out = open(os.path.join(logdir, "router.out.log"), "a", encoding="utf-8", errors="replace")
    err = open(os.path.join(logdir, "router.err.log"), "a", encoding="utf-8", errors="replace")
    subprocess.Popen(args, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                     creationflags=CREATE_NO_WINDOW, close_fds=True)
    return True, ""

def restart(server_bin, models_ini, port, host, api_key, logdir):
    stop(port)
    return start(server_bin, models_ini, port, host, api_key, logdir)
