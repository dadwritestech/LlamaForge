"""Start/stop/restart the llama.cpp router process from the dashboard,
so changing network settings (host, API key) never requires the user
to touch a terminal. Windows uses Get-NetTCPConnection to find the
process bound to a port; Linux/macOS use lsof.
"""
import os, signal, subprocess, time, socket

import network_policy, osplat

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

# ---------------------------------------------------------------- capability
# The router is driven as `<server_bin> --models-preset <ini> --models-max 1`.
# Not every llama-family binary can do that: ik_llama.cpp forked before router
# mode existed and answers `unknown argument: --models-preset`, which would take
# the router down and leave the dashboard with nothing to talk to. Ask first.
#
# Cached on (path, mtime) like the knob schema, and for the same reason: a
# rebuild re-probes, and a failed probe is never cached so fixing the binary
# takes effect without restarting the backend.
_ROUTER_MODE = {}

def clear_router_mode_cache():
    _ROUTER_MODE.clear()

def supports_router_mode(server_bin):
    if not server_bin:
        return False
    try:
        key = (server_bin, os.path.getmtime(server_bin))
    except OSError:
        return False
    if key in _ROUTER_MODE:
        return _ROUTER_MODE[key]
    try:
        out = subprocess.check_output([server_bin, "--help"], text=True, timeout=25,
                                      stderr=subprocess.STDOUT,
                                      creationflags=CREATE_NO_WINDOW if osplat.IS_WIN else 0)
    except Exception:
        return False                      # unreadable -> not cached
    ok = "--models-preset" in out
    _ROUTER_MODE[key] = ok
    return ok


def _pid_on_port(port):
    if not osplat.IS_WIN:
        return osplat.pid_on_port_posix(port)
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

def _kill(pid, force=False):
    if osplat.IS_WIN:
        subprocess.run(["powershell", "-NoProfile", "-Command", f"Stop-Process -Id {pid} -Force"],
                       timeout=10, capture_output=True)
    else:
        try:
            os.kill(pid, signal.SIGKILL if force else signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

def stop(port, timeout=10):
    pid = _pid_on_port(port)
    if not pid:
        return True
    _kill(pid)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _pid_on_port(port) is None:
            return True
        time.sleep(0.5)
    _kill(pid, force=True)               # POSIX escalation; no-op change on Windows
    time.sleep(0.5)
    return _pid_on_port(port) is None

def start(server_bin, models_ini, port, host, api_key, logdir):
    reason = network_policy.start_error(host, api_key)
    if reason:
        return False, reason
    if not server_bin or not os.path.exists(server_bin):
        return False, "server_bin not found - build llama.cpp first"
    # Port 8080 is a popular default (XAMPP, Apache, other dev servers). Without
    # this check we spawned llama-server anyway; it logged "couldn't bind HTTP
    # server socket" into router.err.log and exited, while start() had already
    # reported success - so the dashboard showed every model offline with no
    # hint as to why. Ask before spawning, and name the port in the error.
    if is_running(port):
        return False, (f"port {port} is already in use by another process - "
                       f"stop it, or change the router port in Setup")
    os.makedirs(logdir, exist_ok=True)
    args = [server_bin, "--models-preset", models_ini, "--models-max", "1", "--offline",
            "--host", host, "--port", str(port), "--metrics"]
    if api_key:
        args += ["--api-key", api_key]
    out = open(os.path.join(logdir, "router.out.log"), "a", encoding="utf-8", errors="replace")
    err = open(os.path.join(logdir, "router.err.log"), "a", encoding="utf-8", errors="replace")
    kw = ({"creationflags": CREATE_NO_WINDOW} if osplat.IS_WIN
          else {"start_new_session": True})   # detach from the dashboard's session
    try:
        subprocess.Popen(args, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                         close_fds=True, **kw)
    finally:
        # The child holds its own duplicated handles; these are the parent's
        # copies and nothing reads them here. Leaving them open leaked two
        # handles per restart for the life of the dashboard.
        out.close()
        err.close()
    return True, ""

def restart(server_bin, models_ini, port, host, api_key, logdir):
    reason = network_policy.start_error(host, api_key)
    if reason:
        return False, reason
    stop(port)
    return start(server_bin, models_ini, port, host, api_key, logdir)
