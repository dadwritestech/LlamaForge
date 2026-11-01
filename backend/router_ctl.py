"""Start/stop/restart the llama.cpp router process from the dashboard,
so changing network settings (host, API key) never requires the user
to touch a terminal. Windows uses Get-NetTCPConnection to find the
process bound to a port; Linux/macOS use lsof.
"""
import json, os, signal, subprocess, time, socket, urllib.error, urllib.request

import logfiles, network_policy, osplat, procs

CREATE_NO_WINDOW = 0x08000000

def lan_ip():
    """Best-effort local-network IP (no traffic sent; just picks the
    interface the OS would use to reach the internet)."""
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return None
    finally:
        if s is not None:
            try:
                s.close()
            except Exception:
                pass

# ---------------------------------------------------------------- capability
# The router is driven as `<server_bin> --models-preset <ini> --models-max 1`.
# Not every llama-family binary can do that: ik_llama.cpp forked before router
# mode existed and answers `unknown argument: --models-preset`, which would take
# the router down and leave the dashboard with nothing to talk to. Ask first.
#
# Cached on (path, mtime) like the knob schema, and for the same reason: a
# rebuild re-probes, and a failed probe is never cached so fixing the binary
# takes effect without restarting the backend.
_HELP = {}

def clear_router_mode_cache():
    _HELP.clear()

def _help_text(server_bin):
    if not server_bin:
        return ""
    try:
        key = (server_bin, os.path.getmtime(server_bin))
    except OSError:
        return ""
    if key in _HELP:
        return _HELP[key]
    try:
        # Not check_output: ik_llama.cpp prints its whole --help and exits 1.
        # utf-8 for the same reason argspec.build_schema gives (cp1252 locale).
        r = subprocess.run([server_bin, "--help"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=25,
                           stdin=subprocess.DEVNULL,
                           creationflags=CREATE_NO_WINDOW if osplat.IS_WIN else 0)
    except Exception:
        return ""                         # unreadable -> not cached
    # some builds/forks route usage through the log system -> stderr
    out = r.stdout if (r.stdout or "").strip() else (r.stderr or "")
    if not out.strip():
        return ""                         # nothing printed -> not cached
    _HELP[key] = out
    return out

def help_text(server_bin):
    """The binary's --help, cached per (path, mtime); '' when it can't be read."""
    return _help_text(server_bin)

def supports_router_mode(server_bin):
    return "--models-preset" in _help_text(server_bin)

def supports_no_autoload(server_bin):
    """Multi-model mode needs it: with autoload on, any client request for an
    unloaded model loads it behind the planner's back and the router evicts
    whatever it likes to make room."""
    return "--no-models-autoload" in _help_text(server_bin)

def supports_cors_origins(server_bin):
    """Older builds and forks predate --cors-origins and would refuse to start."""
    return "--cors-origins" in _help_text(server_bin)


def supports_models_memory_margin(server_bin):
    """llama.cpp v0.8.0's router can unload the least recently used model
    when a new one won't fit beside a per-device margin of free memory."""
    return "--models-memory-margin" in _help_text(server_bin)


def supports_slot_save(server_bin):
    """Elastic VRAM keeps conversations through /slots save/restore, which
    needs a save folder; the router hands its own flag to every model."""
    return "--slot-save-path" in _help_text(server_bin)


def slot_save_dir(logdir):
    """<data>/slot-saves, beside logs/. Gitignored; slotkeep sweeps it."""
    return os.path.join(os.path.dirname(os.path.abspath(logdir)), "slot-saves")


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

def auth_state(port, key):
    """How the router on `port` treats auth, via /props (keyed, and unlike
    /metrics it answers 200 in router mode with no model loaded):
    "open" (no key needed), "ok" (our key works), "mismatch" (keyed, but not
    with ours) or "unknown" (down, or an answer we cannot read)."""
    url = f"http://127.0.0.1:{port}/props"

    def status(headers):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                        timeout=3) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code
        except Exception:
            return None

    first = status({})
    if first == 200:
        return "open"
    if first != 401 or not key:
        return "unknown"
    return {200: "ok", 401: "mismatch"}.get(
        status({"Authorization": "Bearer " + key}), "unknown")

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

def start(server_bin, models_ini, port, host, api_key, logdir, local_key="", pool=None,
          slot_save=False):
    """api_key is the user's key and decides policy (LAN needs one the user
    knows); local_key is LlamaForge's own, used when the user has none.
    pool: None for one model at a time, or slots.router_pool()'s
    {"models_max", "autoload"} for multi-model mode. slot_save: give the
    router a slot save folder (Elastic VRAM keeping conversations); off, no
    API-key holder can write slot dumps to disk."""
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
    models_max = str(pool["models_max"]) if pool else "1"
    args = [server_bin, "--models-preset", models_ini, "--models-max", models_max, "--offline",
            "--host", host, "--port", str(port), "--metrics"]
    if pool and not pool.get("autoload", True):
        args.append("--no-models-autoload")
    if pool and pool.get("memory_margin_mib"):
        args += ["--models-memory-margin", str(pool["memory_margin_mib"])]
    if slot_save and supports_slot_save(server_bin):
        try:                      # never let this block the router starting
            os.makedirs(slot_save_dir(logdir), exist_ok=True)
            args += ["--slot-save-path", slot_save_dir(logdir)]
        except OSError:
            pass
    args += network_policy.router_auth_args(host, api_key or local_key,
                                            supports_cors_origins(server_bin))
    out = logfiles.open_append(os.path.join(logdir, "router.out.log"))
    err = logfiles.open_append(os.path.join(logdir, "router.err.log"))
    kw = ({"creationflags": CREATE_NO_WINDOW} if osplat.IS_WIN
          else {"start_new_session": True})   # detach from the dashboard's session
    try:
        proc = subprocess.Popen(args, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                                close_fds=True, **kw)
    finally:
        # The child holds its own duplicated handles; these are the parent's
        # copies and nothing reads them here. Leaving them open leaked two
        # handles per restart for the life of the dashboard.
        out.close()
        err.close()
    procs.write_pid(logdir, "router", proc.pid)   # stop.ps1/.sh stop only this one
    record_pool(logdir, proc.pid, pool)
    return True, ""


def record_pool(logdir, pid, pool):
    """Remember which pool the router under `pid` runs, beside its pidfile."""
    try:
        with open(os.path.join(logdir, "router.pool.json"), "w", encoding="utf-8") as f:
            json.dump({"pid": int(pid), "pool": pool}, f)
    except (OSError, TypeError, ValueError):
        pass


def running_pool(logdir):
    """The pool of the router LlamaForge started, or None (one model at a time)
    when the recorded router is not the one running: run.ps1 / run.sh start it
    single and overwrite the pidfile, not this record."""
    try:
        with open(os.path.join(logdir, "router.pool.json"), encoding="utf-8") as f:
            rec = json.load(f)
        return rec["pool"] if rec.get("pid") == procs.read_pid(logdir, "router") else None
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None

def restart(server_bin, models_ini, port, host, api_key, logdir, local_key="", pool=None,
            slot_save=False):
    reason = network_policy.start_error(host, api_key)
    if reason:
        return False, reason
    # Without a way to see the old router, stop() finds nothing, start() sees a
    # free port, and the old router keeps serving the old settings.
    if not osplat.IS_WIN and not osplat.port_tool():
        return False, ("can't see which process holds the router port - install "
                       "lsof, or iproute2 (ss), or psmisc (fuser)")
    stop(port)
    return start(server_bin, models_ini, port, host, api_key, logdir, local_key, pool,
                 slot_save=slot_save)
