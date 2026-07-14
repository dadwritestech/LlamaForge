"""vLLM process manager — the multi-model router that vLLM itself lacks.

One `vllm serve` process serves one model. This manager owns a *list* of
instances (v1 caps at one via a guard; lifting it + per-instance ports enables
concurrent serving later). It spawns inside WSL via wsl.popen with stdout/stderr
redirected to Windows-side log files (same pattern as router_ctl.py), polls
vLLM's /health over WSL localhost-forwarding, and reconciles live processes on
startup so restarting LlamaForge never loses or double-starts a model.

Pure stdlib.
"""
import os, time, threading, urllib.request

import wsl

MAX_INSTANCES = 1                     # v1 guard; raise for concurrency
READY_TIMEOUT = 600                   # seconds before a stuck load -> failed
HEALTH_INTERVAL = 3


def settings_to_flags(settings):
    """{knob: value} -> "--knob value ..." with store-true booleans handled.
    'true' -> bare flag; 'false' -> omitted; everything else -> --k v."""
    parts = []
    for k, v in settings.items():
        v = str(v).strip()
        if v.lower() == "true":
            parts.append(f"--{k}")
        elif v.lower() == "false" or v == "":
            continue
        else:
            parts.append(f"--{k} {v}")
    return " ".join(parts)


def build_serve_cmd(venv, model_ref, port, flag_str):
    """The bash command run inside WSL to serve one model."""
    base = f"{venv}/bin/vllm serve {model_ref} --host 0.0.0.0 --port {port}"
    return f"{base} {flag_str}".strip()


class Manager:
    def __init__(self, distro, port, venv, logdir):
        self.distro = distro
        self.port = port
        self.venv = venv
        self.logdir = logdir
        self.instances = []           # [{model_id, port, state, started_at}]
        self.lock = threading.Lock()

    # ---------- lifecycle ----------
    def start(self, model_id, model_ref, flag_str):
        with self.lock:
            if len(self.instances) >= MAX_INSTANCES:
                return False, "a vLLM model is already running (stop it first)"
            cmd = build_serve_cmd(self.venv, model_ref, self.port, flag_str)
            os.makedirs(self.logdir, exist_ok=True)
            out = open(os.path.join(self.logdir, "vllm.out.log"), "a",
                       encoding="utf-8", errors="replace")
            err = open(os.path.join(self.logdir, "vllm.err.log"), "a",
                       encoding="utf-8", errors="replace")
            wsl.popen(cmd, stdout=out, stderr=err, distro=self.distro)
            self.instances.append({"model_id": model_id, "port": self.port,
                                   "state": "starting", "started_at": time.time()})
        threading.Thread(target=self._await_ready, args=(model_id,), daemon=True).start()
        return True, ""

    def _await_ready(self, model_id):
        # We sleep before the first health check so the instance provably stays
        # in "starting" for one interval after start() returns — the caller (and
        # tests) can observe "starting" without racing this daemon thread. vLLM
        # takes 1-5 min to come up, so a HEALTH_INTERVAL head start is free.
        # The UI maps both "starting" and "loading" to "loading" anyway.
        deadline = time.time() + READY_TIMEOUT
        while time.time() < deadline:
            time.sleep(HEALTH_INTERVAL)
            if self._health_ok():
                self._set_state(model_id, "ready")
                return
            self._set_state(model_id, "loading")
        self._set_state(model_id, "failed")

    def _health_ok(self):
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/health", timeout=3) as r:
                return r.status == 200
        except Exception:
            return False

    def stop(self, model_id):
        wsl.run("pkill -f 'vllm serve' || true", distro=self.distro, timeout=20)
        deadline = time.time() + 15
        while time.time() < deadline and self._health_ok():
            time.sleep(0.5)
        with self.lock:
            self.instances = [i for i in self.instances if i["model_id"] != model_id]
        return True

    # ---------- state ----------
    def _set_state(self, model_id, state):
        with self.lock:
            for i in self.instances:
                if i["model_id"] == model_id:
                    i["state"] = state

    def status(self):
        with self.lock:
            return [{"model_id": i["model_id"], "port": i["port"],
                     "state": i["state"], "started_at": i["started_at"],
                     "endpoint": f"http://127.0.0.1:{i['port']}"}
                    for i in self.instances]

    def reconcile(self):
        """On startup: if we think something's running but no vllm process
        exists in WSL, drop the stale instance record."""
        if not self.instances:
            return
        code, _out, _err = wsl.run("pgrep -f 'vllm serve'", distro=self.distro, timeout=15)
        if code != 0:                 # pgrep exit 1 == no match
            with self.lock:
                self.instances = []
