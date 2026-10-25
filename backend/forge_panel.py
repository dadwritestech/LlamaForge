"""Auto-Forge routes: wires forge/ (pure logic) to the real router, GPUs and models.ini.

deps = the routes module: config access, the slot manager (help_items, unload),
_active_server_bin, _game_loaded/_game_load (what's loaded now / load it back),
post_unload_all, _apply_knobs_and_reload, ApiError, Req, SLOTS, LOGDIR.
"""
import contextlib
import json
import os
import time
import urllib.request

import config
import hardware
import network_policy
import slotctl
import slotproc
from forge import inventory, job, probe, report, runner

SANDBOX_ID = "forge-sandbox"
SANDBOX_PORT_BASE = 8160
_INTENTS = ("coding", "chat", "long", "agent")


def _vram_used():
    """MiB in use per GPU as total - free (WDDM can report used=0)."""
    return [g["total_mib"] - g["free_mib"] for g in slotctl.smi_snapshot()]


class Panel:
    def __init__(self, deps):
        self._d = deps
        self.dir = os.path.join(deps.LOGDIR, "forge")
        self.procs = slotproc.Manager(self.dir)
        self.runner = runner.Runner()
        self._mid = None

    # ---------------------------------------------------------- wiring
    def _state_path(self, mid):
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in mid)[:80]
        return os.path.join(self.dir, f"job-{safe}.json")

    def _settings(self, mid):
        secs = config.read_sections()
        if mid not in secs:
            raise self._d.ApiError(404, f"unknown model: {mid}")
        return {**(secs.get("*") or {}), **secs[mid]}

    def _model_paths(self, model_path):
        """Candidate drafter files: every model in models.ini, plus the .gguf files
        next to this model and in its parent's subfolders (one level, no drive scan)."""
        out = [s.get("model") for s in config.read_sections().values() if s.get("model")]
        here = os.path.dirname(model_path)
        for d in {here, os.path.dirname(here)}:
            try:
                names = os.listdir(d)
            except OSError:
                continue
            for n in names:
                full = os.path.join(d, n)
                if n.lower().endswith(".gguf"):
                    out.append(full)
                elif d != here and os.path.isdir(full):
                    try:
                        out += [os.path.join(full, f) for f in os.listdir(full)
                                if f.lower().endswith(".gguf")]
                    except OSError:
                        pass
        return out

    def _post(self, path, body, timeout=900):
        st = self.procs.status().get(SANDBOX_ID) or {}
        if not st.get("port"):
            raise OSError("the sandbox server isn't running")
        headers = {"Content-Type": "application/json"}
        key = network_policy.effective_key(config.load())
        if key:
            headers["Authorization"] = "Bearer " + key
        req = urllib.request.Request(f"http://127.0.0.1:{st['port']}{path}",
                                     json.dumps(body).encode(), headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)

    def _make_job(self, mid, req):
        c = config.load()
        sbin = self._d._active_server_bin(c)
        items = self._d.SLOTS.help_items(sbin)
        if not items:
            raise self._d.ApiError(400, f"couldn't read the arguments of {sbin}")
        gpus = hardware.detect_gpus()
        if not gpus:
            raise self._d.ApiError(400, "Forge needs at least one NVIDIA GPU (nvidia-smi found none)")
        settings = self._settings(mid)
        drafters = inventory.drafters(settings.get("model", ""), settings,
                                      self._model_paths(settings.get("model", "")),
                                      size_of=os.path.getsize)
        inv = inventory.build(gpus, items, settings, drafters)
        api_key = network_policy.effective_key(c)

        def start(s):
            try:
                argv, _dropped = slotproc.argv_for(sbin, SANDBOX_ID, s, items, items, api_key=api_key)
            except ValueError as e:
                return False, str(e)
            ok, err, _port = self.procs.start(SANDBOX_ID, argv, port_base=SANDBOX_PORT_BASE)
            return ok, err

        def state():
            return (self.procs.status().get(SANDBOX_ID) or {}).get("state")

        def stop():
            self.procs.stop(SANDBOX_ID)

        prober = probe.Prober(start=start, state=state, stop=stop, vram=_vram_used,
                              log_tail=lambda: self.procs.log_tail(SANDBOX_ID, 400),
                              gpus=gpus, clock=time.monotonic, sleep=time.sleep)
        kwargs = {"enable_thinking": False}

        def bench(counts, round_id, n_predict):
            from forge import bench as fb
            return fb.run_round(self._post, counts, round_id, n_predict, kwargs)

        def sample(prompts):
            from forge import sanity
            return {p: sanity.sample(self._post, p) for p in prompts}

        return job.Job(self._state_path(mid), inv, req, load=prober.load, stop=stop,
                       bench=bench, sample=sample, lease=self._lease, clock=time.time)

    @contextlib.contextmanager
    def _lease(self):
        """The GPUs to Forge alone: unload everything, load it back afterwards."""
        loaded = list(self._d._game_loaded())
        self._d.post_unload_all(self._d.Req({}))
        try:
            yield
        finally:
            self.procs.stop(SANDBOX_ID)
            for mid in loaded:
                try:
                    self._d._game_load(mid)
                except Exception:       # noqa: BLE001 - best effort, the user can load it
                    pass

    def _read(self, mid):
        try:
            with open(self._state_path(mid), encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def _write(self, mid, st):
        with open(self._state_path(mid), "w", encoding="utf-8") as f:
            json.dump(st, f)

    # ---------------------------------------------------------- routes
    def post_start(self, req):
        b = req.body or {}
        mid = b.get("model") or ""
        intent = b.get("intent") if b.get("intent") in _INTENTS else "coding"
        try:
            ctx = int(b.get("ctx") or 32768)
        except (TypeError, ValueError):
            raise self._d.ApiError(400, "ctx must be a number")
        if not 512 <= ctx <= 1 << 20:
            raise self._d.ApiError(400, "ctx must be between 512 and 1048576")
        r = {"intent": intent, "ctx": ctx, "allow_loss": bool(b.get("allow_loss")),
             "time": "thorough" if b.get("time") == "thorough" else "quick"}
        if self.runner.running():
            raise self._d.ApiError(409, "a Forge run is already going")
        j = self._make_job(mid, r)
        self._mid = mid
        self.runner.start(j)
        return 200, {"ok": True, "model": mid, "req": r}

    def get_status(self, req):
        mid = req.q("model") or self._mid
        running = self.runner.running() and mid == self._mid
        st = self.runner.snapshot() if mid == self._mid else None
        if st is None and mid:
            st = self._read(mid)
        out = report.summary(st, running)
        out["model"] = mid
        return 200, out

    def post_cancel(self, req):
        return 200, {"ok": self.runner.cancel()}

    def post_apply(self, req):
        mid = (req.body or {}).get("model") or ""
        if self.runner.running():
            raise self._d.ApiError(409, "wait for the Forge run to finish")
        st = self._read(mid)
        win = ((st or {}).get("result") or {}).get("winner")
        if not win:
            raise self._d.ApiError(400, "no Forge result to apply for this model")
        current = config.read_sections().get(mid) or {}
        updates, undo = report.apply_plan(win["settings"], current)
        updates.pop("model", None)
        undo.pop("model", None)
        self._d._apply_knobs_and_reload(mid, updates)
        st["applied"] = {"undo": undo, "updates": updates}
        self._write(mid, st)
        return 200, {"ok": True, "updates": updates}

    def post_undo(self, req):
        mid = (req.body or {}).get("model") or ""
        st = self._read(mid)
        applied = (st or {}).get("applied")
        if not applied:
            raise self._d.ApiError(400, "nothing to undo")
        self._d._apply_knobs_and_reload(mid, applied["undo"])
        st["applied"] = None
        self._write(mid, st)
        return 200, {"ok": True}

    def register(self, get_routes, post_routes):
        get_routes["/api/forge/status"] = self.get_status
        post_routes["/api/forge/start"] = self.post_start
        post_routes["/api/forge/cancel"] = self.post_cancel
        post_routes["/api/forge/apply"] = self.post_apply
        post_routes["/api/forge/undo"] = self.post_undo
