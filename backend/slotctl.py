"""Multi-model loads, end to end: plan -> place -> load -> measure.

slots.py decides; this module acts. It owns the I/O slots.py is kept free of:
the nvidia-smi snapshot, `llama-server --list-devices`, available RAM, the
models.ini keys that hold a model to the GPU the planner chose, the router's
load/unload, and the before/after readings that teach footprint.Store what a
configuration really costs on this machine.

Placement is reversible. LlamaForge writes three keys (device, split-mode,
main-gpu) into a model's section and remembers in config.json slots.placed what
was there before. A key the user has edited since is theirs: the record is
dropped, and nothing is ever "restored" over their edit.

One load at a time: while a model loads its memory is still climbing, so a
second plan against the same snapshot would book the same free memory twice.

A model pinned to another build (config.json model_builds) is planned, placed
and measured the same way but runs as its own llama-server (slotproc): the
router can only run its own binary, and ik_llama.cpp has no router mode.
Without a pool (single-model mode) such a load swaps out whatever is up, as a
router load would, and the section is used as written.
"""
import hashlib
import os
import subprocess
import threading
import time

import argspec
import config
import footprint
import gguf
import network_policy
import osplat
import router_ctl
import slotproc
import slots

MIB = 1024 * 1024
LOAD_TIMEOUT_S = 600          # a 27B from a spinning disk takes minutes
UNLOAD_TIMEOUT_S = 30
RUNNING = ("loaded", "sleeping")   # the router's is_running(), minus "loading"
_MANAGED = ("device", "split-mode", "main-gpu")
_CREATE_NO_WINDOW = 0x08000000


def _on(v):
    return v is not None and str(v).strip().lower() not in ("0", "false", "off", "no", "disabled")


def _size(path):
    try:
        return gguf.total_size(path) if path else 0
    except OSError:
        return 0


def _bare(v):
    """An ini value without its inline comment (read_sections' rule)."""
    return v.split(";", 1)[0].strip() if isinstance(v, str) else v


def _unknown(mid):
    return _refuse(f"the router doesn't list {mid}; is it in models.ini?")


def _refuse(reason):
    return {"ok": False, "reason": reason, "devices": [], "place": None, "footprint": {},
            "source": None, "evict": [], "already": False}


def proc_status(p):
    """A process slot's state in the router's /models terms."""
    if p.get("state") == "ready":
        return {"value": "loaded", "proc": True}
    if p.get("state") == "loading":
        return {"value": "loading", "proc": True}
    return {"value": "unloaded", "failed": True, "exit_code": p.get("exit_code"), "proc": True}


def smi_snapshot():
    """[{index, name, total_mib, used_mib, free_mib}]; [] without nvidia-smi or rocm-smi."""
    gpus = _nvidia_snapshot()
    if gpus:
        return gpus
    return _amd_snapshot()


def _nvidia_snapshot():
    """Try nvidia-smi; return GPU list or empty on failure."""
    try:
        out = osplat.run_text(["nvidia-smi",
                               "--query-gpu=index,name,memory.total,memory.used,memory.free",
                               "--format=csv,noheader,nounits"], timeout=8)
    except Exception:
        return []
    gpus = []
    for ln in out.strip().splitlines():
        f = [x.strip() for x in ln.split(",")]
        if len(f) < 5:
            continue
        try:
            gpus.append({"index": int(f[0]), "name": f[1], "total_mib": int(f[2]),
                         "used_mib": int(f[3]), "free_mib": int(f[4])})
        except ValueError:
            continue
    return gpus


def _amd_snapshot():
    """Try rocm-smi CSV output; return GPU list or empty on failure."""
    try:
        out = osplat.run_text(["rocm-smi", "--showmeminfo", "vram", "--unit", "MiB", "--csv"],
                              timeout=8)
    except Exception:
        return []
    lines = out.strip().splitlines()
    if not lines:
        return []
    # Parse CSV: Card,Memory Used [MiB],Memory Total [MiB],Memory Free [MiB]
    header = lines[0]
    if "Card" not in header or "Memory Total" not in header:
        return []
    gpus = []
    for ln in lines[1:]:
        f = [x.strip() for x in ln.split(",")]
        if len(f) < 4:
            continue
        try:
            idx = int(f[0])
            used = int(f[1])
            total = int(f[2])
            free = int(f[3])
            gpus.append({"index": idx, "name": f"AMD GPU {idx}", "total_mib": total,
                         "used_mib": used, "free_mib": free})
        except ValueError:
            continue
    return gpus


class SlotManager:
    """deps: the routes module (router(), _active_server_bin(), LOGDIR), as
    backends.LlamaCppBackend takes it. The probes below are plain methods so a
    test can replace them on the instance."""

    def __init__(self, deps, store_path):
        self._d = deps
        self.store = footprint.Store(store_path)
        self._lock = threading.RLock()
        self._live = {}           # model -> {devices, footprint, key, last_used}
        self._loading = None      # the model whose load is in flight
        self._unloads = 0         # an unload mid-load spoils that load's reading
        self._devlist = (None, "")

    # ---------- probes ----------
    def ini_path(self):
        return config.ini_path("llamacpp")

    def gpus(self):
        return smi_snapshot()

    def list_devices(self, sbin):
        try:
            k = (sbin, os.path.getmtime(sbin))
        except (OSError, TypeError):
            return ""
        if self._devlist[0] == k:
            return self._devlist[1]
        try:
            out = subprocess.run([sbin, "--list-devices"], capture_output=True, text=True,
                                 timeout=60, creationflags=_CREATE_NO_WINDOW if osplat.IS_WIN else 0)
            text = (out.stdout or "") + (out.stderr or "")
        except Exception:
            return ""                      # never cache a failure
        self._devlist = (k, text)
        return text

    def ram_free_mib(self):
        b = osplat.available_ram_bytes()
        return b // MIB if b else None

    def build_id(self, sbin):
        try:
            mtime = int(os.path.getmtime(sbin))
        except (OSError, TypeError):
            return ""
        ident = f"{os.path.normcase(os.path.abspath(sbin))}|{mtime}"
        return hashlib.sha1(ident.encode()).hexdigest()[:10]

    def sleep(self, s):
        time.sleep(s)

    def pool(self):
        return router_ctl.running_pool(self._d.LOGDIR)

    def procs(self):
        """The slotproc.Manager for pinned models; None runs none."""
        return getattr(self._d, "PROCS", None)

    def pinned(self, mid):
        """(server_bin, error) of a model pinned to another build; (None, "")
        when the router serves it."""
        f = getattr(self._d, "pinned_bin", None)
        return f(mid) if f else (None, "")

    def help_items(self, sbin):
        return argspec.parse_help(router_ctl.help_text(sbin)) if sbin else []

    def predict(self, mid, settings):
        path = settings.get("model") or ""
        mmproj = settings.get("mmproj")
        draft = settings.get("model-draft") or settings.get("md")
        return footprint.predict(
            settings, gguf.layout(path) if path else None,
            extra_gpu_bytes=_size(draft), size_bytes=_size(path) or None,
            mmproj_bytes=0 if (not mmproj or _on(settings.get("no-mmproj"))) else _size(mmproj))

    # ---------- config.json slots state ----------
    def _slots(self, c=None):
        s = (c or config.load()).get("slots")
        return s if isinstance(s, dict) else {}

    def _placed(self):
        p = self._slots().get("placed")
        return p if isinstance(p, dict) else {}

    def _main(self):
        return self._slots().get("main") or ""

    def _edit(self, fn):
        def apply(c):
            s = c.get("slots") if isinstance(c.get("slots"), dict) else {}
            if not isinstance(s.get("placed"), dict):
                s["placed"] = {}
            fn(s)
            c["slots"] = s
        config.mutate(apply)

    def _set_main(self, mid):
        self._edit(lambda s: s.__setitem__("main", mid))

    def _remember(self, mid, rec):
        self._edit(lambda s: s["placed"].__setitem__(mid, rec))

    def _forget(self, mid):
        self._edit(lambda s: s["placed"].pop(mid, None))

    # ---------- models.ini ----------
    def _effective(self, mid):
        """[*] merged with the model's section, with LlamaForge's own placement
        swapped back for what the user had. Our keys that no longer read as we
        wrote them were edited by the user and are theirs from now on."""
        secs = config.read_sections(self.ini_path())
        sec = dict(secs.get(mid) or {})
        rec = self._placed().get(mid)
        if rec:
            keys, was = rec.get("keys") or {}, rec.get("was") or {}
            if all(sec.get(k) == v for k, v in keys.items()):
                for k in keys:
                    sec.pop(k, None)
                    if was.get(k) is not None:
                        sec[k] = _bare(was[k])
            else:
                self._forget(mid)
        return {**(secs.get("*") or {}), **sec}

    def _apply(self, mid, place):
        """Write LlamaForge's placement for `mid`, or clear it: place None is a
        model the user pinned, "" one that spans; neither wants our keys."""
        if not place:
            return self._unplace(mid)
        want = {"device": place, "split-mode": "none", "main-gpu": "0"}
        rec = self._placed().get(mid)
        if rec and rec.get("keys") == want:
            return False
        path = self.ini_path()
        if rec:
            was = rec.get("was") or {}
        else:
            # raw: the user's inline comment comes back with the value
            sec = config.read_sections(path, raw=True).get(mid) or {}
            was = {k: sec.get(k) for k in _MANAGED}
        config.set_keys(mid, want, path)
        self._remember(mid, {"keys": want, "was": was})
        return True

    def _unplace(self, mid):
        rec = self._placed().get(mid)
        if not rec:
            return False
        path = self.ini_path()
        sec = config.read_sections(path).get(mid)
        back = {}
        if sec is not None:                # a deleted model's section stays deleted
            was = rec.get("was") or {}
            back = {k: was.get(k) for k, v in (rec.get("keys") or {}).items()
                    if sec.get(k) == v}
            if back:
                config.set_keys(mid, back, path)
        self._forget(mid)
        return bool(back)

    def unplace_all(self):
        """Put every section LlamaForge placed back as the user had it (multi-model
        turned off, or found on at startup with the setting off)."""
        with self._lock:
            for mid in list(self._placed()):
                self._unplace(mid)

    # ---------- router ----------
    def _statuses(self):
        code, out = self._d.router("/models")
        if code != 200:
            return None
        res = {}
        for m in (out or {}).get("data") or []:
            s = m.get("status")
            res[m.get("id")] = {"value": s} if isinstance(s, str) else (s or {})
        procs = self.procs()
        for m, p in (procs.status() if procs else {}).items():
            res[m] = proc_status(p)
        return res

    def _used(self):
        """MiB taken per GPU, as total - free: the planner works from `free`, and
        under Windows' WDDM `memory.used` can read 0 with memory in use."""
        out = {}
        for g in self.gpus():
            if g.get("free_mib") is not None and g.get("total_mib"):
                out[g["index"]] = g["total_mib"] - g["free_mib"]
            else:
                out[g["index"]] = g.get("used_mib") or 0
        return out

    def key(self, mid, settings=None, sbin=None):
        sbin = sbin if sbin is not None else self._d._active_server_bin(config.load())
        settings = settings if settings is not None else self._effective(mid)
        return footprint.key(mid, self.build_id(sbin), settings)

    def _cmap(self, gpus, sbin):
        """{CUDA index: nvidia-smi index}; None when the two can't be matched."""
        if len(gpus) > 1:
            return slots.cuda_map(self.list_devices(sbin), gpus)
        return {g["index"]: g["index"] for g in gpus}

    def _running(self, st, gpus=None, cmap=None):
        main = self._main()
        out, seen = [], {"gpus": gpus, "cmap": cmap}
        for m, s in st.items():
            v = s.get("value")
            if m == "default" or (v not in RUNNING and v != "loading"):
                continue
            live = self._live.get(m)
            if live:
                devices, fp = live.get("devices") or [], live.get("footprint") or {}
            else:
                devices, fp = self._where(m, seen)
            out.append({"model": m, "status": v, "role": "main" if m == main else "worker",
                        "devices": devices, "footprint": fp,
                        "last_used": (live or {}).get("last_used") or 0})
        return out

    def _where(self, m, seen):
        """(devices, footprint) of a model running that this process didn't load
        (the dashboard restarted under the router, or it was loaded single):
        where its section sends it, and what it measured there before."""
        if seen.get("gpus") is None:
            seen["gpus"] = self.gpus()
            seen["cmap"] = self._cmap(seen["gpus"], self._d._active_server_bin(config.load()))
        gpus, cmap = seen["gpus"], seen["cmap"]
        secs = config.read_sections(self.ini_path())
        p = slots.pins({**(secs.get("*") or {}), **(secs.get(m) or {})})
        if p is None:                          # llama.cpp's default: split over every GPU
            devices = sorted(g["index"] for g in gpus)
        else:
            devices = ((slots.to_smi(p, cmap) or {}).get("devices") if cmap else None) or []
        if not devices:
            return [], {}
        k = self.key(m, sbin=self.pinned(m)[0] or None)
        return devices, dict(self.store.measured(k).get(tuple(devices)) or {})

    def _plan(self, mid, role, st, fbin=None):
        """(verdict, footprint key) for loading `mid` next to what `st` shows up.
        fbin: the build `mid` is pinned to, which its footprint is measured on.
        CUDA devices are still matched through the router's binary (ik_llama.cpp
        has no --list-devices)."""
        c = config.load()
        sbin = self._d._active_server_bin(c)
        settings = self._effective(mid)
        gpus = self.gpus()
        cmap = self._cmap(gpus, sbin)
        if cmap is None:
            return _refuse("llama-server --list-devices didn't match the GPUs nvidia-smi "
                           "reports, so LlamaForge can't tell which CUDA device is which "
                           "GPU"), None
        pred = self.predict(mid, settings)
        k = self.key(mid, settings, fbin or sbin)
        p = slots.pins(settings)
        cand = {"model": mid, "need_mib": pred.get("need_mib") or 0,
                "first_gpu_mib": pred.get("first_gpu_mib") or 0,
                "ram_mib": pred.get("ram_mib") or 0,
                "pins": slots.to_smi(p, cmap), "measured": self.store.measured(k),
                "ctx_from_model": pred.get("ctx_from_model") or 0}
        pool = self.pool()
        try:
            headroom = int(c.get("slot_headroom_mib", slots.DEFAULT_HEADROOM_MIB))
        except (TypeError, ValueError):
            headroom = slots.DEFAULT_HEADROOM_MIB
        verdict = slots.plan(cand, gpus, self._running(st, gpus, cmap), headroom_mib=headroom,
                             cap=pool["models_max"] if pool else 1, role=role,
                             ram_free_mib=self.ram_free_mib(), cmap=cmap)
        verdict["confident"] = bool(pred.get("confident"))
        return verdict, k

    # ---------- public ----------
    def plan(self, mid, role="worker"):
        """The verdict a load would get now, without acting on it."""
        role = "main" if role == "main" else "worker"
        with self._lock:
            st = self._statuses()
            if st is None:
                return _refuse("the router isn't answering")
            if mid not in st:
                return _unknown(mid)
            if (st.get(mid) or {}).get("value") in RUNNING + ("loading",):
                return dict(_refuse(""), ok=True, already=True, reason="already loaded")
            fbin, err = self.pinned(mid)
            if err:
                return _refuse(err)
            return self._plan(mid, role, st, fbin)[0]

    def loaded(self):
        with self._lock:
            return self._running(self._statuses() or {})

    def devices(self):
        """{model: [nvidia-smi index]} for the models this process loaded, from
        memory: cheap enough for the dashboard's poll (loaded() may ask nvidia-smi)."""
        with self._lock:
            return {m: list(v.get("devices") or []) for m, v in self._live.items()}

    def footprints(self):
        """{model: {"gpu index": MiB}} for the models this process loaded: what
        each measured (or was predicted) on its GPUs, from memory like devices()."""
        with self._lock:
            return {m: {str(g): int(n) for g, n in (v.get("footprint") or {}).items()}
                    for m, v in self._live.items()}

    def touch(self, mid):
        """A worker that just did work is the last one to evict."""
        with self._lock:
            if mid in self._live:
                self._live[mid]["last_used"] = time.time()

    def set_main(self, mid):
        with self._lock:
            self._set_main(mid or "")

    def load(self, mid, role="main", evict=False, wait=True, keep_role=False):
        """(HTTP status, body). 409 = the planner said no (body is its verdict;
        `evict` names the workers whose unload would make room for a main load,
        and evict=True unloads them first). With wait=False the load runs on in
        the background and the body says loading. keep_role: a model that is
        up already keeps its role (a background job must not demote the main)."""
        role = "main" if role == "main" else "worker"
        with self._lock:
            if self._loading and self._loading != mid:
                return 409, _refuse(f"{self._loading} is still loading")
            st = self._statuses()
            if st is None:
                return 502, {"ok": False, "error": "the router isn't answering"}
            if mid not in st:
                return 404, _unknown(mid)
            cur = (st.get(mid) or {}).get("value")
            if cur in RUNNING or cur == "loading":
                if not keep_role:
                    self._take_role(mid, role)
                return 200, {"ok": True, "already": True, "loading": cur == "loading"}
            busy = [m for m, s in st.items() if s.get("value") == "loading"]
            if busy:
                return 409, _refuse(f"{busy[0]} is still loading")
            fbin, err = self.pinned(mid)
            if err:
                return 409, _refuse(err)
            evicted = []
            if fbin and self.pool() is None:
                # single-model mode: a swap, and the section as the user wrote it
                for m, s in st.items():
                    if m != mid and s.get("value") in RUNNING:
                        code, out = self._unload_locked(m)
                        if code != 200:
                            return code, {"ok": False, "error": out.get("error"), "evicted": evicted}
                        evicted.append(m)
                verdict, k = dict(_refuse(""), ok=True), None
            else:
                verdict, k = self._plan(mid, role, st, fbin)
                if not verdict["ok"] and verdict["evict"] and evict:
                    for m in verdict["evict"]:
                        code, out = self._unload_locked(m)
                        if code != 200:
                            return code, {"ok": False, "error": out.get("error"), "evicted": evicted}
                        evicted.append(m)
                    st = self._statuses() or {}
                    verdict, k = self._plan(mid, role, st, fbin)
                if not verdict["ok"]:
                    return 409, dict(verdict, evicted=evicted)
            self._apply(mid, verdict["place"])
            extra = {"evicted": evicted}
            if fbin:
                before = (self._used(), self._unloads)
                ok, err, port, dropped = self._start_proc(mid, fbin)
                if not ok:
                    return 500, {"ok": False, "error": err, "evicted": evicted}
                extra.update(process={"port": port, "build": fbin}, dropped=dropped)
            else:
                self._d.reload_presets()
                before = (self._used(), self._unloads)
                code, out = self._d.router("/models/load", "POST", {"model": mid})
                if code != 200:
                    return (code if code >= 400 else 502), {
                        "ok": False, "error": (out or {}).get("error") or "load failed",
                        "evicted": evicted}
            self._loading = mid
        job = (mid, role, verdict, k, before, extra)
        if wait:
            return self._finish(*job)
        threading.Thread(target=self._finish, args=job, daemon=True).start()
        return 200, dict(verdict, ok=True, loading=True, **extra)

    def _start_proc(self, mid, sbin):
        """(ok, error, port, dropped): `mid`'s section as the ini holds it now
        (placement included), spelled in `sbin`'s own arguments."""
        c = config.load()
        secs = config.read_sections(self.ini_path())
        settings = {**(secs.get("*") or {}), **(secs.get(mid) or {})}
        dst = self.help_items(sbin)
        if not dst:
            return False, f"couldn't read the arguments of {sbin} (its --help printed nothing)", None, []
        try:
            argv, dropped = slotproc.argv_for(
                sbin, mid, settings, dst, self.help_items(self._d._active_server_bin(c)),
                api_key=network_policy.effective_key(c))
        except ValueError as e:
            return False, str(e), None, []
        try:
            base = int(c.get("slot_port_base") or slotproc.PORT_BASE)
        except (TypeError, ValueError):
            base = slotproc.PORT_BASE
        ok, err, port = self.procs().start(mid, argv, port_base=base)
        return ok, err, port, dropped

    def _take_role(self, mid, role):
        if role == "main":
            self._set_main(mid)
        elif self._main() == mid:
            self._set_main("")

    def _wait_loaded(self, mid):
        seen_loading = False
        for _ in range(LOAD_TIMEOUT_S):
            s = (self._statuses() or {}).get(mid) or {}
            v = s.get("value")
            if s.get("failed") and s.get("proc"):
                return False, (f"{mid}'s llama-server exited ({slotproc.exit_reason(s.get('exit_code'))})"
                               f" before it was ready; see its log")
            if s.get("failed"):
                return False, (f"the router reports the load failed "
                               f"({slotproc.exit_reason(s.get('exit_code'))}); see the router log")
            if v in RUNNING:
                return True, ""
            if v == "loading":
                seen_loading = True
            elif v == "unloaded" and seen_loading:
                return False, "the model stopped before it finished loading; see the router log"
            self.sleep(1)
        return False, f"still loading after {LOAD_TIMEOUT_S} s"

    def _finish(self, mid, role, verdict, k, before, extra):
        try:
            ok, err = self._wait_loaded(mid)
            with self._lock:
                if not ok:
                    return 500, {"ok": False, "error": err, "evicted": extra["evicted"]}
                before, unloads = before
                after = self._used()
                fp = {g: after[g] - before[g] for g in after
                      if g in before and after[g] - before[g] >= footprint.NOISE_MIB}
                if unloads != self._unloads:
                    fp = {}                # memory freed meanwhile hides some of the load
                if verdict["devices"] and fp and k:
                    self.store.record(k, verdict["devices"], fp, "load")
                self._live[mid] = {"devices": verdict["devices"],
                                   "footprint": fp or verdict["footprint"], "key": k,
                                   "last_used": time.time()}
                self._take_role(mid, role)
                return 200, dict(verdict, ok=True, measured=fp, **extra)
        finally:
            with self._lock:
                if self._loading == mid:
                    self._loading = None

    def unload(self, mid):
        with self._lock:
            return self._unload_locked(mid)

    def _unload_locked(self, mid):
        before = self._used()
        procs = self.procs()
        if procs and procs.has(mid):
            ok, err = procs.stop(mid)        # waits for the process to exit
            if not ok:
                return 500, {"ok": False, "error": err or "couldn't stop the process"}
            self._unloads += 1
        else:
            code, out = self._d.router("/models/unload", "POST", {"model": mid})
            if code != 200:
                return code, {"ok": False, "error": (out or {}).get("error") or "unload failed"}
            self._unloads += 1
            for _ in range(UNLOAD_TIMEOUT_S):
                if ((self._statuses() or {}).get(mid) or {}).get("value") not in RUNNING + ("loading",):
                    break
                self.sleep(1)
        self.sleep(1)                      # the driver frees a dead process's memory a beat later
        after = self._used()
        freed = {g: before[g] - after[g] for g in before if g in after}
        live = self._live.pop(mid, None)
        if live and live.get("devices") and live.get("key"):
            self.store.record(live["key"], live["devices"], freed, "unload")
        if self._main() == mid:
            self._set_main("")
        return 200, {"ok": True, "freed": {g: m for g, m in freed.items() if m > 0}}
