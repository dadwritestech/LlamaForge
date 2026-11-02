"""Multi-model slots: where a model goes, and whether it fits at all.

With more than one model loaded, the router's own answer to "full" is to evict
the least recently used model, and llama.cpp's --fit (when it is not disabled
by a pinned n-gpu-layers) quietly shrinks the newcomer's context. Neither is
acceptable when the newcomer is a background job and the evictee is the model
the user is chatting with. So every multi-mode load asks plan() first.

plan() is pure: it sees one nvidia-smi snapshot (whose `used` already counts
the desktop, other apps and every loaded slot), the slots that are up, and the
candidate's memory need, and returns a verdict. A footprint measured on a
previous load of the same configuration beats the prediction; the prediction
only covers configurations never loaded before.

Measured on this project's dev box (2026-10-04): a process pinned to GPU1 still
opens a CUDA context on GPU0 (~231 MiB); one on GPU0 costs nothing on GPU1. A
vision projector lands on CUDA0 too, whatever the model's `device` says.

Pure stdlib, no I/O.
"""
import re

MIB_PER_GIB = 1024
DEFAULT_HEADROOM_MIB = 1536   # per GPU, kept free on top of everything planned
OVERHEAD_MIB = 700            # per GPU a process uses: CUDA context + compute buffers
OTHER_GPU0_MIB = 256          # a process pinned off GPU0 still opens a context there
RAM_HEADROOM_MIB = 2048

_CUDA = re.compile(r"cuda(\d+)", re.I)


def _floats(s):
    try:
        return [float(x) for x in str(s).split(",") if x.strip() != ""]
    except ValueError:
        return None


def pins(settings):
    """What a model's effective settings ([*] merged with its section) already
    fix about placement: {"devices": [gpu index], "share": {index: fraction}},
    devices None when they name something we can't map (never "free to place"),
    or None when nothing is pinned and the planner may choose."""
    s = {k: str(v).strip() for k, v in (settings or {}).items() if v is not None}
    devices = None
    if s.get("device"):
        names = [n.strip() for n in s["device"].split(",") if n.strip()]
        if [n.lower() for n in names] == ["none"]:
            return {"devices": [], "share": {}}
        idx = [_CUDA.fullmatch(n) for n in names]
        if not idx or not all(idx):
            return {"devices": None, "share": {}}
        devices = [int(m.group(1)) for m in idx]
    if s.get("split-mode", "").lower() == "none":
        try:
            mg = int(s.get("main-gpu") or 0)
        except ValueError:
            return {"devices": None, "share": {}}
        if devices is not None:
            if not 0 <= mg < len(devices):
                return {"devices": None, "share": {}}
            mg = devices[mg]
        return {"devices": [mg], "share": {mg: 1.0}}
    ts = _floats(s["tensor-split"]) if s.get("tensor-split") else None
    if ts is None and s.get("tensor-split"):
        return {"devices": None, "share": {}}
    if ts:
        order = devices if devices is not None and len(devices) == len(ts) else (
            None if devices is not None else list(range(len(ts))))
        if order is not None and sum(ts) > 0:
            pairs = [(d, w) for d, w in zip(order, ts) if w > 0]
            total = sum(w for _, w in pairs)
            return {"devices": [d for d, _ in pairs],
                    "share": {d: w / total for d, w in pairs}}
    if devices is not None:
        return {"devices": devices, "share": {d: 1.0 / len(devices) for d in devices}}
    return None


CAP_MIN, CAP_MAX, CAP_DEFAULT = 2, 4, 3
HEADROOM_DEFAULT = 1536             # config.DEFAULTS["slot_headroom_mib"]


def router_pool(cfg, no_autoload_supported, margin_supported=False):
    """router_ctl.start's pool for this config: None (one model at a time, as
    always) unless multi_model is on. A build without --no-models-autoload
    stays single unless the user opted into autoload anyway: otherwise any
    client request could load a model the planner never saw.

    With autoload on, a build that has --models-memory-margin (llama.cpp
    v0.8.0) also gets slot_headroom_mib as the router's own margin, so a
    client-loaded model unloads the least recently used one when VRAM runs
    short instead of failing to fit. With autoload off the planner already
    keeps that headroom, and a second, router-side eviction rule would only
    fight it."""
    if not cfg.get("multi_model"):
        return None
    autoload = bool(cfg.get("slot_autoload"))
    if not autoload and not no_autoload_supported:
        return None
    try:
        cap = int(cfg.get("slot_cap", CAP_DEFAULT))
    except (TypeError, ValueError):
        cap = CAP_DEFAULT
    pool = {"models_max": min(max(cap, CAP_MIN), CAP_MAX), "autoload": autoload}
    if autoload and margin_supported:
        try:
            margin = int(cfg.get("slot_headroom_mib", HEADROOM_DEFAULT))
        except (TypeError, ValueError):
            margin = HEADROOM_DEFAULT
        if margin > 0:
            pool["memory_margin_mib"] = margin
    return pool


_LIST_DEVICE = re.compile(r"^\s*CUDA(\d+):\s*(.+?)\s*\(", re.M)


def cuda_map(list_devices, smi_gpus):
    """{CUDA index: nvidia-smi index} from `llama-server --list-devices` output.

    CUDA numbers devices fastest-first unless CUDA_DEVICE_ORDER says otherwise;
    nvidia-smi numbers them by PCI bus. They agree on most boxes, but not on one
    with the slower card in the first slot, and forcing PCI order on the router
    would silently re-aim every device / main-gpu the user already pinned. So
    LlamaForge leaves the order alone and translates. Identical cards keep
    their relative order (CUDA breaks speed ties by PCI bus). None when the two
    lists can't be matched one to one."""
    found = [(int(i), name.strip()) for i, name in _LIST_DEVICE.findall(list_devices or "")]
    if not found or len(found) != len(smi_gpus):
        return None
    left = sorted(smi_gpus, key=lambda g: g["index"])
    out = {}
    for ci, name in found:
        hit = next((g for g in left if g.get("name", "").strip() == name), None)
        if hit is None:
            return None
        left.remove(hit)
        out[ci] = hit["index"]
    return out


def to_smi(p, cmap):
    """pins() (CUDA indices) -> the same pins in nvidia-smi indices; a device the
    map doesn't know makes the pins unmappable, never free to place."""
    if p is None or not p.get("devices"):
        return p
    if any(d not in cmap for d in p["devices"]):
        return {"devices": None, "share": {}}
    return {"devices": [cmap[d] for d in p["devices"]],
            "share": {cmap[d]: s for d, s in p["share"].items()}}


def cuda_name(smi_index, cmap):
    """nvidia-smi index -> the device name llama-server wants ("CUDA0")."""
    if not cmap:
        return f"CUDA{smi_index}"
    back = {v: k for k, v in cmap.items()}
    return f"CUDA{back[smi_index]}"


def _first(cmap):
    """nvidia-smi index of CUDA0: where every process opens its context and
    mtmd puts the projector."""
    return cmap.get(0, 0) if cmap else 0


def _gib(mib):
    return f"{mib / MIB_PER_GIB:.1f} GiB"


def _predicted(cand, devices, share, first):
    """Per-GPU MiB for a placement: the need (which already holds one process
    overhead) plus an overhead per extra device, shared like the weights; then
    on CUDA0 (`first`, as nvidia-smi numbers it) the projector, and the context
    a process placed elsewhere still opens there."""
    need = cand.get("need_mib") or 0
    extra = OVERHEAD_MIB * max(len(devices) - 1, 0)
    fp = {d: int(round((need + extra) * share[d])) for d in devices}
    if devices and first not in devices:
        fp[first] = OTHER_GPU0_MIB
    if devices and cand.get("first_gpu_mib"):
        fp[first] = fp.get(first, 0) + cand["first_gpu_mib"]
    return fp


def _footprint(cand, devices, share, first):
    m = (cand.get("measured") or {}).get(tuple(devices))
    if m:
        return dict(m), "measured"
    return _predicted(cand, devices, share, first), "predicted"


def _fits(fp, free):
    return all(i in free and mib <= free[i] for i, mib in fp.items())


def _speed_rank(i, cmap):
    """CUDA numbers devices fastest-first (unless CUDA_DEVICE_ORDER says
    otherwise), so a GPU's CUDA index is its speed rank."""
    if not cmap:
        return i
    return {v: k for k, v in cmap.items()}.get(i, i)


def _choose(cand, free, role, allow_span, cmap, main_devs=()):
    """(devices, place, footprint, source) for an unpinned candidate, or None.
    main: the fastest GPU that fits - it's the model someone is waiting on.
    worker: off the main's GPUs when it can (two models generating on one GPU
    share its memory bandwidth), and there the tightest fit (keeps the big hole)."""
    first = _first(cmap)
    options = []
    for i in sorted(free):
        fp, src = _footprint(cand, [i], {i: 1.0}, first)
        if _fits(fp, free):
            rank = ((_speed_rank(i, cmap),) if role == "main"
                    else (i in main_devs, free[i]))
            options.append((rank, i, fp, src))
    if options:
        _, i, fp, src = min(options)
        return [i], cuda_name(i, cmap), fp, src
    if allow_span and len(free) > 1:
        devs = sorted(free)
        room = {d: max(free[d], 0) for d in devs}
        if sum(room.values()) > 0:
            share = {d: room[d] / sum(room.values()) for d in devs}
            devs = [d for d in devs if share[d] > 0]
            fp, src = _footprint(cand, devs, share, first)
            if _fits(fp, free):
                return devs, "", fp, src
    return None


def _place(cand, free, role, cmap, main_devs=()):
    """(devices, place, footprint, source) or None; place None = pinned, untouched."""
    p = cand.get("pins")
    if p is None:
        return _choose(cand, free, role, (role == "main"), cmap, main_devs)
    fp, src = _footprint(cand, p["devices"], p["share"], _first(cmap))
    return (p["devices"], None, fp, src) if _fits(fp, free) else None


def _verdict(ok, reason="", devices=None, place=None, footprint=None, source=None,
             evict=None, already=False):
    return {"ok": ok, "reason": reason, "devices": devices or [], "place": place,
            "footprint": footprint or {}, "source": source, "evict": evict or [],
            "already": already}


def _refusal(cand, free, headroom_mib, cmap, alone=False):
    first = _first(cmap)
    p = cand.get("pins")
    if p is not None and p["devices"]:
        fp, src = _footprint(cand, p["devices"], p["share"], first)
        missing = [i for i in fp if i not in free]
        if missing:
            return f"it is pinned to GPU{missing[0]}, which this machine doesn't have"
        need = sum(fp.values())
    else:
        fp, src = _footprint(cand, [min(free)], {min(free): 1.0}, first)
        need = sum(fp.values())
    where = ", ".join(f"GPU{i} has {_gib(max(f, 0))}" for i, f in sorted(free.items()))
    why = (f"needs ~{_gib(need)} ({src}); {where} free after "
           f"{_gib(headroom_mib)} headroom")
    if cand.get("ctx_from_model"):
        # llama-server's ctx-size 0 is the model's own context, and --fit then
        # grows/shrinks it to fill the GPU - either way nothing fits beside it
        why += (f". ctx-size isn't set, so llama-server takes the model's full "
                f"{cand['ctx_from_model']:,}-token context; set a ctx-size "
                + ("so it fits your GPUs" if alone else "to run it beside other models"))
    return why


def plan(candidate, gpus, loaded, headroom_mib=DEFAULT_HEADROOM_MIB, cap=3,
         role="worker", ram_free_mib=None, cmap=None):
    """Verdict for loading `candidate` next to `loaded`.

    candidate: {model, need_mib, first_gpu_mib (the projector, on CUDA0), pins (from
                pins(), already in nvidia-smi indices), measured: {(devices,): {gpu: MiB}},
                ram_mib (the part kept in system RAM)}
    gpus:      [{index, total_mib, used_mib, free_mib}] - one nvidia-smi snapshot
    loaded:    [{model, role, footprint: {gpu: MiB}, last_used}]
    cmap:      cuda_map(); None means CUDA and nvidia-smi number the GPUs alike.
    Every index here is nvidia-smi's; only `place` is in CUDA terms.
    Returns {ok, reason, devices, place, footprint, source, evict, already}.
    place: "CUDAn" to write, "" to clear ours (span), None to leave the section alone.
    evict: workers (oldest first) whose unload would let a *main* load fit. Workers
    never get an evict list: a background job never unloads anything."""
    model = candidate.get("model")
    if any(s.get("model") == model for s in loaded):
        return _verdict(True, "already loaded", already=True)
    p = candidate.get("pins")
    if p is not None and p["devices"] is None:
        return _verdict(False, "its device setting names a device LlamaForge can't map to a "
                               "GPU; set device = CUDA0 / CUDA1, or clear it")
    ram = candidate.get("ram_mib") or 0
    if ram and ram_free_mib is not None and ram > ram_free_mib - RAM_HEADROOM_MIB:
        return _verdict(False, f"needs ~{_gib(ram)} of system RAM for the part kept on the "
                               f"CPU; {_gib(max(ram_free_mib, 0))} is free")
    if p is not None and not p["devices"]:
        if len(loaded) >= cap:
            return _verdict(False, f"{len(loaded)} models are loaded; the limit is {cap}")
        return _verdict(True, devices=[], place=None, footprint={}, source="predicted")
    if not gpus:
        return _verdict(False, "no GPU memory information (nvidia-smi didn't answer)")

    free = {g["index"]: (g["free_mib"] if g.get("free_mib") is not None
                         else g["total_mib"] - g["used_mib"]) - headroom_mib for g in gpus}
    over_cap = len(loaded) >= cap
    main_devs = {d for s in loaded if s.get("role") == "main" for d in s.get("devices") or ()}
    if not over_cap:
        got = _place(candidate, free, role, cmap, main_devs)
        if got:
            devs, place, fp, src = got
            return _verdict(True, devices=devs, place=place, footprint=fp, source=src)

    reason = (f"{len(loaded)} models are loaded; the limit is {cap}" if over_cap
              else _refusal(candidate, free, headroom_mib, cmap, alone=not loaded))
    if role != "main":
        return _verdict(False, reason)
    workers = sorted((s for s in loaded if s.get("role") != "main"),
                     key=lambda s: s.get("last_used") or 0)
    start = max(len(loaded) - cap + 1, 1)
    for k in range(start, len(workers) + 1):
        room = dict(free)
        for s in workers[:k]:
            for i, mib in (s.get("footprint") or {}).items():
                if i in room:
                    room[i] += mib
        got = _place(candidate, room, role, cmap)
        if got:
            names = [s["model"] for s in workers[:k]]
            devs, place, fp, src = got
            return _verdict(False, f"{reason}. Unloading {', '.join(names)} makes room",
                            devices=devs, place=place, footprint=fp, source=src, evict=names)
    return _verdict(False, reason)
