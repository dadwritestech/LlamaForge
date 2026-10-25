"""The search space: which configurations are worth loading, and which one won.

Pure functions, no I/O. Every axis comes from the build's own --help (argspec
items), so a flag the user's llama-server lacks is never generated, and the
settings use that build's own key names (slotproc.translate maps spellings).

M1 is lossless relative to the user's current config: placement, context and
drafters change, the KV cache types only when the request allows a loss (and
those candidates are marked lossy).
"""
import hashlib
import json
import os

# Split modes that run on CUDA. `row` parses and then fails; `graph` is documented
# by some builds and rejected by their parser. Neither is ever generated.
_RUNNABLE_SPLITS = ("none", "layer", "tensor")
_SPLIT_UNITS = 32                       # tensor-split ratios are integers summing to this

# capability -> any of the flags that name it, newest spelling first
_FLAGS = {
    "split_mode": ("--split-mode",),
    "device": ("--device",),
    "tensor_split": ("--tensor-split",),
    "ngl": ("--n-gpu-layers", "--gpu-layers"),
    "fit": ("--fit",),
    "ctk": ("--cache-type-k",),
    "ctv": ("--cache-type-v",),
    "spec_type": ("--spec-type",),
    "draft_model": ("--spec-draft-model", "--model-draft"),
    "draft_device": ("--spec-draft-device", "--device-draft"),
    "draft_ngl": ("--spec-draft-ngl", "--gpu-layers-draft", "--n-gpu-layers-draft"),
    "draft_n_max": ("--spec-draft-n-max", "--draft-max", "--draft-n-max"),
    "ubatch": ("--ubatch-size",),
    "batch": ("--batch-size",),
}

# drafter kind -> (spec-type value, default window, quick windows, thorough windows)
_SPEC = {
    "dflash": ("draft-dflash", 7, (7,), (5, 7, 10)),
    "draft": ("draft-simple", 4, (4,), (4, 7)),
    "eagle3": ("draft-eagle3", 4, (4,), (4, 7)),
    "mtp": ("draft-mtp", 3, (3,), (2, 3)),
}

# what Forge decides; the user's own values for these never survive into a candidate
_OWNED = ("split-mode", "device", "tensor-split", "main-gpu", "n-gpu-layers", "gpu-layers",
          "ctx-size", "fit", "fit-ctx", "fit-target", "spec-type")

_KV_BITS = {"f32": 32, "f16": 16, "bf16": 16, "q8_0": 8, "q5_0": 5, "q5_1": 5,
            "q4_0": 4, "q4_1": 4, "iq4_nl": 4}
_LOSSY_KV = (("q8_0", "q8_0"), ("q8_0", "q4_0"))   # K is more sensitive than V

STEPS_QUICK = (-4, 0, 4, 8)
STEPS_THOROUGH = tuple(range(-8, 9, 2))

# intent -> weight per metric (decode tk/s per category, prefill tk/s, decode at depth)
WEIGHTS = {
    "coding": {"code": .55, "prose": .10, "summary": .10, "prefill": .15, "depth": .10},
    "chat": {"code": .15, "prose": .55, "summary": .10, "prefill": .10, "depth": .10},
    "long": {"code": .05, "prose": .15, "summary": .20, "prefill": .30, "depth": .30},
    "agent": {"code": .35, "prose": .05, "summary": .10, "prefill": .25, "depth": .25},
}
TIE = 0.05                              # within this of the best score, simpler wins


# ---------------------------------------------------------------- capabilities

def _find(items, flags):
    for f in flags:
        for it in items:
            if f in (it.get("flags") or ()):
                return it
    return None


def caps_from_items(items):
    """What this llama-server build can be asked for, from argspec.parse_help items."""
    keys, found = {}, {}
    for cap, flags in _FLAGS.items():
        it = _find(items, flags)
        found[cap] = it
        keys[cap] = it["key"] if it else None

    def opts(cap):
        it = found[cap]
        return list((it.get("options") or it.get("values") or ())) if it else []

    return {"keys": keys,
            "split_modes": [m for m in opts("split_mode") if m in _RUNNABLE_SPLITS],
            "spec_types": opts("spec_type"),
            "kv_types": opts("ctk")}


# ---------------------------------------------------------------- placements

def _ratio(vrams, off):
    if len(vrams) == 2:
        a = round(_SPLIT_UNITS * vrams[0] / max(sum(vrams), 1)) + off
        a = min(max(a, 2), _SPLIT_UNITS - 2)
        return f"{a},{_SPLIT_UNITS - a}"
    return ",".join(str(max(round(v / 1024), 1)) for v in vrams)


def placements(gpus, caps, steps=STEPS_QUICK):
    """[{name, settings, devices}] - where the weights go. Every placement asks for
    full offload; a model that doesn't fit should fail loudly, not crawl."""
    if not gpus:
        return []
    k = caps["keys"]
    ngl = {(k["ngl"] or "n-gpu-layers"): "999"}
    if len(gpus) == 1:
        return [{"name": "gpu", "settings": dict(ngl), "devices": [gpus[0]["index"]]}]
    out = []
    if k["device"] and "none" in caps["split_modes"]:
        for g in gpus:
            s = dict(ngl, **{k["split_mode"]: "none", k["device"]: f"CUDA{g['index']}"})
            out.append({"name": f"solo{g['index']}", "settings": s, "devices": [g["index"]]})
    if k["tensor_split"]:
        modes = [m for m in caps["split_modes"] if m != "none"] or [None]
        vrams = [g.get("vram_mib") or 1 for g in gpus]
        ratios = []
        for off in (steps if len(gpus) == 2 else (0,)):
            r = _ratio(vrams, off)
            if r not in ratios:
                ratios.append(r)
        for mode in modes:
            for r in ratios:
                s = dict(ngl, **{k["tensor_split"]: r})
                if mode:
                    s[k["split_mode"]] = mode
                out.append({"name": f"{mode or 'split'}{r}", "settings": s,
                            "devices": [g["index"] for g in gpus]})
    return out


# ---------------------------------------------------------------- candidates

def _cid(settings):
    return hashlib.sha1(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]


def _drafter_options(inv, caps):
    """[(kind, settings)] - None first: no speculative decoding."""
    k, types = caps["keys"], caps["spec_types"]
    gpus = inv.get("gpus") or []
    out = [(None, {})]
    devices = [f"CUDA{g['index']}" for g in gpus] if len(gpus) >= 2 and k["draft_device"] else [None]
    for d in inv.get("drafters") or ():
        kind = d.get("kind", "draft")
        spec = _SPEC.get(kind)
        if not spec or not k["draft_model"]:
            continue
        if spec[0] not in types and not (kind == "draft" and not k["spec_type"]):
            continue                      # an old build drafts with -md alone; dflash needs its type
        for dev in devices:
            s = {k["draft_model"]: d["path"]}
            if k["spec_type"]:
                s[k["spec_type"]] = spec[0]
            if dev:
                s[k["draft_device"]] = dev
            if k["draft_ngl"]:
                s[k["draft_ngl"]] = "99"
            if k["draft_n_max"]:
                s[k["draft_n_max"]] = str(spec[1])
            out.append((kind, s))
    if inv.get("mtp") and "draft-mtp" in types:
        s = {k["spec_type"]: "draft-mtp"}
        if k["draft_n_max"]:
            s[k["draft_n_max"]] = str(_SPEC["mtp"][1])
        out.append(("mtp", s))
    return out


def _kv_options(base, caps, allow_loss):
    k = caps["keys"]
    bk = base.get(k["ctk"] or "cache-type-k") or "f16"
    bv = base.get(k["ctv"] or "cache-type-v") or "f16"
    out = [(bk, bv, False)]
    if allow_loss and k["ctk"] and k["ctv"]:
        have = set(caps["kv_types"])
        base_bits = _KV_BITS.get(bk, 16) + _KV_BITS.get(bv, 16)
        for ck, cv in _LOSSY_KV:
            if {ck, cv} <= have and (ck, cv) != (bk, bv) and \
                    _KV_BITS[ck] + _KV_BITS[cv] < base_bits:
                out.append((ck, cv, True))
    return out


def candidates(inv, req, drafters=True):
    """Fit-level candidates: placement x drafter (and where it runs) x KV types, at the
    requested context. Draft windows are expanded later (spec_variants) - they don't
    change memory, so the fit stage needn't load each one. drafters=False: plain
    placements only (the staged search adds drafters to the best of them)."""
    caps = inv["caps"]
    k = caps["keys"]
    owned = set(_OWNED) | {v for c, v in k.items() if c.startswith("draft") and v}
    base = {key: v for key, v in (inv.get("base") or {}).items() if key not in owned}
    steps = STEPS_THOROUGH if req.get("time") == "thorough" else STEPS_QUICK
    out = []
    for p in placements(inv.get("gpus") or [], caps, steps):
        for kind, dset in (_drafter_options(inv, caps) if drafters else [(None, {})]):
            for ck, cv, lossy in _kv_options(inv.get("base") or {}, caps, req.get("allow_loss")):
                s = dict(base)
                s.update(p["settings"])
                s["ctx-size"] = str(req["ctx"])
                if k["fit"]:
                    s[k["fit"]] = "off"     # a silent fit-down would fake a pass
                s[k["ctk"] or "cache-type-k"] = ck
                s[k["ctv"] or "cache-type-v"] = cv
                s.update(dset)
                out.append({"id": _cid(s), "label": label(s, inv.get("gpus") or []),
                            "settings": s, "lossy": lossy, "ctx": int(req["ctx"]),
                            "devices": p["devices"], "spec": kind,
                            "parts": len(p["devices"]) + (1 if kind else 0)})
    return out


def with_drafters(cand, inv):
    """A plain candidate with each drafter (and each place it can run) added."""
    out = []
    for kind, dset in _drafter_options(inv, inv["caps"])[1:]:
        s = dict(cand["settings"])
        s.update(dset)
        out.append(dict(cand, id=_cid(s), label=label(s, inv.get("gpus") or []),
                        settings=s, spec=kind, parts=cand.get("parts", 1) + 1))
    return out


def spec_variants(cand, time="quick"):
    """The candidate at each draft window worth timing (itself when it doesn't draft)."""
    kind = cand.get("spec")
    if not kind:
        return [cand]
    n_key = next((key for key in cand["settings"] if key in _FLAGS_N_MAX_KEYS), None)
    if not n_key:
        return [cand]
    out = []
    for n in _SPEC[kind][3 if time == "thorough" else 2]:
        s = dict(cand["settings"], **{n_key: str(n)})
        out.append(dict(cand, id=_cid(s), settings=s))
    return out


_FLAGS_N_MAX_KEYS = ("spec-draft-n-max", "draft-max", "draft-n-max")


# ---------------------------------------------------------------- pruning

def _dev_index(v):
    v = str(v or "")
    return int(v[4:]) if v.upper().startswith("CUDA") and v[4:].isdigit() else None


def split_estimate(settings, total_mib, draft_mib, n_gpus):
    """Per-device MiB for a candidate from one whole-model estimate: a solo
    placement carries it all, a split shares it by the ratio; the drafter adds
    to its own device."""
    per = [0] * max(n_gpus, 1)
    mode = settings.get("split-mode")
    solo = _dev_index(settings.get("device"))
    if mode == "none" and solo is not None and solo < len(per):
        per[solo] = total_mib
    elif n_gpus > 1:
        try:
            r = [float(x) for x in str(settings.get("tensor-split") or "").split(",")]
        except ValueError:
            r = []
        if len(r) != n_gpus or sum(r) <= 0:
            r = [1.0] * n_gpus
        per = [total_mib * x / sum(r) for x in r]
    else:
        per[0] = total_mib
    if draft_mib:
        d = _dev_index(settings.get("spec-draft-device") or settings.get("device-draft"))
        per[d if d is not None and d < len(per) else 0] += draft_mib
    return [int(round(x)) for x in per]


def prune(cands, estimate, gpus, headroom=0.97):
    """(keep, rejected[{id, label, reason}]). estimate(c) -> MiB per device, or None
    when unknown (then the load probe decides)."""
    keep, rej = [], []
    for c in cands:
        est = estimate(c)
        bad = None
        for i, need in enumerate(est or ()):
            if i < len(gpus) and gpus[i].get("vram_mib") and need > headroom * gpus[i]["vram_mib"]:
                g = gpus[i]
                bad = (f"needs ~{need / 1024:.1f} GB on GPU{g['index']} "
                       f"({g['name']}, {g['vram_mib'] / 1024:.1f} GB)")
                break
        if bad:
            rej.append({"id": c["id"], "label": c.get("label", ""), "reason": bad})
        else:
            keep.append(c)
    return keep, rej


# ---------------------------------------------------------------- ranking

def _scores(rows, intent):
    w = WEIGHTS.get(intent, WEIGHTS["coding"])
    best = {m: max((r.get(m) or 0) for r in rows) for m in w}
    out = []
    for r in rows:
        num = den = 0.0
        for m, wt in w.items():
            if r.get(m) and best[m] > 0:
                num += wt * r[m] / best[m]
                den += wt
        out.append(num / den if den else 0.0)
    return out


def rank(rows, intent):
    """Rows best first, each with "score" (0..1, relative to the best per metric).
    Within TIE of the best, the winner is the one with less VRAM, then fewer
    moving parts, then more context - a near-tie isn't worth complexity."""
    if not rows:
        return []
    scored = [dict(r, score=round(s, 4)) for r, s in zip(rows, _scores(rows, intent))]
    scored.sort(key=lambda r: -r["score"])
    top = scored[0]["score"]
    near = [r for r in scored if r["score"] >= top * (1 - TIE)]
    win = min(near, key=lambda r: (r.get("vram_mib") or 0, r.get("parts") or 0, -(r.get("ctx") or 0)))
    return [win] + [r for r in scored if r is not win]


# ---------------------------------------------------------------- words

def _gpu_name(gpus, dev):
    i = _dev_index(dev)
    for g in gpus:
        if g["index"] == i:
            return g["name"]
    return str(dev)


def _kv_words(k, v):
    bk, bv = _KV_BITS.get(k or "f16", 16), _KV_BITS.get(v or "f16", 16)
    return f"KV {bk}-bit" if bk == bv else f"KV {bk}/{bv}-bit"


def label(s, gpus):
    """"RTX 5080 alone + drafter on RTX 5060 Ti, KV 8-bit, 32k"."""
    mode = s.get("split-mode")
    if mode == "none" and s.get("device"):
        where = f"{_gpu_name(gpus, s['device'])} alone"
    elif s.get("tensor-split"):
        where = f"{mode or 'layer'} split {s['tensor-split'].replace(',', ':')}"
    else:
        where = gpus[0]["name"] if gpus else "GPU"
    draft_dev = s.get("spec-draft-device") or s.get("device-draft")
    draft = s.get("spec-draft-model") or s.get("model-draft")
    if draft:
        name = os.path.splitext(os.path.basename(str(draft).replace("\\", "/")))[0]
        n = next((s[k] for k in _FLAGS_N_MAX_KEYS if s.get(k)), None)
        where += f" + drafter {name}" + (f" ({n} ahead)" if n else "")
        where += f" on {_gpu_name(gpus, draft_dev)}" if draft_dev else ""
    elif s.get("spec-type") == "draft-mtp":
        where += " + MTP"
    parts = [where, _kv_words(s.get("cache-type-k"), s.get("cache-type-v"))]
    ctx = int(s.get("ctx-size") or 0)
    if ctx:
        parts.append(f"{ctx // 1024}k")
    return ", ".join(parts)
