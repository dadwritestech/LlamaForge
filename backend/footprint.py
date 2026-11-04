"""How much GPU and system memory one model load takes: predicted from the GGUF,
or measured on an earlier load of the same configuration.

The prediction mirrors where llama.cpp puts things: the input embedding always
stays in system RAM, n-gpu-layers offloads the *last* N blocks (the output layer
only when N exceeds the block count), cpu-moe / n-cpu-moe / override-tensor pull
matching tensors back to the CPU, and each block's KV cache lives with its
block. Hybrid models (Qwen3-Next/3.5, Nemotron-H) only have KV on their
attention layers, sliding-window layers only cache the window, and layers that
share KV (Gemma 3n/4 E-models) or predict ahead (MTP) cache nothing.

A prediction is only the first guess. A footprint measured around a real load
(per-GPU used before vs after) replaces it, and the memory an unload frees is
the better reading still: the CUDA pool grows on the first long prompt, so what
comes back at unload is what the model really held.

Stdlib only.
"""
import hashlib
import json
import os
import re
import threading

import atomicio
import slots
from vramwise import constants as _vw

MIB = 1024 * 1024
NOISE_MIB = 64          # per-GPU deltas below this are other apps breathing
MMPROJ_COMPUTE_MIB = 320   # the projector's compute buffer; measured 177 (gemma) - 252 (qwen)

_KV_BYTES = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32,
             "q4_1": 20 / 32, "iq4_nl": 18 / 32, "q5_0": 22 / 32, "q5_1": 24 / 32}
# Archs whose sliding-window layout lives in llama.cpp's code, not in the GGUF:
# every Nth layer is global, the rest slide.
_SWA_PERIOD = {"gemma2": 2, "gemma3": 6, "cohere2": 4, "gpt-oss": 2}
_EXPS = r"\.ffn_(up|down|gate|gate_up)_(ch|)exps"
_ALIASES = {"c": "ctx-size", "ngl": "n-gpu-layers", "gpu-layers": "n-gpu-layers",
            "ot": "override-tensor", "ctk": "cache-type-k", "ctv": "cache-type-v",
            "ncmoe": "n-cpu-moe", "cmoe": "cpu-moe", "np": "parallel",
            "ub": "ubatch-size", "dev": "device"}
_PLACEMENT = {"device", "dev", "split-mode", "sm", "main-gpu", "mg", "tensor-split", "ts"}


def _norm(settings):
    out = {}
    for k, v in (settings or {}).items():
        if v is None:
            continue
        k = str(k).strip().lower()
        out[_ALIASES.get(k, k)] = str(v).strip()
    return out


def _flag(s, key):
    v = s.get(key)
    return v is not None and v.lower() not in ("0", "false", "off", "no", "disabled")


def _enabled(s, key):
    """A default-on llama.cpp switch: off by `key = false` or by `no-key`."""
    return not (_flag(s, "no-" + key) or s.get(key, "").lower()
                in ("0", "false", "off", "no", "disabled"))


def _int(s, key, default):
    try:
        return int(float(s[key]))
    except (KeyError, ValueError):
        return default


def _overrides(s):
    """[(compiled regex, on_cpu)] in llama.cpp's first-match order, or None if a
    pattern doesn't compile (then the RAM/VRAM split is a guess)."""
    rules = []
    for part in (s.get("override-tensor") or "").split(","):
        if "=" not in part:
            continue
        pat, buft = part.rsplit("=", 1)
        try:
            rules.append((re.compile(pat.strip()), buft.strip().upper().startswith("CPU")))
        except re.error:
            return None
    if _flag(s, "cpu-moe"):
        rules.append((re.compile(_EXPS), True))
    for i in range(_int(s, "n-cpu-moe", 0)):
        rules.append((re.compile(rf"blk\.{i}{_EXPS}"), True))
    return rules


def _kv_layers(kv, n_layer):
    """Per block: (k_elems, v_elems, swa) per token, or None for no KV cache."""
    hkv = kv.get("head_count_kv")
    heads = kv.get("head_count") or 1
    emb = kv.get("embedding_length") or 0
    kl = kv.get("key_length") or (emb // heads if heads else 0)
    vl = kv.get("value_length") or kl
    pattern = kv.get("sliding_window_pattern")
    window = kv.get("sliding_window")
    interval = kv.get("full_attention_interval")
    n_kv = n_layer - (kv.get("nextn_predict_layers") or 0) - (kv.get("shared_kv_layers") or 0)
    out = []
    for i in range(n_layer):
        h = hkv[i] if isinstance(hkv, list) and i < len(hkv) else (
            hkv if isinstance(hkv, int) else heads)
        if i >= n_kv or not h or (interval and (i + 1) % interval):
            out.append(None)
            continue
        if not window:
            swa = False
        elif isinstance(pattern, list) and i < len(pattern):
            swa = bool(pattern[i])
        else:
            period = _SWA_PERIOD.get(kv.get("_arch", ""))
            swa = bool(period) and i % period < period - 1
        k = kv.get("key_length_swa", kl) if swa else kl
        v = kv.get("value_length_swa", vl) if swa else vl
        out.append((h * k, h * v, swa))
    return out


def _mib(b):
    return int(round(b / MIB))


def _projector(s, mmproj_bytes, cpu_only):
    """(first-GPU MiB, RAM MiB) for the mmproj: mtmd puts it, with its compute
    buffer, on the first GPU backend whatever `device` says."""
    if not mmproj_bytes:
        return 0, 0
    if cpu_only or not _enabled(s, "mmproj-offload"):
        return 0, _mib(mmproj_bytes)
    return _mib(mmproj_bytes) + MMPROJ_COMPUTE_MIB, 0


def placer(settings, layout):
    """(on_gpu(name) -> bool, info) for one model's weight tensors, by the rules in
    this module's docstring. info: n_layer, n_gpu (blocks + output offloaded),
    first_gpu (lowest block on the GPU), output_on_gpu, cpu_only, confident (False
    when an override-tensor pattern doesn't compile, so the split is a guess)."""
    s = _norm(settings)
    cpu_only = (s.get("device") or "").lower() == "none"
    n_layer = int((layout.get("kv") or {}).get("block_count") or 0)
    ngl = s.get("n-gpu-layers", "").lower()
    if cpu_only:
        n_gpu = 0
    elif ngl in ("", "all", "auto") or _int(s, "n-gpu-layers", -1) < 0:
        n_gpu = n_layer + 1
    else:
        n_gpu = _int(s, "n-gpu-layers", n_layer + 1)
    first_gpu = max(n_layer - n_gpu, 0)
    output_on_gpu = n_gpu > n_layer
    rules = _overrides(s)
    confident = rules is not None
    rules = rules or []

    def on_gpu(name):
        m = re.match(r"blk\.(\d+)\.", name)
        if name.startswith(("token_embd", "per_layer_token_embd")):
            gpu = False
        elif m:
            gpu = int(m.group(1)) >= first_gpu and n_gpu > 0
        else:
            gpu = output_on_gpu
        for rx, to_cpu in rules:
            if rx.search(name):
                return not to_cpu and not cpu_only
        return gpu

    return on_gpu, {"n_layer": n_layer, "n_gpu": n_gpu, "first_gpu": first_gpu,
                    "output_on_gpu": output_on_gpu, "cpu_only": cpu_only,
                    "confident": confident}


def predict(settings, layout, extra_gpu_bytes=0, size_bytes=None, mmproj_bytes=0):
    """{need_mib (GPU, incl. one process overhead), first_gpu_mib, ram_mib,
    weights_gpu_mib, weights_ram_mib, kv_gpu_mib, kv_ram_mib, staging_mib, ctx,
    ctx_from_model (ctx when ctx-size is unset and the file's own was used, else 0),
    confident}.
    settings: the model's effective settings ([*] merged with its section).
    layout: gguf.layout(); None falls back to the file size and a generous KV.
    extra_gpu_bytes: what goes where the model goes (a draft model).
    first_gpu_mib: the projector's share, charged to CUDA0 by the planner."""
    s = _norm(settings)
    cpu_only = (s.get("device") or "").lower() == "none"
    first_mib, proj_ram = _projector(s, mmproj_bytes, cpu_only)
    if not layout:
        ctx = _int(s, "ctx-size", 0) or 8192
        kv = _vw.KV_GB_PER_1K_CONTEXT * ctx / 1000 * 1e9
        need = (size_bytes or 0) + kv + extra_gpu_bytes
        return {"need_mib": _mib(need) + slots.OVERHEAD_MIB, "first_gpu_mib": first_mib,
                "ram_mib": proj_ram, "weights_gpu_mib": _mib(size_bytes or 0),
                "weights_ram_mib": 0, "kv_gpu_mib": _mib(kv), "kv_ram_mib": 0,
                "staging_mib": 0, "ctx": ctx, "ctx_from_model": 0, "confident": False}

    kv = dict(layout.get("kv") or {}, _arch=layout.get("arch") or "")
    place, info = placer(s, layout)
    n_layer, n_gpu, first_gpu = info["n_layer"], info["n_gpu"], info["first_gpu"]
    output_on_gpu, confident = info["output_on_gpu"], info["confident"]

    w_gpu = w_ram = 0
    names = set()
    embd = 0
    cpu_blocks = {}                         # block -> bytes of it left on the CPU
    for name, _typ, nbytes in layout.get("tensors") or []:
        names.add(name)
        m = re.match(r"blk\.(\d+)\.", name)
        if name.startswith("token_embd."):
            embd += nbytes
        if place(name):
            w_gpu += nbytes
        else:
            w_ram += nbytes
            if m:
                cpu_blocks[m.group(1)] = cpu_blocks.get(m.group(1), 0) + nbytes
    if output_on_gpu and "output.weight" not in names:
        w_gpu += embd                       # tied output: llama.cpp copies the embedding

    ctx = _int(s, "ctx-size", 0) or int(kv.get("context_length") or 4096)
    n_seq = max(_int(s, "parallel", 4), 1)
    ubatch = _int(s, "ubatch-size", 512)
    swa_cells = min(ctx, -(-(int(kv.get("sliding_window") or 0) * n_seq + ubatch) // 256) * 256)
    if _flag(s, "swa-full"):
        swa_cells = ctx
    kb = _KV_BYTES.get(s.get("cache-type-k", "f16").lower(), 2.0)
    vb = _KV_BYTES.get(s.get("cache-type-v", "f16").lower(), 2.0)
    kv_off = _enabled(s, "kv-offload")
    kv_gpu = kv_ram = 0.0
    for i, lay in enumerate(_kv_layers(kv, n_layer)):
        if lay is None:
            continue
        k, v, swa = lay
        b = (swa_cells if swa else ctx) * (k * kb + v * vb)
        if kv_off and n_gpu > 0 and i >= first_gpu:
            kv_gpu += b
        else:
            kv_ram += b

    # op offload runs prompt batches on the GPU even for CPU-resident weights,
    # copying one block's worth at a time into the compute buffer
    staging = (max(cpu_blocks.values()) if cpu_blocks and not cpu_only and n_gpu > 0
               and _enabled(s, "op-offload") else 0)
    gpu = 0 if cpu_only else (_mib(w_gpu) + _mib(kv_gpu) + _mib(extra_gpu_bytes)
                              + _mib(staging) + slots.OVERHEAD_MIB)
    ram = _mib(w_ram) + _mib(kv_ram) + (_mib(extra_gpu_bytes) if cpu_only else 0) + proj_ram
    return {"need_mib": gpu, "first_gpu_mib": first_mib, "ram_mib": ram,
            "weights_gpu_mib": _mib(w_gpu), "weights_ram_mib": _mib(w_ram),
            "kv_gpu_mib": _mib(kv_gpu), "kv_ram_mib": _mib(kv_ram),
            "staging_mib": _mib(staging), "ctx": ctx,
            "ctx_from_model": 0 if _int(s, "ctx-size", 0) else ctx, "confident": confident}


_MEM_EXACT = {"c", "ngl", "ot", "np", "ub", "b", "fa", "ctk", "ctv", "md", "ncmoe",
              "cmoe", "m", "hf", "hfr", "mmproj", "parallel"}
_MEM_PARTS = ("ctx", "cache", "gpu-layers", "moe", "override-tensor", "mmproj", "batch",
              "flash", "swa", "kv", "draft", "model", "fit", "lora", "rope", "yarn",
              "spec", "mtp", "nextn")


def key(model, build, settings):
    """Identity of a load for the footprint store: the model, the build that runs
    it, and every setting that changes memory. Sampling and placement are left
    out (placement is the second level of the store)."""
    rel = sorted((k, v) for k, v in (
        (str(k).strip().lower(), str(v).strip()) for k, v in (settings or {}).items()
        if v is not None)
        if k not in _PLACEMENT and (k in _MEM_EXACT or any(p in k for p in _MEM_PARTS)))
    h = hashlib.sha1(json.dumps(rel).encode()).hexdigest()[:12]
    return f"{model}|{build or ''}|{h}"


class Store:
    """footprints.json: {key: {"0,1": {"0": MiB, "1": MiB}}} - per load identity,
    per device set, per GPU."""

    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()
        self._data = self._read()

    def _read(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                d = json.load(f)
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def measured(self, k):
        """{(devices,): {gpu: MiB}} for slots.plan()."""
        out = {}
        with self._lock:
            for dk, fp in (self._data.get(k) or {}).items():
                try:
                    devs = tuple(int(x) for x in dk.split(",") if x != "")
                    out[devs] = {int(g): int(m) for g, m in fp.items()}
                except (ValueError, AttributeError):
                    continue
        return out

    def record(self, k, devices, fp, when="load"):
        fp = {int(g): int(m) for g, m in (fp or {}).items() if m >= NOISE_MIB}
        if not fp:
            return
        dk = ",".join(str(d) for d in devices)
        with self._lock:
            cur = self._data.setdefault(k, {})
            old = {int(g): int(m) for g, m in (cur.get(dk) or {}).items()}
            if when == "unload" and old:
                fp = {g: max(old.get(g, 0), fp.get(g, 0)) for g in set(old) | set(fp)}
            cur[dk] = {str(g): m for g, m in sorted(fp.items())}
            atomicio.write_json(self.path, self._data)
