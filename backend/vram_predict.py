"""LlamaForge-owned adapter over the vendored vramwise physics core.

The ONLY module that imports backend/vramwise/. Maps LlamaForge's data sources
(hardware detection, GGUF metadata, HuggingFace config.json, config bandwidth
overrides) onto vramwise's Model/Hardware dataclasses, runs predict(), and
returns a normalized dict. Never raises into the request path: any missing
input degrades to a lower-confidence result with a plain-language note.
"""
import json, os, re, threading, urllib.request

import config, hardware, gguf, osplat
from vramwise import physics, catalog, constants as C

HF = "https://huggingface.co"
UA = {"User-Agent": "LlamaForge/1.0 (+local model manager)"}

_CACHE = {}
_CACHE_LOCK = threading.Lock()
_CFG_CACHE = {}            # repo -> config.json dict (one network call per repo)


def _preset_vram_bw(name):
    """Detected nvidia-smi GPU name -> catalog VRAM bandwidth (GB/s); default if unknown."""
    n = (name or "").lower()
    for junk in ("nvidia", "geforce", "rtx", "gtx", " "):
        n = n.replace(junk, "")
    for key, (_vram, bw) in catalog.GPUS.items():
        if key.replace("-", "") in n:
            return bw
    return C.DEFAULT_VRAM_BW


def build_hardware(cfg=None, gpus=None, ram_gb=None):
    """Assemble a vramwise Hardware from detection + config overrides.
    gpus/ram_gb are injectable for tests; None triggers real detection."""
    cfg = cfg if cfg is not None else config.load()
    auto_detect = gpus is None and ram_gb is None
    gpus = hardware.detect_gpus() if gpus is None else gpus
    vram_mib = sum((g.get("vram_mib") or 0) for g in gpus)
    ram_gb = hardware.detect_ram_gb() if ram_gb is None else ram_gb
    name = gpus[0]["name"] if gpus else "cpu"
    apple_unified = False
    if osplat.IS_MAC and auto_detect:
        mem_bytes = osplat.mac_mem_bytes()
        if mem_bytes:
            name = "Apple Silicon (unified memory)"
            vram_mib = mem_bytes * osplat.METAL_BUDGET / (1024 * 1024)
            ram_gb = 0.0
            apple_unified = True
    if not apple_unified:
        ram_gb = ram_gb or 16.0
    ov = (cfg or {}).get("vram_bandwidths") or {}
    return physics.Hardware(
        name=name,
        vram_gb=vram_mib / 1024.0,
        ram_gb=float(ram_gb),
        vram_bw=float(ov.get("vram_bw") or _preset_vram_bw(name)),
        ram_bw=float(ov.get("ram_bw") or C.DEFAULT_RAM_BW),
        disk_bw=float(ov.get("disk_bw") or C.DEFAULT_DISK_BW),
    )


_DEFAULT_BPW = 4.8   # unknown quant ~ q4


def _bpw_from_quant(quant):
    """(bpw, known?) from a quant label; falls back to a q4-ish default."""
    try:
        return catalog.resolve_quant(quant), True
    except (KeyError, AttributeError):
        return _DEFAULT_BPW, False


def _model_from_gguf(meta, size_bytes):
    """Build a vramwise Model from GGUF header facts + the file's real size.
    Returns (Model|None, confidence)."""
    if not size_bytes or size_bytes <= 0:
        return None, "unknown"
    bpw, known = _bpw_from_quant(meta.get("quantization") or "")
    total = size_bytes * 8.0 / bpw
    layers = int(meta.get("block_count") or 32)
    ec = meta.get("expert_count")
    eu = meta.get("expert_used_count")
    if ec and eu and ec > 1:
        active = min(total, total * (eu / ec) * 0.85 + total * 0.10)
        conf = "high" if known else "estimate"
    elif ec and ec > 1:
        active = total
        conf = "low"
    else:
        active = total
        conf = "high" if known else "estimate"
    return physics.Model(name=meta.get("name") or "model", total_params=total,
                         active_params=active, bpw=bpw, n_layers=layers), conf


def _mtime(path):
    try:
        return os.path.getmtime(path) if path and os.path.exists(path) else 0
    except OSError:
        return 0


def _hw_sig(hw):
    return (round(hw.vram_gb, 1), round(hw.ram_gb, 1), hw.vram_bw, hw.ram_bw, hw.disk_bw)


def _cache_get(key):
    with _CACHE_LOCK:
        return _CACHE.get(key)


def _cache_put(key, value):
    with _CACHE_LOCK:
        _CACHE[key] = value


def _normalize(pred, confidence, source):
    return {
        "regime": pred.regime,
        "tok_s": round(pred.tok_s, 1),
        "usability": physics.usability(pred.tok_s),
        "gpu_resident_frac": round(pred.gpu_resident_frac, 3),
        "time_budget_ms": {"disk": round(pred.t_disk_ms, 1),
                           "weight_read": round(pred.t_mem_ms, 1),
                           "compute": round(pred.t_compute_ms, 1)},
        "note": pred.note,
        "confidence": confidence,
        "source": source,
    }


def fit_label(predict):
    """Rate a model fits/tight/offload from a physics prediction, or "unknown".

    Replaces hub._fit()'s size-only guess where a real prediction exists, so the
    Discover fit badge stops contradicting the tok/s estimate beside it. The
    reported failure was a large MoE (hybrid regime, experts on CPU) that runs
    fine but got labelled "offload" purely for being bigger than VRAM. Here a
    hybrid placement that is still interactive/usable reads as "tight", not
    "offload"; only genuinely slow generation, or full disk streaming, is
    "offload". Callers fall back to hub._fit() on "unknown".
    """
    if not predict or predict.get("confidence") == "unknown":
        return "unknown"
    regime = predict.get("regime")
    use = predict.get("usability")
    if regime is None or use is None:
        return "unknown"
    if use in ("slow", "impractical"):
        return "offload"                 # too slow to matter, wherever it sits
    if regime == "gpu-resident":
        return "fits"
    if regime == "hybrid":
        return "tight"                   # partial offload but still usable (MoE sweet spot)
    return "offload"                     # streaming from disk


def _unknown(note):
    return {"regime": None, "tok_s": None, "usability": None,
            "gpu_resident_frac": 0.0,
            "time_budget_ms": {"disk": 0, "weight_read": 0, "compute": 0},
            "note": note, "confidence": "unknown", "source": None}


def predict_local(gguf_path, size_bytes=None, cfg=None, context=4096, hw=None, meta=None):
    """Estimate for a model already on disk. Never raises.
    `meta` may be injected; otherwise it is read from gguf_path."""
    try:
        if size_bytes is None and gguf_path and os.path.exists(gguf_path):
            size_bytes = os.path.getsize(gguf_path)
        hw = hw if hw is not None else build_hardware(cfg)
        if meta is None:
            meta = (gguf.metadata(gguf_path) or {}) if gguf_path else {}
        key = ("local", gguf_path, _mtime(gguf_path), size_bytes, _hw_sig(hw), context)
        hit = _cache_get(key)
        if hit is not None:
            return hit
        model, conf = _model_from_gguf(meta, size_bytes)
        if model is None:
            return _unknown("couldn't read model size")
        out = _normalize(physics.predict(model, hw, context=context), conf, "gguf")
        _cache_put(key, out)
        return out
    except Exception:
        return _unknown("prediction unavailable")


def _fetch_hf_config(repo):
    """Best-effort GET of a repo's config.json (cached per repo). None on failure."""
    if repo in _CFG_CACHE:
        return _CFG_CACHE[repo]
    try:
        req = urllib.request.Request(f"{HF}/{repo}/resolve/main/config.json", headers=UA)
        with urllib.request.urlopen(req, timeout=20) as f:
            cfgj = json.loads(f.read().decode())
    except Exception:
        cfgj = None
    _CFG_CACHE[repo] = cfgj
    return cfgj


def _estimate_total_params(cfg):
    """Rough dense param count from a transformers config, or None."""
    h = cfg.get("hidden_size") or cfg.get("n_embd")
    L = cfg.get("num_hidden_layers") or cfg.get("n_layer")
    inter = cfg.get("intermediate_size") or (4 * h if h else None)
    vocab = cfg.get("vocab_size", 128000)
    if not (h and L and inter):
        return None
    return (4 * h * h + 3 * h * inter) * L + 2 * vocab * h


def _bpw_from_gguf_name(fname):
    m = re.search(r"(ud-)?(i?q\d[\w]*|f16|bf16)", (fname or "").lower())
    return m.group(0) if m else None


def _model_from_hf(cfgj, quant, gguf_file):
    """Build a vramwise Model from a HF config.json. Returns (Model|None, confidence)."""
    total = cfgj.get("num_parameters") or _estimate_total_params(cfgj)
    if not total:
        return None, "low"
    layers = cfgj.get("num_hidden_layers") or cfgj.get("n_layer") or 32
    n_exp = (cfgj.get("num_experts") or cfgj.get("n_routed_experts")
             or cfgj.get("num_local_experts"))
    top_k = (cfgj.get("num_experts_per_tok") or cfgj.get("moe_topk")
             or cfgj.get("num_experts_per_token"))
    if n_exp and top_k:
        active = min(total, total * (top_k / n_exp) * 0.85 + total * 0.10)
    else:
        active = total
    bpw, known = _bpw_from_quant(_bpw_from_gguf_name(gguf_file) or quant)
    conf = "high" if known else "estimate"
    return physics.Model(name="model", total_params=float(total),
                         active_params=float(active), bpw=bpw, n_layers=int(layers)), conf


def _dense_from_size(size_bytes, quant, gguf_file):
    bpw, _ = _bpw_from_quant(_bpw_from_gguf_name(gguf_file) or quant)
    total = size_bytes * 8.0 / bpw
    return physics.Model(name="model", total_params=total, active_params=total,
                         bpw=bpw, n_layers=32)


def predict_remote(repo, quant="q4_k_m", gguf_file=None, size_bytes=None,
                   cfg=None, context=4096, hw=None):
    """Estimate for a repo BEFORE download, from its HF config.json. Never raises."""
    try:
        hw = hw if hw is not None else build_hardware(cfg)
        key = ("remote", repo, quant, gguf_file, _hw_sig(hw), context)
        hit = _cache_get(key)
        if hit is not None:
            return hit
        cfgj = _fetch_hf_config(repo)
        if cfgj:
            model, conf = _model_from_hf(cfgj, quant, gguf_file)
            source = "hf-config"
        else:
            model, conf, source = None, "low", "size-fallback"
        if model is None:
            if size_bytes:
                model, conf, source = _dense_from_size(size_bytes, quant, gguf_file), "low", "size-fallback"
            else:
                return _unknown("no model geometry available")
        out = _normalize(physics.predict(model, hw, context=context), conf, source)
        _cache_put(key, out)
        return out
    except Exception:
        return _unknown("prediction unavailable")
