"""Discover safetensors (transformers) models on huggingface.co for the vLLM
backend — the counterpart of hub.py's GGUF search.

Weights size = sum of *.safetensors shard sizes. Quant is read from
config.json's quantization_config (NVFP4/FP8/AWQ/GPTQ) or falls back to the
model's torch_dtype. Fit is rated against combined VRAM at vLLM's default 0.9
gpu-memory-utilization. Pure stdlib.
"""
import json, urllib.request, urllib.parse

HF = "https://huggingface.co"
UA = {"User-Agent": "LlamaForge/1.0 (+local model manager)"}
GPU_UTIL = 0.9        # vLLM default gpu-memory-utilization


def _get_json(url, timeout=25):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def search(query="", sort="downloads", limit=30):
    """Search transformers-library repos (excludes GGUF-only repos)."""
    params = {"filter": "safetensors", "library": "transformers",
              "limit": str(limit), "direction": "-1", "sort": sort}
    if query:
        params["search"] = query
    url = f"{HF}/api/models?{urllib.parse.urlencode(params)}"
    out = []
    for m in _get_json(url):
        out.append({"repo": m.get("id", ""), "downloads": m.get("downloads", 0),
                    "likes": m.get("likes", 0),
                    "updated": (m.get("lastModified") or "")[:10]})
    return out


def fit(size_bytes, vram_mib):
    """Rate weights against combined VRAM at 0.9 utilization."""
    if not vram_mib:
        return "unknown"
    usable = vram_mib * 1024 * 1024 * GPU_UTIL
    if size_bytes * 1.25 <= usable:      # weights + KV/activation headroom
        return "fits"
    if size_bytes <= usable:
        return "tight"
    return "wont"


def detect_quant(config_json):
    """NVFP4/FP8/AWQ/GPTQ from quantization_config, else dtype, else unknown."""
    qc = config_json.get("quantization_config") or {}
    algo = str(qc.get("quant_algo") or "").lower()
    method = str(qc.get("quant_method") or "").lower()
    if "nvfp4" in algo or "nvfp4" in method:
        return "nvfp4"
    for tag in ("awq", "gptq", "fp8"):
        if tag in method or tag in algo:
            return tag
    if qc:
        return method or "quantized"
    dt = str(config_json.get("torch_dtype") or "").lower()
    if "bfloat16" in dt:
        return "bf16"
    if "float16" in dt:
        return "fp16"
    return "unknown" if not dt else dt


def repo_info(repo, vram_mib=0):
    """Summed weights size + quant + fit for a repo."""
    tree = _get_json(f"{HF}/api/models/{repo}/tree/main?recursive=1")
    size = sum(f.get("size", 0) for f in tree
               if f.get("path", "").endswith(".safetensors"))
    is_st = size > 0
    try:
        cfg = _get_json(f"{HF}/{repo}/resolve/main/config.json")
    except Exception:
        cfg = {}
    quant = detect_quant(cfg)
    return {"repo": repo, "size_bytes": size, "quant": quant,
            "fit": fit(size, vram_mib), "is_safetensors": is_st}
