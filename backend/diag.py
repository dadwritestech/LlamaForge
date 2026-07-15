"""Turn a raw router (llama.cpp) log tail into a one-line error + a suggested
fix for the model editor. Pure string heuristics, no I/O, so it's unit-testable
and the UI never has to make the user scroll the log panel to learn why a load
failed.
"""
import re

# Ordered most-specific first; the first rule whose pattern hits wins. Each
# suggestion may reference {ngl}/{ctx} - the model's current knob values - so the
# advice is concrete ("reduce n-gpu-layers from 99").
_RULES = [
    (r"out of memory|failed to allocate|cudamalloc failed|\boom\b",
     "Ran out of memory loading the model.",
     "Lower n-gpu-layers{ngl_from} to offload fewer layers, or reduce ctx-size{ctx_from} "
     "to shrink the KV cache."),
    (r"cuda(?:art)? error|cudamalloc|cublas|ggml_cuda",
     "GPU/CUDA error while loading.",
     "The build hit a CUDA error. Confirm the GPU has free VRAM (Models tab) and "
     "that this llama.cpp build matches your CUDA driver."),
    (r"not enough space in the context|kv[_ ]?cache|n_ctx",
     "The context is too large for available memory.",
     "Reduce ctx-size{ctx_from} - the KV cache scales with it."),
    (r"unknown argument|invalid argument|unrecognized|error: unknown",
     "The router rejected a launch flag.",
     "One of the knobs isn't supported by this llama.cpp build. Clear the most recently "
     "changed knob, or rebuild from the Build tab."),
    (r"no such file|does not exist|failed to open gguf|cannot find the file",
     "The model file could not be opened.",
     "The GGUF path is missing or moved. Re-scan drives from Setup, or fix the model path."),
    (r"unsupported|unknown model architecture|unknown (?:pre-)?tokenizer",
     "This build can't run this model.",
     "The architecture/quant isn't supported by the current build. Update llama.cpp from the Build tab."),
    (r"failed to load model|error loading model|llama_(?:model_)?load",
     "The model failed to load.",
     "Check the Router Log below for the exact llama.cpp line. Common causes: too little VRAM "
     "(reduce n-gpu-layers{ngl_from}) or a corrupt download."),
]

# Lines that are pure noise - never surface these as "the error".
_SKIP = re.compile(r"^\s*(srv|slot|main:|system_info|build:|\s*$)", re.I)


def _last_error_line(text):
    """The most relevant single line from a log tail (last error-ish line,
    else the last non-noise line)."""
    lines = [ln.rstrip() for ln in (text or "").splitlines() if ln.strip()]
    for ln in reversed(lines):
        if re.search(r"error|fail|abort|terminate|exception|panic|assert", ln, re.I):
            return ln.strip()
    for ln in reversed(lines):
        if not _SKIP.match(ln):
            return ln.strip()
    return ""


def diagnose(log_text, settings=None):
    """Return {error, suggestion} for a failed load, or None if the log shows
    no recognizable failure. `settings` supplies current knob values so the
    suggestion can name concrete numbers."""
    settings = settings or {}
    ngl = settings.get("n-gpu-layers")
    ctx = settings.get("ctx-size")
    ngl_from = f" (currently {ngl})" if ngl else ""
    ctx_from = f" (currently {ctx})" if ctx else ""
    blob = (log_text or "").lower()
    line = _last_error_line(log_text)
    for pat, err, fix in _RULES:
        if re.search(pat, blob):
            return {"error": line or err,
                    "suggestion": fix.format(ngl_from=ngl_from, ctx_from=ctx_from)}
    # Nothing matched a rule, but there's still an error-ish last line worth showing.
    if line and re.search(r"error|fail|abort|terminate|exception", line, re.I):
        return {"error": line,
                "suggestion": "See the Router Log below for full context."}
    return None
