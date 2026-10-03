"""Turn a raw router (llama.cpp) log tail into a one-line error + a suggested
fix for the model editor. Pure string heuristics, no I/O, so it's unit-testable
and the UI never has to make the user scroll the log panel to learn why a load
failed.

Two rules keep it honest (review 03 H6):
- Only the model's LAST load attempt is read. The router logs "spawning
  server instance with name=<id> on port <P>"; the child's own output comes
  back prefixed "[<P>]", and the router logs "instance name=<id> exited with
  status N". Anything else in the tail - earlier loads, other models - is not
  this failure.
- Patterns are llama.cpp's actual error strings. Broad words like
  "ggml_cuda" or "n_ctx" appear in every healthy CUDA load.
"""
import re

_PIN_KEYS = ("n-gpu-layers", "tensor-split", "ctx-size")

# Ordered most-specific first; the first rule whose pattern hits wins.
_RULES = [
    ("arch", r"unknown model architecture",
     "This llama.cpp build doesn't know this model's architecture.",
     "Update llama.cpp (Build / Update tab). New architectures land in llama.cpp "
     "first, and this model is newer than your build."),
    ("tokenizer", r"unknown pre-tokenizer type",
     "This llama.cpp build doesn't know this model's tokenizer.",
     "Update llama.cpp (Build / Update tab)."),
    ("argument", r"error while handling argument|error: invalid argument|unknown argument",
     "llama.cpp rejected a setting.",
     "Clear {arg} in this model's settings. If it's a newer flag, update llama.cpp."),
    ("flash", r"requires flash_attn to be enabled",
     "This setting needs Flash Attention.",
     "Set flash-attn to on (or auto), or set cache-type-v back to f16."),
    ("oom", r"cudamalloc failed|out of memory|unable to allocate|failed to allocate (?:\S+ )?(?:buffer|compute|context|graph)",
     "Ran out of memory loading the model.",
     None),
    ("mtp", r"context type mtp requested|failed to create mtp context|\bmtp (?:requires|currently only|does not support|missing)",
     "The MTP speculative draft could not start.",
     "Clear spec-type and spec-draft-model for this model: the model or this "
     "llama.cpp build doesn't support its MTP layers."),
    ("mmproj", r"failed to load multimodal model|clip_model_load",
     "The vision projector (mmproj) did not load.",
     "The mmproj file doesn't match this model or build. Clear mmproj, or use the "
     "mmproj from the same repo as the model."),
    ("file", r"failed to open gguf file|failed to read magic|no such file|cannot find the file",
     "The model file could not be opened.",
     "The GGUF path is missing, moved, or an unfinished download. Re-scan drives "
     "from Setup, or fix the model path."),
    ("generic", r"error loading model|failed to load model",
     "The model failed to load.",
     "The line above is llama.cpp's own reason. The full log is at the bottom of Models."),
]

_PORT = re.compile(r"^\[\s*\d+\]\s?")
_STAMP = re.compile(r"^[\d.]+\s+[IWED]\s+")       # llama.cpp log timestamp + level


def _clean(line):
    return _STAMP.sub("", _PORT.sub("", line.strip())).strip()


def last_attempt(text, model):
    """The lines of `model`'s most recent load, or None if the tail holds no
    load of it. LlamaForge's tail puts router.out.log (child output) before
    router.err.log (router lines), so child lines are picked by port prefix
    and the exit line by position after the spawn line."""
    lines = (text or "").splitlines()
    spawn = re.compile(r"spawning server instance with name=" + re.escape(model) + r" on port (\d+)")
    at, port = None, None
    for i, ln in enumerate(lines):
        m = spawn.search(ln)
        if m:
            at, port = i, m.group(1)
    if at is None:
        return None
    child = re.compile(r"^\[\s*" + port + r"\]")
    exited = re.compile(r"instance name=" + re.escape(model) + r" exited with status")
    return [ln for i, ln in enumerate(lines)
            if child.match(ln) or (i > at and exited.search(ln))]


def _oom_fix(settings):
    pins = [f"{k} = {settings[k]}" for k in _PIN_KEYS if settings.get(k)]
    if pins:
        return ("This model pins " + ", ".join(pins) + ", which switches off llama.cpp's "
                "automatic fit to your VRAM. Clear those settings and load again, or pick "
                "a smaller quant.")
    return ("Even llama.cpp's automatic fit could not place it. Pick a smaller quant, "
            "or set a smaller ctx-size.")


def _arg_name(line):
    m = (re.search(r'argument "([^"]+)"', line) or
         re.search(r"(?:invalid|unknown) argument:\s*(\S+)", line))
    return m.group(1) if m else "the most recently changed setting"


def diagnose(log_text, settings=None, model=None):
    """Return {error, suggestion} for a failed load, or None if the log shows
    no recognizable failure. With `model`, only that model's last load attempt
    is considered. `settings` (the model's merged models.ini keys) lets the
    advice name concrete values."""
    settings = settings or {}
    if model:
        lines = last_attempt(log_text, model)
        if lines is None:
            return None
    else:
        lines = (log_text or "").splitlines()
    lines = [ln for ln in lines if ln.strip()]
    for kind, pat, err, fix in _RULES:
        rx = re.compile(pat, re.I)
        hits = [ln for ln in lines if rx.search(ln)]
        if not hits:
            continue
        line = _clean(hits[-1])
        if kind == "oom":
            fix = _oom_fix(settings)
        elif kind == "argument":
            fix = fix.format(arg=_arg_name(line))
        return {"error": line or err, "suggestion": fix}
    for ln in reversed(lines):
        m = re.search(r"exited with status (-?\d+)", ln)
        if m and m.group(1) != "0":
            return {"error": f"llama.cpp exited with status {m.group(1)} while loading.",
                    "suggestion": "The full log is at the bottom of Models."}
    return None
