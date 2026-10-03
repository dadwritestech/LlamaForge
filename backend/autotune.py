"""Hardware-aware knob recommendations for LlamaForge (pure stdlib).

Turns detected hardware + a GGUF's header facts into a small set of
llama-server knobs, shaped by the user's intent. Memory sizing is llama.cpp's
own --fit; this only sets what fit doesn't decide. `recommend` is a pure function
(no I/O) so it is trivially testable; `refine` (Task 4) optionally benchmarks.
Only the ~8 knobs that materially affect fit/throughput are set; everything
else keeps llama.cpp's own defaults.
"""

import time

INTENTS = ("balanced", "speed", "context", "coding")

# Context floor for the "context" intent.
_CTX_MAX = 150000


# What llama.cpp's --fit (on by default) sizes at load from the real free VRAM:
# context, GPU layers, the multi-GPU split and MoE expert placement. Fit gives
# up entirely if any of these is pinned, so autotune leaves them blank - which
# also clears a pin an older version wrote, since blank means "unset".
_FIT_OWNS = ("ctx-size", "n-gpu-layers", "tensor-split")


def recommend(meta, hw, intent="balanced", size_bytes=None, prediction=None):
    intent = intent if intent in INTENTS else "balanced"
    knobs, why = {}, {}
    cpu = hw.get("cpu") or {}
    threads = cpu.get("threads") or cpu.get("cores")

    knobs["fit"] = "on"
    why["fit"] = ("llama.cpp sizes context, GPU layers and the multi-GPU split to "
                  "your free VRAM at load, and moves MoE experts to CPU when needed.")
    for k in _FIT_OWNS:
        knobs[k] = ""
        why[k] = "Left unset so --fit can size it (pinning it turns fit off)."
    knobs["flash-attn"] = "auto"
    why["flash-attn"] = "llama.cpp enables it wherever the backend supports it, CPU included."

    if threads:
        knobs["threads"] = str(threads)
        why["threads"] = f"Matched to this CPU's {threads} hardware threads."

    _apply_intent(knobs, why, meta.get("context_length"), intent)
    if prediction and prediction.get("regime"):
        tok = prediction.get("tok_s")
        why["fit"] += f" Predicted {prediction['regime']}" + (f" ~{tok} tok/s." if tok is not None else ".")
        return {"knobs": knobs, "rationale": why, "prediction": prediction}
    return {"knobs": knobs, "rationale": why}


def _apply_intent(knobs, why, trained, intent):
    if intent == "context":
        if trained and trained > 0:
            # a floor for fit, not a pin: it offloads layers before going below it
            knobs["fit-ctx"] = str(min(trained, _CTX_MAX))
            why["fit-ctx"] = (f"Max-context: fit keeps at least {knobs['fit-ctx']} tokens "
                              f"(trained {trained}), moving layers to CPU if it must.")
        knobs["cache-type-k"] = knobs["cache-type-v"] = "q8_0"
        why["cache-type-k"] = why["cache-type-v"] = (
            "Max-context: 8-bit KV cache roughly halves memory per token.")
    elif intent == "speed":
        knobs["cache-type-k"] = knobs["cache-type-v"] = "f16"
        why["cache-type-k"] = why["cache-type-v"] = "Max-speed: full-precision KV cache."
        knobs["batch-size"] = "2048"
        knobs["ubatch-size"] = "512"
        why["batch-size"] = "Larger batch for higher prompt throughput."
        why["ubatch-size"] = "Micro-batch tuned for throughput (refine can adjust)."
    elif intent == "coding":
        knobs["temp"] = "0.2"
        knobs["top-p"] = "0.9"
        why["temp"] = "Coding: low temperature for deterministic output."
        why["top-p"] = "Coding: tightened nucleus sampling."


def _candidates(base, intent):
    """Base first, then a few high-impact variants worth benchmarking."""
    out = [dict(base)]
    if intent == "speed":
        for ub in ("1024",):
            c = dict(base); c["ubatch-size"] = ub; out.append(c)
        for bs in ("4096",):
            c = dict(base); c["batch-size"] = bs; out.append(c)
    else:
        for ub in ("1024",):
            c = dict(base); c["ubatch-size"] = ub; out.append(c)
    return out


def refine(base_knobs, intent, load_fn, measure_fn, budget_s=60, clock=time.monotonic):
    start = clock()
    best_knobs, best_tok = dict(base_knobs), -1.0
    cands = []
    for cand in _candidates(base_knobs, intent):
        if clock() - start >= budget_s:
            break
        try:
            load_fn(cand)
            tok = float(measure_fn())
        except Exception:
            continue
        cands.append({"knobs": cand, "tok_s": tok})
        if tok > best_tok:
            best_knobs, best_tok = cand, tok
    return {"knobs": best_knobs,
            "measurements": {"candidates": cands,
                             "chosen_tok_s": best_tok if best_tok >= 0 else 0.0}}
