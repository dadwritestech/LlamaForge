"""Honest speed measurement against one loaded candidate.

Numbers come from llama-server's own `timings` (decode excludes prefill and
vice versa). Prompt cache off, a fixed seed, and no sampler overrides: the
server's defaults are the user's models.ini values, so drafter acceptance is
what they will really see. Decode tk/s is token-weighted (total tokens / total
decode time), with the median and spread alongside so noise is visible.
"""
import statistics

from . import prompts

WARMUP = "Say hi in one short sentence."
SEED = 1234
MIN_TOKENS = 32                 # shorter answers time mostly overhead


def measure(post, prompt, n_predict, template_kwargs=None):
    """One request. post(path, body) -> response JSON."""
    body = {"messages": [{"role": "user", "content": prompt}],
            "max_tokens": n_predict, "cache_prompt": False, "seed": SEED}
    if template_kwargs:
        body["chat_template_kwargs"] = dict(template_kwargs)
    t = (post("/v1/chat/completions", body) or {}).get("timings") or {}
    return {"decode": float(t.get("predicted_per_second") or 0),
            "prefill": float(t.get("prompt_per_second") or 0),
            "n": int(t.get("predicted_n") or 0), "prompt_n": int(t.get("prompt_n") or 0),
            "draft": int(t.get("draft_n") or 0), "accepted": int(t.get("draft_n_accepted") or 0)}


def summarize(samples, min_tokens=MIN_TOKENS):
    """{decode, median, spread, acceptance, prefill, n} or None without usable samples."""
    ok = [s for s in samples if s["decode"] > 0 and s["n"] >= min_tokens]
    if not ok:
        return None
    toks = sum(s["n"] for s in ok)
    secs = sum(s["n"] / s["decode"] for s in ok)
    rates = [s["decode"] for s in ok]
    med = statistics.median(rates)
    drafted = sum(s["draft"] for s in ok)
    return {"decode": round(toks / secs, 2), "median": round(med, 2),
            "spread": round((max(rates) - min(rates)) / med, 3) if med else 0.0,
            "acceptance": round(sum(s["accepted"] for s in ok) / drafted, 3) if drafted else None,
            "prefill": round(statistics.median(s["prefill"] for s in ok), 1), "n": len(ok)}


def run_round(post, counts, round_id, n_predict=256, template_kwargs=None):
    """{<category>: decode tk/s, ..., prefill, detail: {cat: summary}, error}.
    counts = {"code": 3, ...}. One warm-up request first, never counted."""
    out = {"detail": {}, "error": None, "prefill": None}
    try:
        measure(post, WARMUP, 16, template_kwargs)
        for cat, n in counts.items():
            samples = [measure(post, p, n_predict, template_kwargs)
                       for p in prompts.select(cat, n, round_id)]
            s = summarize(samples)
            out["detail"][cat] = s
            out[cat] = s["decode"] if s else None
    except (OSError, ValueError, KeyError) as e:
        out["error"] = f"benchmark request failed: {e}"
        return out
    summ = out["detail"].get("summary")
    if summ:
        out["prefill"] = summ["prefill"]       # only long prompts say anything about prefill
    return out
