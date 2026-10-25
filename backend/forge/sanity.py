"""Is a drafter config still the same model? Greedy output with and without it.

At temperature 0 speculative decoding should be lossless: the target model
verifies every drafted token. Batched verification does drift in the last bits
of float math, so late divergence is normal - but a split in the first few dozen
tokens on several prompts means a broken drafter or build, not a fast one.

The plain and drafted runs happen when each candidate is benched (it's already
loaded); this module only samples and compares.
"""
EARLY = 32          # divergence before this token counts against the config
MAX_EARLY = 1       # tolerated early splits
N_TOKENS = 64
PROMPTS = (
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "List the planets of the solar system in order, one per line.",
    "def quicksort(xs):\n",
    "The three most important ideas in thermodynamics are",
    "Translate to French: The library opens at nine and closes at six.",
    "SELECT name, COUNT(*) FROM orders",
)


def sample(post, prompt, n=N_TOKENS):
    """Greedy token ids for a raw prompt (no chat template: the same text both ways)."""
    r = post("/completion", {"prompt": prompt, "n_predict": n, "temperature": 0,
                             "top_k": 1, "seed": 0, "cache_prompt": False,
                             "return_tokens": True}) or {}
    return list(r.get("tokens") or ())


def first_divergence(a, b):
    """Index of the first differing token, or None when one is a prefix of the other."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None


def check(plain, spec, early=EARLY, max_early=MAX_EARLY):
    """plain/spec = {prompt: tokens}. {ok, early, compared, splits, reason}."""
    splits = {}
    for p, a in plain.items():
        if p in spec:
            splits[p] = first_divergence(a, spec[p])
    n_early = sum(1 for d in splits.values() if d is not None and d < early)
    ok = n_early <= max_early
    return {"ok": ok, "early": n_early, "compared": len(splits),
            "splits": list(splits.values()),
            "reason": "" if ok else (f"greedy output changed within {early} tokens on "
                                     f"{n_early} of {len(splits)} prompts - the drafter "
                                     f"is changing the answers")}
