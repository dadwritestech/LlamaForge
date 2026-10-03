"""Starter models for a first run with nothing downloaded yet (review 01 #2).

A short hand-picked list instead of a Hugging Face search: a newcomer facing
50 repos and 20 quants each picks wrong or gives up. Each entry is one Q4_K_M
(or the model's native MXFP4) file whose size was checked against the Hub;
pick() returns the three largest that fit this machine's VRAM.
"""
import hub

GB = 1000 ** 3

# (repo, file, bytes, title, blurb[, mmproj file, mmproj bytes[, specialist]])
_CATALOG = [
    ("unsloth/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q4_K_M.gguf", 1107409472,
     "Qwen3 1.7B", "Tiny and quick. Runs on almost anything, CPU included."),
    ("unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf", 2497281120,
     "Qwen3 4B Instruct", "The best all-rounder at this size."),
    ("unsloth/gemma-3-4b-it-GGUF", "gemma-3-4b-it-Q4_K_M.gguf", 2489894016,
     "Gemma 3 4B", "Small, and it can read images.",
     "mmproj-F16.gguf", 851251328),
    ("unsloth/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf", 5027784512,
     "Qwen3 8B", "Solid chat and coding help; thinks before it answers."),
    ("unsloth/gemma-3-12b-it-GGUF", "gemma-3-12b-it-Q4_K_M.gguf", 7300778336,
     "Gemma 3 12B", "Strong writer, and it can read images.",
     "mmproj-F16.gguf", 854200448),
    ("unsloth/Qwen3-14B-GGUF", "Qwen3-14B-Q4_K_M.gguf", 9001753984,
     "Qwen3 14B", "A clear step up in reasoning over the 8B."),
    ("ggml-org/gpt-oss-20b-GGUF", "gpt-oss-20b-MXFP4.gguf", 12109566624,
     "gpt-oss 20B", "OpenAI's open-weight model. Fast for its size (MoE)."),
    ("unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF", "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf", 18556686752,
     "Qwen3 30B-A3B Instruct", "Big-model quality at small-model speed (MoE)."),
    ("unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf", 18556689568,
     "Qwen3 Coder 30B-A3B", "Built for coding agents: Claude Code, Codex, pi.", None, 0, True),
]


def pick_all():
    out = []
    for row in _CATALOG:
        repo, path, weights, title, blurb = row[:5]
        s = {"repo": repo, "path": path, "weights": weights, "size": weights,
             "title": title, "blurb": blurb, "shards": 1}
        if len(row) > 5 and row[5]:
            s.update(mmproj=row[5], mmproj_size=row[6], size=weights + row[6])
        s["specialist"] = len(row) > 7 and row[7]
        out.append(s)
    return out


def pick(vram_mib, n=3):
    """The n largest starters that fit fully in VRAM, biggest first. Without a
    GPU (or one too small for any of them): the n smallest, marked "cpu"."""
    every = pick_all()
    fits = [s for s in every if hub._fit(s["size"], vram_mib) == "fits"]
    if fits:
        # same-class models (to the GB): a general one leads a specialist
        chosen = sorted(fits, key=lambda s: (round(s["size"] / GB), not s["specialist"]),
                        reverse=True)[:n]
        for s in chosen:
            s["fit"] = "fits"
    else:
        chosen = sorted(every, key=lambda s: s["size"])[:n]
        for s in chosen:
            s["fit"] = "cpu"
    for i, s in enumerate(chosen):
        s["recommended"] = i == 0
    return chosen
