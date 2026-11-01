#!/usr/bin/env python3
"""Refresh docs/canirun/models.json: the real GGUF headers behind the "Can I
run X on Y?" pages (tools/canirun.py).

For each model below it lists the Hugging Face repo, picks one quant per rung
of canirun.LADDER, reads only the header of every file over an HTTP Range
(a few MB, never the weights) and keeps a compacted tensor layout. Run it by
hand when the model list changes; the docs build itself stays offline.

    python tools/canirun_fetch.py
"""
import datetime
import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import canirun

sys.path.insert(0, os.path.join(canirun._ROOT, "backend"))
import gguf  # noqa: E402

HF = "https://huggingface.co"
UA = {"User-Agent": "LlamaForge-docs/1.0 (+https://github.com/dadwritestech/LlamaForge)"}

# (slug, display name, repo). Popular, unmodified GGUF repos only.
MODELS = [
    ("qwen3.5-4b", "Qwen3.5 4B", "unsloth/Qwen3.5-4B-GGUF"),
    ("qwen3.5-9b", "Qwen3.5 9B", "unsloth/Qwen3.5-9B-GGUF"),
    ("gemma-4-12b", "Gemma 4 12B", "unsloth/gemma-4-12b-it-GGUF"),
    ("gpt-oss-20b", "gpt-oss 20B", "ggml-org/gpt-oss-20b-GGUF"),
    ("gemma-4-26b-a4b", "Gemma 4 26B-A4B", "unsloth/gemma-4-26B-A4B-it-GGUF"),
    ("qwen3.8-27b", "Qwen3.8 27B", "unsloth/Qwen3.8-27B-GGUF"),
    ("qwen3.6-35b-a3b", "Qwen3.6 35B-A3B", "unsloth/Qwen3.6-35B-A3B-GGUF"),
    ("gpt-oss-120b", "gpt-oss 120B", "ggml-org/gpt-oss-120b-GGUF"),
    ("qwen3.8-flash-next", "Qwen3.8 Flash Next", "unsloth/Qwen3.8-Flash-Next-GGUF"),
]


def _get(url, headers=None, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=dict(UA, **(headers or {})))
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read(), r.headers
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


def repo_files(repo):
    body, _ = _get(f"{HF}/api/models/{repo}/tree/main?recursive=1")
    return [f["path"] for f in json.loads(body) if f.get("type") == "file"]


def header(repo, path):
    """(the file's first bytes, its full size), widening the range until the
    whole GGUF header fits."""
    url = f"{HF}/{repo}/resolve/main/{urllib.parse.quote(path)}"
    n = 4 * 1024 * 1024
    while True:
        body, h = _get(url, {"Range": f"bytes=0-{n - 1}"})
        total = int((h.get("Content-Range") or "").rsplit("/", 1)[-1] or len(body))
        if gguf.layout_streams([(io.BytesIO(body), total)]) is not None or len(body) < n:
            return body, total
        n *= 4
        if n > 256 * 1024 * 1024:
            raise RuntimeError(f"header of {path} is larger than 256 MB")


def fetch_model(slug, name, repo):
    quants, base = [], None
    for q in canirun.pick_quants(repo_files(repo)):
        parts = [header(repo, p) for p in q["files"]]
        lay = gguf.layout_streams([(io.BytesIO(b), total) for b, total in parts])
        if lay is None:
            print(f"  skip {q['quant']}: unreadable header", file=sys.stderr)
            continue
        base = base or lay
        quants.append({"quant": q["quant"], "file": q["files"][0],
                       "bytes": sum(total for _, total in parts),
                       "tensors": canirun.compact(lay)["tensors"]})
        print(f"  {q['quant']}: {quants[-1]['bytes'] / 2**30:.1f} GB", file=sys.stderr)
    if not base:
        return None
    return {"slug": slug, "name": name, "repo": repo, "arch": base["arch"], "kv": base["kv"],
            "moe": bool(base["kv"].get("expert_count")), "quants": quants}


def main():
    models = []
    for slug, name, repo in MODELS:
        print(repo, file=sys.stderr)
        m = fetch_model(slug, name, repo)
        if m:
            models.append(m)
    os.makedirs(os.path.dirname(canirun.DATA), exist_ok=True)
    with open(canirun.DATA, "w", encoding="utf-8") as f:
        json.dump({"fetched": datetime.date.today().isoformat(), "models": models}, f,
                  separators=(",", ":"))
        f.write("\n")
    print(f"wrote {len(models)} models to {canirun.DATA}", file=sys.stderr)


if __name__ == "__main__":
    main()
