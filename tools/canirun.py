""""Can I run X on Y?" pages for the docs site.

One page per (model, GPU) pair, plus a matrix index. Every number comes from
footprint.predict, the same memory math the app runs on a GGUF before loading
it, applied to real quant headers fetched once from Hugging Face by
tools/canirun_fetch.py into docs/canirun/models.json. Nothing here touches the
network, so the Pages build stays offline and reproducible.

Fit only, never speed: speed depends on far more than memory (and the app
measures it for real once a model is loaded).

Pure stdlib. build_docs.py wraps each page's body in the site template.
"""
import html
import os
import re
import sys
import urllib.parse

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "backend"))
import footprint  # noqa: E402

DATA = os.path.join(_ROOT, "docs", "canirun", "models.json")
MIB = 1024 * 1024
DESKTOP_MIB = 512          # what the desktop / driver already holds on a typical card
ANSWER_CTX = 8192          # the context every verdict is judged at
_LADDER = (4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576)
_SLUG = re.compile(r"^[a-z0-9][a-z0-9.-]*$")
_EXPS = re.compile(footprint._EXPS)
PANEL = "http://127.0.0.1:8090/"

# (slug, name, VRAM GB, article), by group. Nominal VRAM: what the box (or the
# laptop maker's spec sheet) says. Slugs are page URLs: never rename one.
_D, _L, _A = "Desktop GeForce", "Laptop GeForce", "Radeon and Intel Arc"
GPUS = [dict(slug=s, name=n, gb=g, a=a, group=grp) for grp, s, n, g, a in (
    (_D, "rtx-2080-ti", "RTX 2080 Ti", 11, "an"), (_D, "rtx-3050", "RTX 3050 8GB", 8, "an"),
    (_D, "rtx-3060", "RTX 3060", 12, "an"), (_D, "rtx-3060-ti", "RTX 3060 Ti", 8, "an"),
    (_D, "rtx-3070", "RTX 3070", 8, "an"), (_D, "rtx-3080", "RTX 3080", 10, "an"),
    (_D, "rtx-3080-ti", "RTX 3080 Ti", 12, "an"), (_D, "rtx-3090", "RTX 3090", 24, "an"),
    (_D, "rtx-4060", "RTX 4060", 8, "an"), (_D, "rtx-4060-ti", "RTX 4060 Ti 8GB", 8, "an"),
    (_D, "rtx-4060-ti-16gb", "RTX 4060 Ti 16GB", 16, "an"), (_D, "rtx-4070", "RTX 4070", 12, "an"),
    (_D, "rtx-4070-super", "RTX 4070 Super", 12, "an"), (_D, "rtx-4070-ti", "RTX 4070 Ti", 12, "an"),
    (_D, "rtx-4070-ti-super", "RTX 4070 Ti Super", 16, "an"), (_D, "rtx-4080", "RTX 4080", 16, "an"),
    (_D, "rtx-4080-super", "RTX 4080 Super", 16, "an"), (_D, "rtx-4090", "RTX 4090", 24, "an"),
    (_D, "rtx-5060", "RTX 5060", 8, "an"), (_D, "rtx-5060-ti-16gb", "RTX 5060 Ti 16GB", 16, "an"),
    (_D, "rtx-5070", "RTX 5070", 12, "an"), (_D, "rtx-5070-ti", "RTX 5070 Ti", 16, "an"),
    (_D, "rtx-5080", "RTX 5080", 16, "an"), (_D, "rtx-5090", "RTX 5090", 32, "an"),
    (_L, "rtx-3060-laptop", "RTX 3060 Laptop", 6, "an"), (_L, "rtx-4060-laptop", "RTX 4060 Laptop", 8, "an"),
    (_L, "rtx-4070-laptop", "RTX 4070 Laptop", 8, "an"), (_L, "rtx-4080-laptop", "RTX 4080 Laptop", 12, "an"),
    (_L, "rtx-4090-laptop", "RTX 4090 Laptop", 16, "an"), (_L, "rtx-5070-laptop", "RTX 5070 Laptop", 8, "an"),
    (_L, "rtx-5070-ti-laptop", "RTX 5070 Ti Laptop", 12, "an"),
    (_L, "rtx-5080-laptop", "RTX 5080 Laptop", 16, "an"), (_L, "rtx-5090-laptop", "RTX 5090 Laptop", 24, "an"),
    (_A, "rx-6800-xt", "Radeon RX 6800 XT", 16, "a"), (_A, "rx-7600", "Radeon RX 7600", 8, "a"),
    (_A, "rx-7800-xt", "Radeon RX 7800 XT", 16, "a"), (_A, "rx-7900-xt", "Radeon RX 7900 XT", 20, "a"),
    (_A, "rx-7900-xtx", "Radeon RX 7900 XTX", 24, "a"), (_A, "rx-9070", "Radeon RX 9070", 16, "a"),
    (_A, "rx-9070-xt", "Radeon RX 9070 XT", 16, "a"), (_A, "arc-a770", "Intel Arc A770 16GB", 16, "an"),
    (_A, "arc-b580", "Intel Arc B580", 12, "an"))]
GPU_BY_SLUG = {g["slug"]: g for g in GPUS}


# Quant rungs shown per model, best first; alternatives in each rung in order of preference.
LADDER = (("Q8_0",), ("Q6_K",), ("Q5_K_M",), ("Q4_K_M",), ("IQ4_XS",), ("Q3_K_M", "Q3_K_XL"),
          ("Q2_K_XL", "Q2_K"), ("MXFP4_MOE", "MXFP4"))
_SHARD = re.compile(r"-(\d{5})-of-(\d{5})$")
_SKIP = re.compile(r"(^|/)(mmproj|mtp|eagle|imatrix)|BF16|F16|F32", re.I)


def pick_quants(paths):
    """[{quant, files}] from a repo's file list: one quant per LADDER rung, the
    plain name before unsloth's UD- variant, shards of one quant together."""
    groups = {}
    for p in paths:
        if not p.endswith(".gguf") or _SKIP.search(p):
            continue
        stem = _SHARD.sub("", os.path.basename(p)[:-5])
        groups.setdefault(stem, []).append(p)
    out = []
    for rung in LADDER:
        for label in rung:
            hit = None
            for pre in ("-", "-UD-"):
                hit = next((s for s in sorted(groups) if s.endswith(pre + label)
                            and (pre == "-UD-" or not s.endswith("-UD-" + label))), None)
                if hit:
                    out.append({"quant": ("UD-" if pre == "-UD-" else "") + label,
                                "files": sorted(groups[hit])})
                    break
            if hit:
                break
    return out


def compact(layout):
    """A layout small enough to commit: per block, the expert tensors and the
    rest each summed into one entry (names footprint still recognizes), every
    other tensor as it is. Byte totals are unchanged."""
    keep, blocks = [], {}
    for name, typ, nbytes in layout["tensors"]:
        m = re.match(r"blk\.(\d+)\.", name)
        if not m:
            keep.append([name, typ, nbytes])
            continue
        part = "exps" if _EXPS.search(name) else "rest"
        key = (int(m.group(1)), part)
        blocks[key] = blocks.get(key, 0) + nbytes
    for (i, part), nbytes in sorted(blocks.items()):
        keep.append([f"blk.{i}.ffn_up_exps.weight" if part == "exps" else f"blk.{i}.rest", 0, nbytes])
    return {"arch": layout["arch"], "kv": layout["kv"], "tensors": keep}


def _gb(mib):
    return round(mib / 1024, 1)


def _need(model, quant, **settings):
    lay = {"arch": model["arch"], "kv": model["kv"], "tensors": quant["tensors"]}
    return footprint.predict({k.replace("_", "-"): str(v) for k, v in settings.items()}, lay)


def answer(model, quant, vram_gb):
    """How one quant goes on a card of vram_gb: {status: fits | experts | partial
    | no, need_gb, max_ctx, cpu_moe, gpu_layers, ram_gb, ctx}."""
    usable = vram_gb * 1024 - DESKTOP_MIB
    trained = int(model["kv"].get("context_length") or ANSWER_CTX)
    ctx = min(ANSWER_CTX, trained)
    layers = int(model["kv"].get("block_count") or 0)
    p = _need(model, quant, ctx_size=ctx)
    out = {"status": "no", "need_gb": _gb(p["need_mib"]), "ctx": ctx, "max_ctx": 0,
           "cpu_moe": 0, "gpu_layers": 0, "layers": layers, "ram_gb": 0.0}
    if p["need_mib"] <= usable:
        out["status"] = "fits"
        out["ram_gb"] = _gb(p["ram_mib"])
        for c in sorted({c for c in _LADDER if c < trained} | {trained}):
            if _need(model, quant, ctx_size=c)["need_mib"] <= usable:
                out["max_ctx"] = c
        return out
    if model.get("moe"):
        for n in range(1, layers + 1):
            q = _need(model, quant, ctx_size=ctx, n_cpu_moe=n)
            if q["need_mib"] <= usable:
                out.update(status="experts", cpu_moe=n, ram_gb=_gb(q["ram_mib"]),
                           need_gb=_gb(q["need_mib"]))
                return out
    for g in range(layers, 0, -1):
        q = _need(model, quant, ctx_size=ctx, n_gpu_layers=g)
        if q["need_mib"] <= usable:
            out.update(status="partial", gpu_layers=g, ram_gb=_gb(q["ram_mib"]),
                       need_gb=_gb(q["need_mib"]))
            return out
    return out


def flags(a):
    """The llama-server flags for an answer: every layer offloaded (blocks + the
    output layer) unless only some fit, the experts of cpu_moe blocks in RAM, and
    the context it was judged at. "" when nothing fits."""
    if a["status"] == "no":
        return ""
    ngl = a["gpu_layers"] if a["status"] == "partial" else a["layers"] + 1
    moe = f" --n-cpu-moe {a['cpu_moe']}" if a["status"] == "experts" else ""
    ctx = a["max_ctx"] if a["status"] == "fits" else a["ctx"]
    return f"-ngl {ngl}{moe} -c {ctx}"


def command(model, q, a):
    """A copyable llama-server line that downloads the quant and loads it as answered.
    --no-mmproj: -hf would also load a vision projector, which the math leaves out."""
    f = flags(a)
    return f"llama-server -hf {model['repo']}:{q['quant']} {f} --no-mmproj" if f else ""


def valid_models(data):
    return [m for m in data.get("models") or [] if _SLUG.match(str(m.get("slug") or ""))]


def _ctx(n):
    return f"{n // 1024}K" if n >= 1024 else str(n)


def _best(rows, trained):
    """The quant to recommend: the largest that fits with a useful context,
    else the largest that fits at all, else the MoE quant nearest 4-bit with
    experts in RAM, else the smallest file."""
    want = min(16384, trained)
    by_size = sorted(rows, key=lambda r: -r[0]["bytes"])
    for test in (lambda a: a["status"] == "fits" and a["max_ctx"] >= want,
                 lambda a: a["status"] == "fits"):
        for q, a in by_size:
            if test(a):
                return q, a
    moe = [(q, a) for q, a in by_size if a["status"] == "experts"]
    if moe:
        return min(moe, key=lambda r: abs(_bits(r[0]) - 4.4))   # ties go to the 4-bit
    return by_size[-1]


def _bits(q):
    m = re.search(r"(\d)", q["quant"])
    return int(m.group(1)) if m else 4


def _verdict(model, gpu, q, a):
    name = html.escape(q["quant"])
    if a["status"] == "fits":
        return (f"<b>Yes.</b> {name} ({_gb(q['bytes'] / MIB)} GB) fits fully on the GPU, "
                f"with room for {_ctx(a['max_ctx'])} tokens of context.")
    if a["status"] == "experts":
        return (f"<b>Yes, with some experts in system RAM.</b> {name} keeps the expert weights of "
                f"{a['cpu_moe']} of its {a['layers']} layers in RAM (about {a['ram_gb']} GB of system "
                f"RAM) and the rest on the GPU. That's the usual way to run a mixture-of-experts "
                f"model bigger than the card, since only a few experts work on each token.")
    if a["status"] == "partial":
        return (f"<b>Only partly.</b> Even {name} ({_gb(q['bytes'] / MIB)} GB) needs "
                f"{a['layers'] - a['gpu_layers']} of its {a['layers']} layers on the CPU (about "
                f"{a['ram_gb']} GB of system RAM), and a dense model slows down a lot that way. "
                f"A smaller model or a bigger card is the better fix.")
    return f"<b>No.</b> Not even {name} fits on {gpu['a']} {html.escape(gpu['name'])}."


def _row(q, a):
    if a["status"] == "fits":
        how = f"fits, up to {_ctx(a['max_ctx'])} context"
    elif a["status"] == "experts":
        how = f"experts of {a['cpu_moe']}/{a['layers']} layers in RAM ({a['ram_gb']} GB)"
    elif a["status"] == "partial":
        how = f"{a['gpu_layers']}/{a['layers']} layers on GPU, rest on CPU"
    else:
        how = "doesn't fit"
    f = flags(a)
    return (f"<tr><td>{html.escape(q['quant'])}</td><td>{_gb(q['bytes'] / MIB)} GB</td>"
            f"<td>{a['need_gb']} GB</td><td>{how}</td>"
            f"<td>{'<code>' + html.escape(f) + '</code>' if f else ''}</td></tr>")


def _deep(model, q):
    qs = urllib.parse.urlencode({"repo": model["repo"], "file": q["file"]})
    return PANEL + "#discover?" + qs


def pair_page(model, gpu, fetched=""):
    rows = [(q, answer(model, q, gpu["gb"])) for q in model["quants"]]
    trained = int(model["kv"].get("context_length") or ANSWER_CTX)
    bq, ba = _best(rows, trained)
    mname, gname = html.escape(model["name"]), html.escape(gpu["name"])
    title = f"Can I run {model['name']} on {gpu['a']} {gpu['name']} ({gpu['gb']} GB)?"
    repo = html.escape(model["repo"])
    body = [
        f"<h1>Can I run {mname} on {gpu['a']} {gname} ({gpu['gb']} GB)?</h1>",
        f"<p>{_verdict(model, gpu, bq, ba)}</p>",
        *([f"<p>Run it with llama.cpp:</p><pre><code>{html.escape(command(model, bq, ba))}</code></pre>",
           "<p><code>-ngl</code> is how many layers go on the GPU"
           + (", <code>--n-cpu-moe</code> how many of the first blocks keep their expert weights in "
              "system RAM" if ba["status"] == "experts" else "")
           + ", <code>-c</code> the context, and <code>--no-mmproj</code> skips the vision part, which "
           "isn't counted here. Worked out with the same memory math as the table below, "
           "so a long prompt or a busy desktop may need one step more toward the CPU.</p>"]
          if flags(ba) else []),
        f"<h2>Every quant on {gname}</h2>",
        "<table><tr><th>Quant</th><th>Size</th><th>VRAM at "
        f"{_ctx(ba['ctx'])} context</th><th>On this card</th><th>Flags</th></tr>",
        *[_row(q, a) for q, a in sorted(rows, key=lambda r: -r[0]["bytes"])],
        "</table>",
        "<h2>Try it</h2>",
        "<ul>",
        '<li><a href="install.html">Install LlamaForge</a> (one line, no admin). It runs this '
        "same math on the file before loading, then measures the real footprint and tok/s on your machine.</li>",
        f'<li>Already running it? <a href="{html.escape(_deep(model, bq))}">Open {html.escape(bq["quant"])} '
        "in LlamaForge's Discover tab</a> (it shows the quant; the download still takes your click).</li>",
        f'<li>The files: <a href="https://huggingface.co/{repo}">huggingface.co/{repo}</a></li>',
        "</ul>",
        "<h2>How this is worked out</h2>",
        f"<p>This is a prediction, not a benchmark. The quant sizes and the model's attention layout come "
        f"from the real GGUF headers in {repo}{' (read ' + html.escape(fetched) + ')' if fetched else ''}. "
        "VRAM counts every layer on the GPU, an f16 KV cache, about 0.7 GB of CUDA overhead, and about "
        "0.5 GB already taken by the desktop. Speed isn't predicted here: it depends on far more than memory.</p>",
    ]
    others = [g for g in GPUS if g is not gpu]
    body.append("<p>The same model on: " + ", ".join(
        f'<a href="can-i-run-{model["slug"]}-on-{g["slug"]}.html">{html.escape(g["name"])}</a>'
        for g in others) + '. <a href="can-i-run.html">Every model and card</a>.</p>')
    if ba["status"] == "fits":
        desc = (f"Yes: {bq['quant']} of {model['name']} fits on {gpu['a']} {gpu['name']} "
                f"with {_ctx(ba['max_ctx'])} context. Every quant's VRAM, predicted from the real GGUF files.")
    elif ba["status"] == "experts":
        desc = (f"Yes, with {flags(ba)}: {bq['quant']} of {model['name']} on {gpu['a']} {gpu['name']} "
                f"keeps {ba['cpu_moe']} blocks' experts in RAM. Every quant's VRAM and llama.cpp flags.")
    else:
        desc = (f"{model['name']} on {gpu['a']} {gpu['name']} ({gpu['gb']} GB): which quant, how much "
                "goes to system RAM, and every quant's VRAM, predicted from the real GGUF files.")
    return {"slug": f"can-i-run-{model['slug']}-on-{gpu['slug']}", "title": title,
            "desc": desc, "html": "\n".join(body)}


def _cell(model, gpu):
    rows = [(q, answer(model, q, gpu["gb"])) for q in model["quants"]]
    q, a = _best(rows, int(model["kv"].get("context_length") or ANSWER_CTX))
    label = {"fits": html.escape(q["quant"]), "experts": html.escape(q["quant"]) + " + RAM",
             "partial": "CPU offload", "no": "no"}[a["status"]]
    return (f'<td class="fit-{a["status"]}"><a href="can-i-run-{model["slug"]}-on-{gpu["slug"]}.html">'
            f"{label}</a></td>")


def index_page(data):
    models = valid_models(data)
    tables = []
    for group in dict.fromkeys(g["group"] for g in GPUS):
        cards = [g for g in GPUS if g["group"] == group]
        head = "".join(f"<th>{html.escape(g['name'])}<br>{g['gb']} GB</th>" for g in cards)
        rows = "".join(f"<tr><th>{html.escape(m['name'])}</th>{''.join(_cell(m, g) for g in cards)}</tr>"
                       for m in models)
        tables.append(f"<h2>{html.escape(group)}</h2>"
                      f'<div style="overflow:auto"><table><tr><th>Model</th>{head}</tr>{rows}</table></div>')
    body = ("<h1>Can I run it?</h1>"
            "<p>Which quant of a popular model fits on your graphics card, predicted from the real GGUF "
            "files with the memory math LlamaForge runs before every load. Each cell is the biggest quant "
            "that fits fully on the card with at least 16K of context; \"+ RAM\" means a mixture-of-experts "
            "model with some expert weights in system RAM. Click a cell for every quant and the "
            "llama.cpp line that runs it (<code>-ngl</code>, <code>--n-cpu-moe</code>, <code>-c</code>). "
            "Laptop cards use the maker's VRAM spec; check yours, since some laptops ship a smaller one.</p>"
            + "".join(tables) +
            "<p>Predictions, not benchmarks. Once a model is loaded, LlamaForge measures the real "
            'footprint and speed on your machine. <a href="install.html">Install it</a>.</p>')
    return {"slug": "can-i-run", "title": "Can I run it? Models vs graphics cards",
            "desc": "Which quant of Qwen, Gemma and gpt-oss fits on your RTX or Radeon card, with how much "
                    "context and the -ngl / --n-cpu-moe flags: predicted from the real GGUF files.", "html": body}


def build_pages(data):
    models = valid_models(data)
    pages = [index_page(data)]
    for m in models:
        for g in GPUS:
            pages.append(pair_page(m, g, data.get("fetched", "")))
    return pages


def load(path=DATA):
    import json
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)
