"""Model X-ray: a GGUF taken apart, layer by layer.

Turns gguf.layout() into what the panel draws: one row per block (with the
embedding first and the output head last), each split into the kinds of
tensors it holds (attention, dense FFN, MoE experts, the shared expert, SSM,
norms), the quant types ranked by bytes, and where every byte lands under the
model's settings, by the same rule footprint.predict() uses.

Pure and stdlib only: the header is read by gguf.py, nothing is loaded.
"""
import io
import re
import threading
import time
import urllib.parse
import urllib.request

import compat
import footprint
import gguf
import hub

# A quant that isn't on this disk is read like the template check reads one:
# the header only, by HTTP Range, starting small and widening until it parses.
FIRST = 4 * 1024 * 1024
CAP = 256 * 1024 * 1024
_REMOTE = {}            # (repo, path) -> layout; headers never change for a commit
_LISTS = {}             # repo -> (time, files)
_LIST_TTL = 600
_LOCK = threading.Lock()

_BLK = re.compile(r"blk\.(\d+)\.")
# first match wins: expert tensors are also "ffn", an MTP head holds its own norms.
# per-layer = Gemma 3n/4 E-models' per-layer embedding input (gate + projection)
_KINDS = (("mtp", re.compile(r"^nextn\.")),
          ("per-layer", re.compile(r"^(inp_gate|proj)\.")),
          ("shared", re.compile(r"_shexp|_chexps")),
          ("experts", re.compile(r"_exps|ffn_gate_inp")),
          ("norm", re.compile(r"norm")),
          ("attention", re.compile(r"attn|\.wq|\.wk|\.wv|\.wo")),
          ("ssm", re.compile(r"ssm|mixer|time_mix|conv")),
          ("ffn", re.compile(r"ffn|mlp")))


def kind(name):
    rest = _BLK.sub("", name, count=1)
    return next((k for k, rx in _KINDS if rx.search(rest)), "other")


def view(layout, settings):
    """{arch, n_layer, ngl, confident, total, gpu, ram, tied_copy (bytes of
    embedding copied to the GPU for a tied output, on top of gpu), types: [{type, bytes,
    share}], rows: [{label, gpu, ram, type, parts: {kind: bytes},
    ram_parts: {kind: bytes left in system RAM}}]}, or None
    when the layout is missing or holds no tensors."""
    if not layout or not layout.get("tensors"):
        return None
    on_gpu, info = footprint.placer(settings, layout)
    rows, by_type = {}, {}
    order = []

    def row(key, label):
        if key not in rows:
            rows[key] = {"label": label, "gpu": 0, "ram": 0, "parts": {}, "ram_parts": {},
                         "_types": {}}
            order.append(key)
        return rows[key]

    for name, typ, nbytes in layout["tensors"]:
        m = _BLK.match(name)
        if m:
            r = row(("b", int(m.group(1))), f"blk {m.group(1)}")
            part = kind(name)
        elif name.startswith(("token_embd", "per_layer_token_embd")):
            r, part = row(("a", 0), "embedding"), "embedding"
        else:
            r, part = row(("z", 0), "output"), "output"
        if on_gpu(name):
            r["gpu"] += nbytes
        else:
            r["ram"] += nbytes
            r["ram_parts"][part] = r["ram_parts"].get(part, 0) + nbytes
        r["parts"][part] = r["parts"].get(part, 0) + nbytes
        tn = compat.type_name(typ)
        r["_types"][tn] = r["_types"].get(tn, 0) + nbytes
        by_type[tn] = by_type.get(tn, 0) + nbytes

    out_rows = []
    for key in sorted(order):
        r = rows[key]
        types = r.pop("_types")
        r["type"] = max(types, key=types.get) if types else ""
        out_rows.append(r)
    total = sum(by_type.values())
    names = {n for n, _, _ in layout["tensors"]}
    # a tied output reads the embedding: llama.cpp keeps a copy of it on the GPU
    tied = (sum(b for n, _, b in layout["tensors"] if n.startswith("token_embd."))
            if info["output_on_gpu"] and "output.weight" not in names else 0)
    gpu = sum(r["gpu"] for r in out_rows)
    return {"arch": layout.get("arch") or "", "n_layer": info["n_layer"], "ngl": info["n_gpu"],
            "confident": info["confident"], "total": total, "tied_copy": tied, "gpu": gpu, "ram": total - gpu,
            "types": [{"type": t, "bytes": b, "share": round(b / total, 4) if total else 0}
                      for t, b in sorted(by_type.items(), key=lambda kv: -kv[1])],
            "rows": out_rows}


def budget(vram):
    """Bytes of VRAM the weights may take: the rest (10%, at least 1 GB) is
    left for the KV cache and compute buffers."""
    return max(0, vram - max(1024 ** 3, vram // 10))


def with_ngl(settings, ngl):
    """settings with n-gpu-layers set to ngl, whatever spelling it had."""
    s = {k: v for k, v in settings.items()
         if footprint._ALIASES.get(k.strip().lower(), k.strip().lower()) != "n-gpu-layers"}
    s["n-gpu-layers"] = str(ngl)
    return s


def fit_ngl(layout, settings, room):
    """The most GPU layers whose weights (tied copy included) fit in room bytes."""
    def used(n):
        v = view(layout, with_ngl(settings, n))
        return v["gpu"] + v["tied_copy"] if v else 0
    top = footprint.placer(settings, layout)[1]["n_layer"] + 1
    lo, hi = 0, top                     # gpu use only grows with ngl: binary search
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if used(mid) <= room:
            lo = mid
        else:
            hi = mid - 1
    return lo


def has_experts(layout):
    """True when the model holds MoE expert tensors --n-cpu-moe can move."""
    rx = re.compile(footprint._EXPS)
    return any(rx.search(n) for n, _, _ in (layout or {}).get("tensors") or ())


def with_ncmoe(settings, k):
    """settings with the experts of the first k blocks on the CPU (no cpu-moe
    of any spelling left over); k = 0 drops the flag."""
    s = {key: v for key, v in settings.items()
         if footprint._ALIASES.get(key.strip().lower(), key.strip().lower()) not in ("n-cpu-moe", "cpu-moe")}
    if k:
        s["n-cpu-moe"] = str(k)
    return s


def fit_ncmoe(layout, room):
    """With every layer on the GPU, the fewest blocks whose experts go to the CPU
    so the weights fit in room bytes: None for a dense model or when even all
    experts on the CPU leave too much."""
    if not has_experts(layout):
        return None
    top = footprint.placer({}, layout)[1]["n_layer"]

    def used(k):
        v = view(layout, with_ncmoe(with_ngl({}, top + 1), k))
        return v["gpu"] + v["tied_copy"] if v else 0
    if used(top) > room:
        return None
    lo, hi = 0, top                     # gpu use only shrinks as k grows
    while lo < hi:
        mid = (lo + hi) // 2
        if used(mid) <= room:
            hi = mid
        else:
            lo = mid + 1
    return lo


def placement(layout, room):
    """{ngl, ncmoe}: for a MoE, every layer on the GPU with the experts of the
    first blocks in RAM (--n-cpu-moe), when that fits; else whole layers off
    (-ngl). The experts fit stops within one block's experts of the budget, so it
    is never a whole block behind the -ngl fit, and the CPU then runs only the few
    active experts of those blocks, not their attention and KV cache too."""
    k = fit_ncmoe(layout, room)
    if k is None:
        return {"ngl": fit_ngl(layout, {}, room), "ncmoe": 0}
    return {"ngl": footprint.placer({}, layout)[1]["n_layer"] + 1, "ncmoe": k}


def flags(ngl, ncmoe):
    """The llama-server flags for a placement."""
    return f"-ngl {ngl}" + (f" --n-cpu-moe {ncmoe}" if ncmoe else "")


def _get(url, headers):
    """(body, response headers) for a GET: patched in tests."""
    req = urllib.request.Request(url, headers=dict(hub.UA, **headers))
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read(CAP), dict(r.headers)


def _header(url):
    """(header bytes, file size) for one GGUF, or None: the Range widens x4
    until gguf can parse the header, and stops at the whole file or CAP."""
    n = FIRST
    while True:
        body, hdrs = _get(url, {"Range": f"bytes=0-{n - 1}"})
        cr = {k.lower(): v for k, v in hdrs.items()}.get("content-range", "")
        total = int(cr.rsplit("/", 1)[1]) if "/" in cr and cr.rsplit("/", 1)[1].isdigit() else len(body)
        if gguf.layout_streams([(io.BytesIO(body), total)]) is not None:
            return body, total
        if len(body) >= total or n >= CAP:
            return None
        n *= 4


def remote_layout(repo, path, shards=1):
    """gguf.layout() of a Hugging Face file (every shard of a split set) from
    its headers alone, or None when it can't be read. Cached per file."""
    key = (repo, path)
    with _LOCK:
        if key in _REMOTE:
            return _REMOTE[key]
    parts = []
    try:
        for p in hub.shard_paths(path, shards):
            got = _header(f"{hub.HF}/{repo}/resolve/main/{urllib.parse.quote(p)}")
            if got is None:
                return None
            parts.append((io.BytesIO(got[0]), got[1]))
    except (OSError, ValueError):
        return None
    lay = gguf.layout_streams(parts)
    if lay is not None:
        with _LOCK:
            _REMOTE[key] = lay
    return lay


def quants(repo):
    """The repo's model files (hub.files, shard sets collapsed), kept 10 minutes."""
    with _LOCK:
        hit = _LISTS.get(repo)
        if hit and time.time() - hit[0] < _LIST_TTL:
            return hit[1]
    # an importance matrix ships as .gguf too, but it's no model
    got = [{"path": f["path"], "size": f["size"], "shards": f.get("shards", 1)}
           for f in hub.files(repo)["files"]
           if not f["path"].rsplit("/", 1)[-1].lower().startswith("imatrix")]
    with _LOCK:
        _LISTS[repo] = (time.time(), got)
    return got
