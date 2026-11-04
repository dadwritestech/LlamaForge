"""Model X-ray: a GGUF taken apart, layer by layer.

Turns gguf.layout() into what the panel draws: one row per block (with the
embedding first and the output head last), each split into the kinds of
tensors it holds (attention, dense FFN, MoE experts, the shared expert, SSM,
norms), the quant types ranked by bytes, and where every byte lands under the
model's settings, by the same rule footprint.predict() uses.

Pure and stdlib only: the header is read by gguf.py, nothing is loaded.
"""
import re

import compat
import footprint

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
