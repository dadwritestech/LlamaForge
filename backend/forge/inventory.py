"""What Forge has to work with for one model: its drafters, the build, the GPUs."""
import os
import re

from . import space

_QUANT = re.compile(r"[-_.]((ud|iq\d|q\d|bf16|f16|f32|mxfp\d|nvfp\d)([-_].*)?)$", re.I)
_SHARD = re.compile(r"-\d{5}-of-\d{5}$", re.I)
_DRAFT_KIND = {"draft-dflash": "dflash", "draft-mtp": "mtp", "draft-eagle3": "eagle3"}


def family(path):
    """'Qwen3.8-27B-UD-IQ4_XS.gguf' -> 'qwen3.8-27b'."""
    stem = os.path.basename(str(path).replace("\\", "/"))
    stem = re.sub(r"\.gguf$", "", stem, flags=re.I)
    stem = _SHARD.sub("", stem)
    return _QUANT.sub("", stem).lower()


def _norm(p):
    return str(p).replace("\\", "/")


def drafters(model_path, settings, paths, size_of):
    """[{path, kind, mib, configured}] for model_path: the one the user already set
    up first, then same-family DFlash files found among the scanned models."""
    out, seen = [], set()
    own = settings.get("spec-draft-model") or settings.get("model-draft")
    if own:
        kind = _DRAFT_KIND.get(settings.get("spec-type") or "", "dflash" if "dflash" in own.lower() else "draft")
        out.append({"path": _norm(own), "kind": kind, "mib": _mib(own, size_of), "configured": True})
        seen.add(_norm(own).lower())
    fam = family(model_path)
    if "dflash" in fam:
        return out
    for p in paths:
        f = family(p)
        if "dflash" in f and f.startswith(fam) and _norm(p).lower() not in seen:
            out.append({"path": _norm(p), "kind": "dflash", "mib": _mib(p, size_of),
                        "configured": False})
            seen.add(_norm(p).lower())
    return out


def _mib(p, size_of):
    try:
        return int(size_of(p) // 2**20)
    except OSError:
        return 0


def build(gpus, items, settings, drafter_list, mtp=False):
    return {"gpus": gpus, "caps": space.caps_from_items(items), "base": dict(settings),
            "drafters": [d for d in drafter_list if d["kind"] != "mtp"],
            "mtp": mtp or any(d["kind"] == "mtp" for d in drafter_list)}
