"""Scan directories (or whole drives) for GGUF models and turn them into
model entries the router understands. Portable: no hardcoded paths.

Rules:
- skip mmproj* (vision projectors; attached to their model instead)
- skip recycle bin and obvious non-model shards handling
- attach a same-folder mmproj only when it provably belongs to the model
- treat *embed* models as embedding endpoints
- multi-shard sets (foo-00001-of-00005.gguf) collapse to the first shard
- disambiguate duplicate names by parent-folder prefix
"""
import os, re
from collections import defaultdict

import osplat
from gguf import total_size

def _windows_drives():
    import string, ctypes
    drives = []
    bitmask = ctypes.windll.kernel32.GetLogicalDrives()
    for i, letter in enumerate(string.ascii_uppercase):
        if bitmask & (1 << i):
            root = f"{letter}:\\"
            # only fixed drives
            if ctypes.windll.kernel32.GetDriveTypeW(root) == 3:
                drives.append(root)
    return drives

def posix_roots(home, extra_candidates, isdir=os.path.isdir):
    """Home dir plus whichever common mount points exist. Pure for tests."""
    roots = [home]
    for c in extra_candidates:
        if isdir(c):
            roots.append(c)
    return roots

def list_drives():
    if osplat.IS_WIN:
        return _windows_drives()
    extra = ["/Volumes"] if osplat.IS_MAC else ["/mnt", "/media", "/srv", "/data"]
    return posix_roots(os.path.expanduser("~"), extra)

def find_ggufs(roots, min_mb=50):
    hits = []
    for root in roots:
        for dirpath, dirnames, files in os.walk(root):
            low = dirpath.lower().replace("\\", "/")
            if "$recycle.bin" in low or "/.git" in low or "/.trash" in low or "/proc" == low[:5]:
                dirnames[:] = []
                continue
            for fn in files:
                if fn.lower().endswith(".gguf"):
                    full = os.path.join(dirpath, fn)
                    try:
                        if os.path.getsize(full) >= min_mb * 1024 * 1024:
                            hits.append(full)
                    except OSError:
                        pass
    return hits

def _slug(s):
    s = re.sub(r"\.gguf$", "", s, flags=re.I).lower().replace("_", "-").replace(" ", "-")
    s = re.sub(r"[^a-z0-9.\-]", "", s)
    return re.sub(r"-+", "-", s).strip("-")

def _base(p): return os.path.basename(p)
def _is_mmproj(p):
    """mmproj-*.gguf, or a <model>-mmproj-*.gguf token match (not a substring)."""
    return re.search(r"(?:^|[-_.])mmproj(?:[-_.]|$)", _base(p), re.I) is not None
def _is_mtp(p):    return _base(p).lower().startswith("mtp-")
def _is_embed(p):  return "embed" in _base(p).lower()
def _shard(p):
    m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", _base(p), re.I)
    return (m.group(1), m.group(2)) if m else None

def _model_stem(p):
    """Normalized logical basename, with a shard suffix removed."""
    return re.sub(r"-\d{5}-of-\d{5}$", "", _slug(_base(p)), flags=re.I)

def mtp_pairs(mains, sidecars):
    """Return {main_path: mtp_path} for unambiguous same-directory pairs.

    An exact normalized ``mtp-<main basename>`` match wins. A generic sidecar
    is safe only when its directory contains exactly one main and one sidecar.
    """
    mains_by_dir, sidecars_by_dir = defaultdict(list), defaultdict(list)
    for path in mains:
        mains_by_dir[os.path.dirname(path)].append(path)
    for path in sidecars:
        sidecars_by_dir[os.path.dirname(path)].append(path)

    pairs = {}
    for directory, dir_mains in mains_by_dir.items():
        dir_sidecars = sidecars_by_dir.get(directory, [])
        candidates = defaultdict(list)
        for sidecar in dir_sidecars:
            tail = _model_stem(sidecar)
            if tail.startswith("mtp-"):
                tail = tail[4:]
            matches = [main for main in dir_mains if _model_stem(main) == tail]
            if len(matches) == 1:
                candidates[matches[0]].append(sidecar)
        for main, matches in candidates.items():
            if len(matches) == 1:
                pairs[main] = matches[0]
        if not pairs.keys() & set(dir_mains) and len(dir_mains) == 1 and len(dir_sidecars) == 1:
            pairs[dir_mains[0]] = dir_sidecars[0]
    return pairs

_PRECISION_RANK = {"f16": 0, "bf16": 1, "f32": 2}
_PRECISION_TAIL = re.compile(r"[-.](f16|bf16|f32|q\d+(?:-\d+|-k(?:-[sml])?)?)$")

def _projector_rank(path):
    """Deterministic preference among sibling projectors: F16, BF16, F32, then name."""
    m = _PRECISION_TAIL.search(_model_stem(path))
    return (_PRECISION_RANK.get(m.group(1) if m else "", 3), _base(path).lower())

def _projector_target(path):
    """The model stem a projector names: mmproj-Swift-27B-F16 and
    Swift-27B-mmproj-F16 both -> swift-27b.
    Empty for generic names like mmproj-F16."""
    tail = re.sub(r"(?:^|[-.])mmproj(?=[-.]|$)", "", _model_stem(path)).strip("-.")
    return _PRECISION_TAIL.sub("", tail) if tail not in _PRECISION_RANK else ""

def mmproj_pairs(mains, projectors):
    """Return {main_path: mmproj_path} for same-directory projectors that fit.

    A projector fits a model when its clip projection_dim equals the model's
    embedding_length (llama.cpp rejects the pair otherwise). Among fitting
    projectors, one whose name targets the model wins; a generic one (mmproj-F16)
    is used only when every model in the folder is the same base model, so a
    shared folder never hands one model's projector to an unrelated model.
    """
    import gguf
    mains_by_dir, projs_by_dir = defaultdict(list), defaultdict(list)
    for path in mains:
        mains_by_dir[os.path.dirname(path)].append(path)
    for path in projectors:
        projs_by_dir[os.path.dirname(path)].append(path)

    pairs = {}
    for directory, dir_projs in projs_by_dir.items():
        dir_mains = mains_by_dir.get(directory)
        if not dir_mains:
            continue
        info = {pj: i for pj in dir_projs if (i := gguf.projector_info(pj)) is not None}
        if not info:
            continue
        meta = {m: gguf.metadata(m) or {} for m in dir_mains}
        one_base = len({(md.get("architecture"), md.get("embedding_length"))
                        for md in meta.values()}) == 1
        for main in dir_mains:
            emb = meta[main].get("embedding_length")
            fits = [pj for pj, i in info.items()
                    if emb is None or i.get("projection_dim") in (None, emb)]
            named = [pj for pj in fits if _projector_target(pj)
                     and _model_stem(main).startswith(_projector_target(pj))]
            pick = named or (fits if one_base else [])
            if pick:
                pairs[main] = min(pick, key=_projector_rank)
    return pairs

def build_entries(paths):
    """Return list of {id, model, mmproj?, embeddings?, gib, existing_id?}."""
    projectors, mtp_sidecars = [], []
    for p in paths:
        if _is_mmproj(p):
            projectors.append(p)
        elif _is_mtp(p):
            mtp_sidecars.append(p)

    mains = []
    seen_shard_sets = set()
    for p in paths:
        if _is_mmproj(p) or _is_mtp(p):
            continue
        sh = _shard(p)
        if sh:
            key = (os.path.dirname(p), re.sub(r"-\d{5}-of-\d{5}\.gguf$", "", _base(p), flags=re.I))
            if key in seen_shard_sets or sh[0] != "00001":
                continue  # only the first shard represents the set
            seen_shard_sets.add(key)
        mains.append(p)

    mtp_by_main = mtp_pairs(mains, mtp_sidecars)
    mmproj_by_main = mmproj_pairs(mains, projectors)

    stem_counts = defaultdict(int)
    for p in mains:
        stem_counts[_slug(_base(p))] += 1

    def mk_id(p):
        b = _slug(_base(p))
        if stem_counts[b] > 1:
            return _slug(_base(os.path.dirname(p))) + "--" + b
        return b

    entries = []
    for p in sorted(mains):
        try:
            gib = round(total_size(p) / 1024**3, 2)
        except OSError:
            gib = 0
        e = {"id": mk_id(p), "model": p.replace("\\", "/"), "gib": gib}
        mm = mmproj_by_main.get(p)
        if mm:
            e["mmproj"] = mm.replace("\\", "/")
        # Attach an mtp-* sibling as a speculative draft model. Attaching alone
        # is inert; only enable spec-type=draft-mtp when the sidecar actually
        # declares NextN layers, the signal llama.cpp itself gates on.
        mt = mtp_by_main.get(p)
        if mt:
            from gguf import has_nextn
            e["draft_model"] = mt.replace("\\", "/")
            if has_nextn(mt):
                e["draft_mtp"] = True
        if _is_embed(p):
            e["embeddings"] = True
        entries.append(e)
    return entries

def merge(found, extra):
    """found + extra (Ollama's models), prefixing an extra id that clashes."""
    ids = {e["id"] for e in found}
    out = list(found)
    for e in extra:
        if e["id"] in ids:
            e = dict(e, id="ollama--" + e["id"])
        ids.add(e["id"])
        out.append(e)
    return out

def scan(roots=None, min_mb=50):
    import ollama
    roots = roots or list_drives()
    paths = find_ggufs(roots, min_mb)
    return merge(build_entries(paths), ollama.entries())

if __name__ == "__main__":
    import json, sys
    roots = sys.argv[1:] or None
    e = scan(roots)
    print(f"found {len(e)} models")
    print(json.dumps(e[:5], indent=2))
