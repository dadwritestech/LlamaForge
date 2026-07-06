"""Scan directories (or whole drives) for GGUF models and turn them into
model entries the router understands. Portable: no hardcoded paths.

Rules:
- skip mmproj* (vision projectors; attached to their model instead)
- skip recycle bin and obvious non-model shards handling
- attach an mmproj sibling as the model's `mmproj`
- treat *embed* models as embedding endpoints
- multi-shard sets (foo-00001-of-00005.gguf) collapse to the first shard
- disambiguate duplicate names by parent-folder prefix
"""
import os, re
from collections import defaultdict

def list_drives():
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

def find_ggufs(roots, min_mb=50):
    hits = []
    for root in roots:
        for dirpath, dirnames, files in os.walk(root):
            low = dirpath.lower()
            if "$recycle.bin" in low or "\\.git" in low:
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
def _is_mmproj(p): return _base(p).lower().startswith("mmproj")
def _is_mtp(p):    return _base(p).lower().startswith("mtp-")
def _is_embed(p):  return "embed" in _base(p).lower()
def _shard(p):
    m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", _base(p), re.I)
    return (m.group(1), m.group(2)) if m else None

def build_entries(paths):
    """Return list of {id, model, mmproj?, embeddings?, gib, existing_id?}."""
    mmproj_by_dir = {}
    for p in paths:
        if _is_mmproj(p):
            mmproj_by_dir[os.path.dirname(p)] = p

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
            gib = round(os.path.getsize(p) / 1024**3, 2)
        except OSError:
            gib = 0
        e = {"id": mk_id(p), "model": p.replace("\\", "/"), "gib": gib}
        mm = mmproj_by_dir.get(os.path.dirname(p))
        if mm:
            e["mmproj"] = mm.replace("\\", "/")
        if _is_embed(p):
            e["embeddings"] = True
        entries.append(e)
    return entries

def scan(roots=None, min_mb=50):
    roots = roots or list_drives()
    paths = find_ggufs(roots, min_mb)
    return build_entries(paths)

if __name__ == "__main__":
    import json, sys
    roots = sys.argv[1:] or None
    e = scan(roots)
    print(f"found {len(e)} models")
    print(json.dumps(e[:5], indent=2))
