"""Disk reclaim: which registered GGUFs haven't been used in a while, and which
are byte-identical copies of each other (the same quant downloaded into the HF
cache, LM Studio and Downloads). Deletes only on an explicit per-model request,
and only a registered .gguf (with its split shards) that no other model uses.

"Used" is stats.py's last_used: the last poll that saw the model generate.
Pure stdlib."""
import glob
import hashlib
import os
import re

DAY = 86400
HEAD = 4 * 1024 * 1024            # bytes hashed to tell same-size files apart
SAMPLE = 1024 * 1024              # plus this much at 25/50/75/100% of the file
_SHARD = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", re.I)


def files_of(path):
    """The files a model is: every shard of a split GGUF, or the file itself;
    [] when it's gone."""
    if not path or not os.path.isfile(path):
        return []
    m = _SHARD.match(path)
    if not m:
        return [path]
    shards = glob.glob(glob.escape(m.group(1)) + "-[0-9][0-9][0-9][0-9][0-9]-of-%s.gguf" % m.group(3))
    return sorted(p for p in shards if os.path.isfile(p)) or [path]


def _size(files):
    return sum(os.path.getsize(p) for p in files)


def _key(path):
    return os.path.normcase(os.path.realpath(path))


def _head_hash(path):
    """The header plus 1 MB samples from the weights: a fine-tune with the same
    layout and metadata has the same size and header but not the same weights."""
    h = hashlib.sha256()
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        h.update(f.read(HEAD))
        for frac in (0.25, 0.5, 0.75, 1.0):
            f.seek(max(0, int(size * frac) - SAMPLE))
            h.update(f.read(SAMPLE))
    return h.hexdigest()


def _models(sections):
    """{id: {path, files, bytes}} for every [section] whose file exists."""
    out = {}
    for mid, kv in sections.items():
        if mid == "*":
            continue
        files = files_of(kv.get("model"))
        if files:
            out[mid] = {"path": kv["model"], "files": files, "bytes": _size(files)}
    return out


def report(sections, last_used, since, now, days=90):
    """stale: models not used for `days` (or never, since stats began),
    largest first. dups: groups of distinct files with the same size and the
    same header and weight samples; dup_bytes is what deleting all but one copy would free.
    tracked_long_enough: stats are older than `days`, so "never used" means it."""
    models = _models(sections)
    cutoff = now - days * DAY
    stale = [{"id": mid, "path": m["path"], "bytes": m["bytes"],
              "last_used": last_used.get(mid) or 0}
             for mid, m in models.items() if (last_used.get(mid) or 0) < cutoff]
    stale.sort(key=lambda m: -m["bytes"])

    by_file = {}                      # one entry per real file, with every id on it
    for mid, m in models.items():
        e = by_file.setdefault(_key(m["files"][0]), {"path": m["path"], "bytes": m["bytes"], "ids": []})
        e["ids"].append(mid)
    by_size = {}
    for e in by_file.values():
        by_size.setdefault(e["bytes"], []).append(e)
    dups = []
    for size, group in by_size.items():
        if len(group) < 2:
            continue
        by_hash = {}
        for e in group:
            try:
                by_hash.setdefault(_head_hash(e["path"]), []).append(e)
            except OSError:
                continue
        for same in by_hash.values():
            if len(same) > 1:
                dups.append({"bytes": size, "copies": len(same),
                             "ids": sorted(i for e in same for i in e["ids"]),
                             "files": sorted(({"path": e["path"], "ids": sorted(e["ids"])} for e in same),
                                             key=lambda f: f["path"])})
    dups.sort(key=lambda d: -d["bytes"] * (d["copies"] - 1))
    return {"days": days, "stale": stale, "stale_bytes": sum(m["bytes"] for m in stale),
            "dups": dups, "dup_bytes": sum(d["bytes"] * (d["copies"] - 1) for d in dups),
            "since": since, "tracked_long_enough": since <= cutoff}


def check(mid, sections):
    """The files deleting `mid` would remove. ValueError unless it is a
    registered .gguf that exists, isn't a link into another app's cache, and
    no other model points at."""
    if not mid or mid == "*" or mid not in sections:
        raise ValueError(f"{mid!r} is not a registered model")
    path = sections[mid].get("model") or ""
    if not path.lower().endswith(".gguf"):
        raise ValueError("only .gguf model files are deleted here")
    if os.path.islink(path):
        raise ValueError("this file is a link into another app's cache; remove it there")
    files = files_of(path)
    if not files:
        raise ValueError("the file is already gone")
    mine = {_key(p) for p in files}
    others = sorted(o for o, kv in sections.items()
                    if o not in (mid, "*") and kv.get("model")
                    and _key(kv["model"]) in mine)
    if others:
        raise ValueError("also used by " + ", ".join(others) + "; delete or re-point those first")
    return files


def delete_model(mid, sections):
    """Delete model `mid`'s GGUF file(s) (check() first) and return the bytes freed."""
    freed = 0
    for p in check(mid, sections):
        n = os.path.getsize(p)
        os.remove(p)
        freed += n
    return freed
