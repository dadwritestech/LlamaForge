"""Re-upload watch: has the Hugging Face repo a model came from replaced its
file since you downloaded it? Quantizers often re-upload in a release's first
week (template fixes, broken quants), and nothing tells you.

One `paths-info` call per model returns the repo's file sha256, size and last
commit. Your side:
  HF cache   - the blob's file name *is* its sha256: an exact answer.
  downloads  - no hash on record (hashing 20 GB to check is no option), so a
               different size means re-uploaded, and a commit newer than your
               copy with the same size is "changed" (often only metadata).
Results are cached in reupload.json and refreshed at most daily. Pure stdlib."""
import concurrent.futures
import json
import os
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime

import atomicio
import tmpldoctor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "reupload.json")
MAX_AGE = 24 * 3600
_SHA = re.compile(r"^[0-9a-f]{64}$")


def local(path):
    """{size, mtime, sha256|None} for a file on disk."""
    st = os.stat(path)
    real = os.path.realpath(path)
    name, parent = os.path.basename(real), os.path.basename(os.path.dirname(real))
    sha = name if parent == "blobs" and _SHA.match(name) else None
    return {"size": st.st_size, "mtime": st.st_mtime, "sha256": sha}


def _paths_info(repo, paths):
    data = urllib.parse.urlencode([("paths", p) for p in paths] + [("expand", "true")]).encode()
    req = urllib.request.Request(f"{tmpldoctor.HF}/api/models/{repo}/paths-info/main", data=data,
                                 headers=tmpldoctor.UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read(4 * 1024 * 1024))


def _tree(repo):
    return [x.get("path", "") for x in tmpldoctor._json(f"{tmpldoctor.HF}/api/models/{repo}/tree/main?recursive=true")
            if isinstance(x, dict) and x.get("type") == "file"]


def remote(repo, rel):
    """The repo's current copy of `rel`: {path, size, sha256, commit, title,
    date}, or None when the repo no longer has it. Downloads are saved flat, so
    a miss is looked up by file name in the repo tree."""
    info = _paths_info(repo, [rel])
    if not info:
        name = rel.rsplit("/", 1)[-1]
        hits = [p for p in _tree(repo) if p.rsplit("/", 1)[-1] == name]
        if len(hits) != 1:
            return None
        info = _paths_info(repo, hits)
        if not info:
            return None
    x = info[0]
    lfs, last = x.get("lfs") or {}, x.get("lastCommit") or {}
    return {"path": x.get("path", rel), "size": lfs.get("size", x.get("size")),
            "sha256": lfs.get("oid"), "commit": last.get("id", ""),
            "title": last.get("title", ""), "date": last.get("date", "")}


def _ts(iso):
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except (AttributeError, ValueError):
        return 0


def verdict(mine, theirs):
    """current | reuploaded | changed | gone."""
    if theirs is None:
        return "gone"
    if mine.get("sha256") and theirs.get("sha256"):
        return "current" if mine["sha256"] == theirs["sha256"] else "reuploaded"
    if theirs.get("size") is not None and mine["size"] != theirs["size"]:
        return "reuploaded"
    return "changed" if _ts(theirs.get("date")) > mine["mtime"] else "current"


def check_one(path):
    repo, rel = tmpldoctor.repo_path(path)
    out = {"repo": repo, "path": "", "verdict": "unknown", "title": "", "date": "",
           "hf_cache": False, "error": ""}
    try:
        mine = local(path)
        out["hf_cache"] = bool(mine["sha256"])
        theirs = remote(repo, rel)
        out["verdict"] = verdict(mine, theirs)
        if theirs:
            out.update(path=theirs["path"], title=theirs["title"], date=theirs["date"])
    except Exception as e:
        out["error"] = str(e)
    return out


def _read():
    try:
        with open(CACHE, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def cached(models):
    """The last scan, limited to `models` ({id: path}) whose file is unchanged."""
    d = _read()
    seen = d.get("models") or {}
    keep = {m: r for m, r in seen.items() if m in models and r.get("file") == models[m]}
    return {"checked_at": d.get("checked_at", 0), "models": keep}


def stale(d):
    return time.time() - (d.get("checked_at") or 0) > MAX_AGE


def scan(models):
    """Check every model ({id: path}) that came from a nameable HF repo."""
    named = {m: p for m, p in models.items() if p and tmpldoctor.repo_path(p)[0]}
    with concurrent.futures.ThreadPoolExecutor(4) as ex:
        res = dict(zip(named, ex.map(check_one, named.values())))
    for m, r in res.items():
        r["file"] = named[m]
    out = {"checked_at": time.time(), "models": res}
    try:
        atomicio.write_json(CACHE, out)
    except Exception:
        pass
    return out
