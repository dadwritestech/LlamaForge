"""Community recipe gallery: tested setups people share as files in the repo's
recipes/ folder (see recipes/README.md).

The panel lists them live from GitHub, so a merged recipe shows up without a
release; offline (or rate-limited) it falls back to the copy bundled with the
installed release. Gallery files are as untrusted as a pasted recipe: every
one goes through recipes.parse(), and the "about" text is length-capped plain
text the UI escapes.

Pure functions here; the cached network calls live at the bottom.
"""
import json, os, threading, time, urllib.request

import recipes

LISTING = "https://api.github.com/repos/dadwritestech/LlamaForge/contents/recipes"
RAW = "https://raw.githubusercontent.com/dadwritestech/LlamaForge/master/recipes/"
UA = {"User-Agent": "LlamaForge/1.0 (+local model manager)"}
TTL = 6 * 3600
MAX_FILES = 200
MAX_BYTES = 64 * 1024
_ABOUT = {"title": 80, "hardware": 120, "author": 40, "notes": 400}


def _about(src):
    a = src.get("about") if isinstance(src.get("about"), dict) else {}
    out = {}
    for k, limit in _ABOUT.items():
        v = " ".join(str(a.get(k) or "").split())     # newlines/tabs -> single spaces
        out[k] = v[:limit]
    return out


def entries(files, sections=None, known_keys=None):
    """[(file name, JSON text)] -> gallery entries, skipping anything that
    isn't a valid recipe. `sections` (models.ini) marks models already here."""
    out = []
    for fname, text in files:
        try:
            src = json.loads(text)
            r = recipes.parse(src, known_keys)
        except (ValueError, TypeError):
            continue
        about = _about(src)
        stem = os.path.splitext(os.path.basename(fname))[0]
        out.append({"id": stem, "title": about["title"] or r["name"], "hardware": about["hardware"],
                    "author": about["author"], "notes": about["notes"],
                    "model": r["model"], "settings": r["settings"], "engine": r["engine"],
                    "dropped": r["dropped"],
                    "have": bool(sections) and recipes.match_local(r["model"], sections) is not None,
                    "recipe": {"llamaforge_recipe": recipes.VERSION, "name": r["name"],
                               "model": r["model"], "settings": r["settings"], "engine": r["engine"]}})
    out.sort(key=lambda e: e["title"].lower())
    return out


def remote_names(listing):
    """Recipe file names from a GitHub contents listing, ignoring anything odd."""
    if not isinstance(listing, list):
        raise ValueError("unexpected listing")
    names = [it.get("name", "") for it in listing
             if isinstance(it, dict) and it.get("type") == "file"
             and it.get("name", "").endswith(".json") and "/" not in it.get("name", "")
             and (it.get("size") or 0) <= MAX_BYTES]
    return sorted(names)[:MAX_FILES]


def bundled(folder):
    """The recipes/ folder shipped with this install."""
    out = []
    try:
        names = sorted(n for n in os.listdir(folder) if n.endswith(".json"))[:MAX_FILES]
    except OSError:
        return out
    for n in names:
        p = os.path.join(folder, n)
        if os.path.getsize(p) <= MAX_BYTES:
            with open(p, encoding="utf-8") as f:
                out.append((n, f.read()))
    return out


# ---------- network (cached) ----------

def _get(url, timeout=15):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(MAX_BYTES + 1).decode("utf-8")


def remote(get=_get):
    names = remote_names(json.loads(get(LISTING)))
    return [(n, get(RAW + n)) for n in names]


_cache, _lock = {}, threading.Lock()


def files(folder, force=False, get=_get):
    """(files, source): live from GitHub when reachable, else the bundled copy."""
    with _lock:
        hit = _cache.get("remote")
        if hit and not force and time.time() - hit[0] < TTL:
            return hit[1], "github"
    try:
        got = remote(get)
    except Exception:
        return bundled(folder), "bundled"
    with _lock:
        _cache["remote"] = (time.time(), got)
    return got, "github"
