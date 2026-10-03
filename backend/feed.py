""""New this week": what llama.cpp just learned to run, and LlamaForge updates.

llama.cpp publishes a release (b<N>) for every merged commit, with the commit
title as the first line of the body. Model-support commits follow a loose
convention ("model : add X", "add X support"), so a conservative title filter
finds them without any curated list. Each item is then marked against the
build the user's engine is running, so the panel can say "you have it" or
"update your engine for this".

Pure functions here; the cached network calls live at the bottom.
"""
import json, os, re, subprocess, threading, time, urllib.request

LLAMA_API = "https://api.github.com/repos/ggml-org/llama.cpp"
APP_API = "https://api.github.com/repos/dadwritestech/LlamaForge"
UA = {"User-Agent": "LlamaForge/1.0 (+local model manager)",
      "Accept": "application/vnd.github+json"}
TTL = 3600

_PR = re.compile(r"\s*\(#(\d+)\)\s*$")
_PREFIX = re.compile(r"^(?:(models?|llama|convert)\s*:\s*)?(.*)$", re.I)
_VERB = re.compile(r"^(add|support)\b\s*(?:support\s+for\s+)?(.*)$", re.I)


def _strip_pr(title):
    return _PR.sub("", title).strip()


def is_model_support(title):
    """True for commits that add a model architecture, e.g. "model : add X",
    "add X support". Kernels, quant types and features are left out."""
    m = _PREFIX.match(_strip_pr(title))
    prefix, rest = m.group(1), m.group(2)
    if not prefix and re.match(r"^[\w\- ,/.]+?\s*:", rest):
        return False                      # some other subsystem: "metal : add ..."
    v = _VERB.match(rest)
    if not v:
        return False
    name = re.sub(r"\s+support$", "", v.group(2), flags=re.I).strip()
    if not prefix and not re.search(r"\bsupport\b", rest, re.I):
        return False
    if not name or "_" in name or re.search(r"\b(for|kernel|quant|ops?)\b", name, re.I):
        return False
    return bool(re.search(r"[A-Z0-9]", name) or re.search(r"\bmodels?$", name, re.I))


def parse_release(rel):
    """{build, tag, title, pr, date, url} for a b<N> release, else None."""
    tag = rel.get("tag_name") or ""
    body = rel.get("body") or ""
    if not re.fullmatch(r"b\d+", tag) or not body:
        return None
    m = re.search(r"<details[^>]*>\s*(.*?)\s*</details>", body, re.S)
    lines = [ln.strip() for ln in (m.group(1) if m else body).splitlines() if ln.strip()]
    if not lines:
        return None
    first = lines[0]
    pr = _PR.search(first)
    return {"build": int(tag[1:]), "tag": tag, "title": _strip_pr(first),
            "pr": f"https://github.com/ggml-org/llama.cpp/pull/{pr.group(1)}" if pr else None,
            "date": (rel.get("published_at") or "")[:10],
            "url": rel.get("html_url") or ""}


def engine_news(releases, current_build=None, limit=12):
    out = []
    for rel in releases:
        r = parse_release(rel)
        if r and is_model_support(r["title"]):
            r["have"] = None if current_build is None else current_build >= r["build"]
            out.append(r)
    out.sort(key=lambda r: -r["build"])
    return out[:limit]


def parse_version(text):
    """Build number from `llama-server --version`; None when unknown. Builds
    without git history (shallow clones, tarballs) report "version: 0"."""
    m = re.search(r"version:\s*(\d+)", text or "")
    return (int(m.group(1)) or None) if m else None


def _semver(tag):
    m = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", tag or "")
    return tuple(int(x) for x in m.groups()) if m else None


def app_update(installed, latest):
    """Offer an update only to installer-managed copies on a release tag;
    a git checkout updates with git pull."""
    tag = (latest or {}).get("tag_name")
    have, new = _semver(installed), _semver(tag)
    return {"installed": installed, "latest": tag,
            "name": (latest or {}).get("name") or tag,
            "url": (latest or {}).get("html_url"),
            "available": bool(have and new and new > have)}


# ---- cached network side ---------------------------------------------------
_cache, _lock = {}, threading.Lock()


def _get_json(url, timeout=20):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _cached(key, fn, force=False):
    with _lock:
        hit = _cache.get(key)
        if hit and not force and time.time() - hit[0] < TTL:
            return hit[1]
    val = fn()                     # failures raise and are not cached
    with _lock:
        _cache[key] = (time.time(), val)
    return val


def engine_build(server_bin):
    """Build number of the engine (`llama-server --version`), cached per binary."""
    if not server_bin or not os.path.isfile(server_bin):
        return None
    key = ("ver", server_bin, os.path.getmtime(server_bin))
    with _lock:
        if key in _cache:
            return _cache[key][1]
    try:
        r = subprocess.run([server_bin, "--version"], capture_output=True, text=True,
                           timeout=60, cwd=os.path.dirname(server_bin))
        build = parse_version(r.stdout + r.stderr)
    except (OSError, subprocess.SubprocessError):
        build = None
    with _lock:
        _cache[key] = (time.time(), build)
    return build


def llama_releases(force=False, get_json=_get_json, pages=3):
    """The last ~300 builds (llama.cpp releases several a day: about 5 weeks)."""
    def fetch():
        out = []
        for p in range(1, pages + 1):
            out += get_json(f"{LLAMA_API}/releases?per_page=100&page={p}")
        return out
    return _cached("llama", fetch, force)


def app_latest(force=False, get_json=_get_json):
    return _cached("app", lambda: get_json(f"{APP_API}/releases/latest"), force)
