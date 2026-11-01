"""Record the live demo's data from a running panel.

The demo (docs/demo/, built into site/demo/ by tools/build_docs.py) is the real
web/ UI with fetch() answered from docs/demo/fixtures.json. This script fills
that file from a panel you are running, so the demo shows real screens:

    python tools/demo_record.py [--port 8099]

It keeps only the models in KEEP (others are dropped from every list and map),
replaces paths, hostnames, LAN addresses and secrets with neutral ones, stages
the endpoints that would leak or hang (build info, logs), and refuses to write
if anything from BANNED is still in the output. Pure stdlib.
"""
import argparse
import json
import os
import re
import socket
import sys
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "demo", "fixtures.json")

# The models a visitor sees. The 27B IQ4_XS carries the Auto-Forge result.
KEEP = [
    "qwen3.8-27b-ud-iq4-xs",
    "qwen3.6-35b-a3b-ud-q6-k",
    "gemma-4-31b-it-ud-q6-k-xl",
    "gemma-4-26b-a4b-it-mxfp4-moe",
    "qwen3.5-9b-deepseek-v4-flash-q8-0",
    "ministral-3-3b-instruct-2512-q4-k-m",
    "nomic-embed-text-v1.5.q4-k-m",
]
DEMO_HOME = r"D:\lf-demo"
DEMO_HOST = "forge-box"
DEMO_IP = "192.168.1.20"
SECRET_KEYS = re.compile(r"(api_?key|token|secret|password|auth)", re.I)
IP = re.compile(r"\b(?:10|100|172|192)\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
# Anything left over that must never ship. Checked case-insensitively.
BANNED = ["vamsi", "bhaskar", "uncen", "hearthfire", "turbofc", "abliterat", "nsfw", "c:\\\\users",
          "c:/users", "/home/", "\\\\appdata\\\\"]
# Hugging Face search results with these in the repo name are left out.
NSFW_REPO = re.compile(r"uncen|abliterat|nsfw|erotic|lewd|roleplay|\brp\b|hentai", re.I)

GETS = [
    "/api/state", "/api/feed", "/api/gpus", "/api/network", "/api/setup", "/api/stats",
    "/api/tts/status", "/api/vllm/setup", "/api/vllm/version", "/api/wiki/docs",
    "/api/wiki/profiles", "/api/docs", "/api/embers", "/api/embers/templates", "/api/elastic",
    "/api/idle", "/api/reclaim", "/api/reupload", "/api/rig", "/api/starters",
    "/api/recipes/gallery", "/api/pi/status", "/api/mcp/setup", "/api/presets", "/api/slots",
    "/api/schema", "/api/hub/progress", "/api/engine/prebuilt",
]
PER_MODEL = [
    "/api/forge/status?model={m}", "/api/forge/share?model={m}", "/api/model/metadata?model={m}",
    "/api/slots/plan?model={m}&role=main", "/api/slots/plan?model={m}&role=worker",
]
# Staged rather than recorded: the real ones name a private fork branch, or
# hold raw logs with paths in them.
STAGED = {
    "GET /api/build/info": {"ok": True, "repo": "https://github.com/ggml-org/llama.cpp", "branch": "master",
                            "built": True, "behind": 0, "target": "llamacpp"},
    "GET /api/build/log": {"running": False, "log": "", "ok": True},
    "GET /api/router/log": {"log": ""},
    "GET /api/embers/log": {"log": ""},
    # the real one carries the pairing link, token included
    "GET /api/phone": {"enabled": False, "running": False, "port": 8092, "urls": []},
}


def get(port, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path), data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def _is_model_id(s, known):
    return isinstance(s, str) and s in known


def prune(v, known, keep):
    """Drop every list item / map entry that is about a model not in keep."""
    if isinstance(v, dict):
        out = {}
        for k, x in v.items():
            if _is_model_id(k, known) and k not in keep:
                continue
            out[k] = prune(x, known, keep)
        return out
    if isinstance(v, list):
        out = []
        for x in v:
            if isinstance(x, dict) and any(_is_model_id(x.get(f), known) and x.get(f) not in keep
                                           for f in ("id", "model", "name")):
                continue
            if _is_model_id(x, known) and x not in keep:
                continue
            out.append(prune(x, known, keep))
        return out
    return v


def scrub(v, subs, key=""):
    if isinstance(v, dict):
        return {k: scrub(x, subs, k) for k, x in v.items()}
    if isinstance(v, list):
        return [scrub(x, subs, key) for x in v]
    if isinstance(v, str):
        if v and SECRET_KEYS.search(key) and not key.endswith(("_set", "_ok")):
            return "demo-key"
        for a, b in subs:
            v = re.sub(re.escape(a), lambda _m: b, v, flags=re.I)
        return IP.sub(DEMO_IP, v)
    return v


def substitutions():
    """Concrete strings to replace, longest first: home dirs, the LlamaForge
    folder, model folders, the machine's name."""
    home = os.path.expanduser("~")
    subs = [(ROOT, DEMO_HOME + r"\LlamaForge"), (ROOT.replace("\\", "/"), "D:/lf-demo/LlamaForge"),
            (home, DEMO_HOME), (home.replace("\\", "/"), "D:/lf-demo")]
    host = socket.gethostname()
    if host:
        subs.append((host, DEMO_HOST))
    try:
        sys.path.insert(0, os.path.join(ROOT, "backend"))
        import config
        cfg = config.load()
        for k, v in cfg.items():
            if isinstance(v, str) and SECRET_KEYS.search(k) and len(v) >= 8:
                subs.append((v, "demo-key"))
            elif isinstance(v, str) and re.match(r"^[A-Za-z]:[\\/]", v) and len(v) > 3:
                subs.append((v, DEMO_HOME + "\\" + k))
    except Exception as e:  # recorder still works without config access
        print("warning: config not read:", e, file=sys.stderr)
    return sorted(subs, key=lambda s: -len(s[0]))


def _ids_in(v, out):
    if isinstance(v, dict):
        for f in ("id", "model"):
            if isinstance(v.get(f), str):
                out.add(v[f])
        for x in v.values():
            _ids_in(x, out)
    elif isinstance(v, list):
        for x in v:
            _ids_in(x, out)
    return out


def drive_paths(v):
    """Any remaining absolute Windows path: keep only its last part under D:\\lf-demo."""
    if isinstance(v, dict):
        return {k: drive_paths(x) for k, x in v.items()}
    if isinstance(v, list):
        return [drive_paths(x) for x in v]
    if isinstance(v, str):
        # an interpreter path (the MCP setup command) says only which interpreter
        v = re.sub(r"[A-Za-z]:[\\/][^\s\"']*?[\\/](pythonw?(?:\.exe)?)\b", lambda m: m.group(1), v, flags=re.I)
        return re.sub(r"\b[A-Za-z]:[\\/](?![\\/]?lf-demo)(?:[^\s\"'<>|]*[\\/])?([^\s\\/\"'<>|]*)",
                      lambda m: DEMO_HOME + "\\" + m.group(1), v)
    return v


def check(text):
    low = text.lower()
    return [b for b in BANNED if b in low]


def record(port):
    state = get(port, "/api/state")
    known = {m.get("id") for m in state.get("models", []) if isinstance(m, dict)} | set(KEEP)
    keep = set(KEEP)
    fx = {}
    for p in GETS:
        try:
            fx["GET " + p] = get(port, p)
        except Exception as e:
            print("skip", p, e, file=sys.stderr)
    for m in KEEP:
        for t in PER_MODEL:
            p = t.format(m=urllib.parse.quote(m))
            try:
                fx["GET " + p] = get(port, p)
            except Exception as e:
                print("skip", p, e, file=sys.stderr)
    for d in (fx.get("GET /api/docs") or {}).get("pages", []):
        p = "/api/docs/page?slug=" + d["slug"]
        fx["GET " + p] = get(port, p)
    for e in (fx.get("GET /api/embers") or {}).get("embers", []):
        p = "/api/embers/brief?id=" + urllib.parse.quote(e["id"])
        try:
            fx["GET " + p] = get(port, p)
        except Exception as ex:
            print("skip", p, ex, file=sys.stderr)
    # read-only POSTs the demo answers: the Discover tab's opening search, and
    # the file list of the repos the kept models come from
    try:
        fx["POST /api/hub/search"] = get(port, "/api/hub/search", {"query": "", "sort": "trending"})
    except Exception as e:
        print("skip hub search", e, file=sys.stderr)
    hs = fx.get("POST /api/hub/search")
    if isinstance(hs, dict) and isinstance(hs.get("results"), list):
        hs["results"] = [r for r in hs["results"] if not NSFW_REPO.search(str(r.get("repo", "")))]
    # models Stats remembers but that are gone from disk count as models too
    known |= _ids_in(fx.get("GET /api/stats"), set()) - {None}
    fx.update(STAGED)
    fx = drive_paths(scrub(prune(fx, known, keep), substitutions()))
    text = json.dumps(fx, ensure_ascii=False, sort_keys=True, indent=0)
    bad = check(text)
    if bad:
        sys.exit("refusing to write fixtures: still contains %s" % bad)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")
    print("wrote %s: %d endpoints, %d KB" % (OUT, len(fx), len(text) // 1024))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    record(ap.parse_args().port)
