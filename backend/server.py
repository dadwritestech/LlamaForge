"""LlamaForge backend: one local HTTP server that powers the whole GUI.

Serves the dashboard and a JSON API wiring together config, model tuning
(all knobs), the CMake build/update manager, hardware + prerequisite
detection, and drive scanning. Pure Python stdlib.
"""
import json, os, subprocess, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import config, argspec, hardware, prereqs, scanner, hub, router_ctl, stats
from builder import BuildManager

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB     = os.path.join(ROOT, "web")
LOGDIR  = os.path.join(ROOT, "logs")
BUILDER = BuildManager(LOGDIR)
DOWNLOADS = hub.DownloadManager()

def _tail_file(path, n):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.readlines()[-n:]

def router_log_tail(n=400):
    err = _tail_file(os.path.join(LOGDIR, "router.err.log"), n)
    out = _tail_file(os.path.join(LOGDIR, "router.out.log"), n)
    if not err and not out:
        return "(no router log yet - restart LlamaForge to start capturing router.err.log / router.out.log)"
    return "".join(out) + ("\n--- stderr ---\n" if out and err else "") + "".join(err)

def total_vram_mib():
    return sum(g["total"] for g in _gpu_telemetry() if "total" in g)

def download_dir():
    c = cfg()
    if c.get("model_dirs"):
        return os.path.join(c["model_dirs"][0], "LlamaForge-downloads")
    return os.path.join(ROOT, "models")
_SCHEMA = None       # cached knob schema
_SCHEMA_KEY = None   # (server_bin, mtime) the cache was built from

def cfg():          return config.load()
def router_base():  return f"http://127.0.0.1:{cfg()['router_port']}"

# ---------- router proxy ----------
def router(path, method="GET", body=None, timeout=30):
    url = router_base() + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        try: return e.code, json.loads(e.read().decode())
        except Exception: return e.code, {"error": str(e)}
    except Exception as e:
        return 599, {"error": str(e)}

def gpus():
    return hardware.detect_gpus_verbose() if hasattr(hardware, "detect_gpus_verbose") else _gpu_telemetry()

def _gpu_telemetry():
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"], text=True, timeout=8)
    except Exception as e:
        return [{"error": str(e)}]
    res = []
    for ln in out.strip().splitlines():
        f = [x.strip() for x in ln.split(",")]
        if len(f) >= 6:
            res.append({"index": int(f[0]), "name": f[1], "used": int(f[2]),
                        "total": int(f[3]), "util": int(f[4]), "temp": int(f[5])})
    return res

def schema():
    """Knob schema, cached per (server_bin path, binary mtime).

    Keying on the mtime means the cache self-invalidates when config.json is
    repointed at a different binary or the binary is rebuilt. Failed attempts
    are never cached, so fixing the config takes effect without a backend
    restart."""
    global _SCHEMA, _SCHEMA_KEY
    bin_ = cfg()["server_bin"]
    try:
        key = (bin_, os.path.getmtime(bin_))
    except OSError:
        key = (bin_, None)
    if _SCHEMA is None or _SCHEMA_KEY != key or _SCHEMA.get("error"):
        _SCHEMA = argspec.build_schema(bin_)
        _SCHEMA_KEY = key
    return _SCHEMA

# ---------- model list (router status + ini settings) ----------
def model_state():
    st, data = router("/models")
    rmap = {m["id"]: m for m in data.get("data", [])} if st == 200 else {}
    ini  = config.read_sections()
    glob = ini.get("*", {})
    models = []
    for mid, rm in rmap.items():
        if mid == "default":
            continue
        sect = ini.get(mid, {})
        models.append({
            "id": mid,
            "status": rm.get("status", {}).get("value", "unknown"),
            "failed": rm.get("status", {}).get("failed", False),
            "modalities": rm.get("architecture", {}).get("input_modalities", ["text"]),
            "in_ini": mid in ini,
            "settings": sect,       # only keys explicitly set for this model
            "eff_ctx": _eff(rm, glob, "ctx-size", "--ctx-size"),
        })
    # also expose ini-only models not yet known to a (possibly-down) router
    for name in ini:
        if name != "*" and name not in rmap:
            models.append({"id": name, "status": "offline", "failed": False,
                           "modalities": ["text"], "in_ini": True,
                           "settings": ini[name], "eff_ctx": ini[name].get("ctx-size", glob.get("ctx-size", "?"))})
    models.sort(key=lambda m: (m["status"] != "loaded", m["id"]))
    return {"models": models, "global": glob}

def _eff(rm, glob, key, flag):
    args = rm.get("status", {}).get("args", [])
    if flag in args:
        return args[args.index(flag) + 1]
    return glob.get(key, "?")

# ---------- HTTP ----------
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)): body = json.dumps(body).encode()
        elif isinstance(body, str):        body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, name, ctype):
        path = os.path.join(WEB, name)
        if not os.path.exists(path):
            return self._send(404, {"error": "not found"})
        with open(path, "rb") as f:
            self._send(200, f.read(), ctype)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/", "/index.html"): return self._file("index.html", "text/html; charset=utf-8")
        if p == "/app.js":            return self._file("app.js", "application/javascript; charset=utf-8")
        if p == "/api/state":
            s = model_state(); s["gpus"] = _gpu_telemetry(); s["config"] = cfg(); return self._send(200, s)
        if p == "/api/schema":   return self._send(200, schema())
        if p == "/api/gpus":     return self._send(200, {"gpus": _gpu_telemetry()})
        if p == "/api/setup":
            return self._send(200, {"prereqs": prereqs.status(), "hardware": hardware.recommend()})
        if p == "/api/build/info":
            c = cfg()
            return self._send(200, {
                "current": BUILDER.current_commit(c["llama_src"]),
                "updates": BUILDER.check_updates(c["llama_src"]),
                "recommended_flags": hardware.recommend()["cmake_flags"],
                "saved_flags": c.get("cmake_flags", {}),
            })
        if p == "/api/build/log":
            s = dict(BUILDER.state); s["log"] = BUILDER.tail(300); return self._send(200, s)
        if p == "/api/hub/progress":
            return self._send(200, DOWNLOADS.progress())
        if p == "/api/router/log":
            return self._send(200, {"log": router_log_tail(400)})
        if p == "/api/stats":
            return self._send(200, stats.TRACKER.summary())
        if p == "/api/scan/missing":
            ini = config.read_sections()
            st, data = router("/models")
            loaded = {m["id"] for m in data.get("data", [])
                      if st == 200 and m.get("status", {}).get("value") == "loaded"}
            missing = [{"id": sec, "model": kv["model"], "loaded": sec in loaded}
                       for sec, kv in ini.items()
                       if sec != "*" and kv.get("model") and not os.path.exists(kv["model"])]
            return self._send(200, {"missing": missing})
        if p == "/api/network":
            c = cfg()
            return self._send(200, {
                "host": c.get("router_host", "127.0.0.1"),
                "port": c["router_port"],
                "has_api_key": bool(c.get("router_api_key")),
                "lan_ip": router_ctl.lan_ip(),
                "router_running": router_ctl.is_running(c["router_port"]),
            })
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or "{}") if n else {}
        p = self.path.split("?")[0]

        if p == "/api/save":
            mid = body.get("model"); updates = body.get("settings", {})
            clean = {}
            for k, v in updates.items():
                v = ("" if v is None else str(v)).strip()
                clean[k] = None if v == "" else v
            config.set_keys(mid, clean)
            st, data = router("/models")
            running = any(m["id"] == mid and m["status"]["value"] == "loaded"
                          for m in data.get("data", [])) if st == 200 else False
            if running: router("/models/unload", "POST", {"model": mid})
            router("/models?reload=1")
            return self._send(200, {"ok": True, "was_running": running})

        if p == "/api/load":
            code, res = router("/models/load", "POST", {"model": body.get("model")})
            return self._send(200 if code == 200 else 400, res)
        if p == "/api/unload":
            code, res = router("/models/unload", "POST", {"model": body.get("model")})
            return self._send(200 if code == 200 else 400, res)

        if p == "/api/build/start":
            c = cfg()
            flags = body.get("flags") or c.get("cmake_flags") or hardware.recommend()["cmake_flags"]
            c["cmake_flags"] = flags; config.save(c)
            ok = BUILDER.start(c["llama_src"], c["build_dir"], flags,
                               pull=body.get("pull", True))
            return self._send(200, {"started": ok})

        if p == "/api/setup/install":
            ok, log = prereqs.install(body.get("tool", ""))
            return self._send(200, {"ok": ok, "log": log})

        if p == "/api/scan":
            roots = body.get("roots") or cfg().get("model_dirs") or None
            return self._send(200, {"entries": scanner.scan(roots)})

        if p == "/api/scan/apply":
            entries = body.get("entries", [])
            for e in entries:
                keys = {"model": e["model"]}
                if e.get("mmproj"): keys["mmproj"] = e["mmproj"]
                if e.get("embeddings"): keys["embeddings"] = "true"
                config.set_keys(e["id"], keys)
            config.apply_ctx_defaults()
            router("/models?reload=1")
            return self._send(200, {"ok": True, "added": len(entries)})

        if p == "/api/scan/prune":
            ids, removed = body.get("ids", []), []
            st, data = router("/models")
            loaded = {m["id"] for m in data.get("data", [])
                      if st == 200 and m.get("status", {}).get("value") == "loaded"}
            for mid in ids:
                sect = config.read_sections().get(mid)
                if sect is None:
                    continue
                mpath = sect.get("model")
                if mpath and os.path.exists(mpath):
                    continue                     # file reappeared - don't remove
                if mid in loaded:
                    router("/models/unload", "POST", {"model": mid})
                if config.remove_section(mid):
                    removed.append(mid)
            if removed:
                router("/models?reload=1")
            return self._send(200, {"removed": removed})

        if p == "/api/hub/search":
            try:
                res = hub.search(body.get("query", ""), body.get("sort", "downloads"))
                return self._send(200, {"results": res, "vram_mib": total_vram_mib()})
            except Exception as e:
                return self._send(200, {"error": str(e), "results": []})

        if p == "/api/hub/files":
            try:
                return self._send(200, hub.files(body.get("repo", ""), total_vram_mib()))
            except Exception as e:
                return self._send(200, {"error": str(e), "files": [], "mmproj": []})

        if p == "/api/hub/download":
            repo   = body.get("repo", "")
            first  = body.get("path", "")
            shards = int(body.get("shards", 1))
            paths  = hub.shard_paths(first, shards)
            if body.get("mmproj"):
                paths.append(body["mmproj"])
            dest = os.path.join(download_dir(),
                                repo.replace("/", "--"))
            ok = DOWNLOADS.start(repo, paths, dest)
            return self._send(200, {"started": ok, "dest": dest})

        if p == "/api/hub/add":
            # register a finished download in models.ini
            path = body.get("path", "")
            if not path or not os.path.exists(path):
                return self._send(400, {"error": "file not found"})
            entries = scanner.build_entries(
                [os.path.join(os.path.dirname(path), f)
                 for f in os.listdir(os.path.dirname(path)) if f.lower().endswith(".gguf")])
            for e in entries:
                keys = {"model": e["model"]}
                if e.get("mmproj"): keys["mmproj"] = e["mmproj"]
                if e.get("embeddings"): keys["embeddings"] = "true"
                config.set_keys(e["id"], keys)
            config.apply_ctx_defaults()
            router("/models?reload=1")
            return self._send(200, {"ok": True, "added": [e["id"] for e in entries]})

        if p == "/api/config":
            c = cfg(); c.update(body or {}); config.save(c)
            return self._send(200, {"ok": True, "config": c})

        if p == "/api/network":
            c = cfg()
            host = body.get("host", "127.0.0.1")
            api_key = body.get("api_key")
            if api_key is None:
                api_key = c.get("router_api_key", "")   # field left blank -> keep existing key
            c["router_host"] = host
            c["router_api_key"] = api_key
            config.save(c)
            ok, err = router_ctl.restart(c["server_bin"], c["models_ini"], c["router_port"],
                                          host, api_key, LOGDIR)
            return self._send(200 if ok else 500, {"ok": ok, "error": err, "host": host})

        return self._send(404, {"error": "not found"})

def main():
    port = cfg()["panel_port"]
    print(f"LlamaForge -> http://127.0.0.1:{port}")
    try:                    # backfill ctx-size defaults, then nudge the router
        if config.apply_ctx_defaults().get("changed"):
            router("/models?reload=1")
    except Exception:
        pass
    stats.TRACKER.start()   # background usage poller
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()

if __name__ == "__main__":
    main()
