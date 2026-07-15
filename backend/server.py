"""LlamaForge backend: one local HTTP server that powers the whole GUI.

Serves the dashboard and a JSON API wiring together config, model tuning
(all knobs), the CMake build/update manager, hardware + prerequisite
detection, and drive scanning. Pure Python stdlib.
"""
import json, os, subprocess, urllib.request, urllib.error, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import config, argspec, hardware, osplat, prereqs, scanner, hub, router_ctl, stats
import wsl, vllm_ctl, vllm_registry, vllm_setup, vllm_job, vllm_hub, vllm_download

# vLLM is managed through WSL2, so the whole vLLM surface is Windows-only.
VLLM_SUPPORTED = osplat.IS_WIN
from builder import BuildManager

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB     = os.path.join(ROOT, "web")
LOGDIR  = os.path.join(ROOT, "logs")
BUILDER = BuildManager(LOGDIR)
DOWNLOADS = hub.DownloadManager()

VLLM_SETUP_JOB = vllm_job.WslJob(LOGDIR, "vllm-setup.log")

_VLLM = None
def vllm_mgr():
    """Lazily build the vLLM manager from current config."""
    global _VLLM
    c = cfg()
    distro = c.get("wsl_distro") or wsl.default_distro()
    if _VLLM is None:
        _VLLM = vllm_ctl.Manager(
            distro=distro, port=c.get("vllm_port", 8081),
            venv="~/.llamaforge/vllm-venv", logdir=LOGDIR)
        _VLLM.reconcile()
    else:
        _VLLM.distro = distro
        _VLLM.port = c.get("vllm_port", 8081)
    return _VLLM

_VLLM_DL = None
def vllm_dl():
    global _VLLM_DL
    c = cfg()
    distro = c.get("wsl_distro") or wsl.default_distro()
    if _VLLM_DL is None:
        _VLLM_DL = vllm_download.Manager(distro)
    else:
        _VLLM_DL.distro = distro
    return _VLLM_DL

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

def vllm_log_tail(n=400):
    err = _tail_file(os.path.join(LOGDIR, "vllm.err.log"), n)
    out = _tail_file(os.path.join(LOGDIR, "vllm.out.log"), n)
    if not err and not out:
        return "(no vLLM log yet - load a vLLM model to start capturing vllm.out/err.log)"
    return "".join(out) + ("\n--- stderr ---\n" if out and err else "") + "".join(err)

def total_vram_mib():
    return sum(g["total"] for g in _gpu_telemetry() if "total" in g)

def download_dir():
    c = cfg()
    if c.get("model_dirs"):
        return os.path.join(c["model_dirs"][0], "LlamaForge-downloads")
    return os.path.join(ROOT, "models")
_SCHEMA = None   # cached knob schema

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
    if osplat.IS_MAC:
        return osplat.mac_gpu_telemetry()
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
    global _SCHEMA
    if _SCHEMA is None:
        _SCHEMA = argspec.build_schema(cfg()["server_bin"])
    return _SCHEMA

_VLLM_SCHEMA = None
def vllm_schema():
    global _VLLM_SCHEMA
    if _VLLM_SCHEMA is None:
        import vllm_argspec
        c = cfg()
        distro = c.get("wsl_distro") or wsl.default_distro()
        _VLLM_SCHEMA = vllm_argspec.build_schema(distro, "~/.llamaforge/vllm-venv")
    return _VLLM_SCHEMA

def installed_repos(results, ini_sections, vllm_ids):
    """Which Discover results are already on this machine. GGUF downloads land
    in a '<org>--<name>' folder that models.ini paths retain; vLLM registry
    keys are the repo ids themselves."""
    blob = " ".join(kv.get("model", "") for kv in ini_sections.values())
    vset = set(vllm_ids)
    out = []
    for r in results:
        repo = r.get("repo", "")
        if repo and (repo in vset or repo.replace("/", "--") in blob):
            out.append(repo)
    return out

def vllm_save(model_id, settings, is_running, restart):
    """Persist knob changes; restart the process if the model is loaded
    (vLLM has no hot reload). Returns whether a restart was triggered."""
    vllm_registry.set_settings(model_id, settings)
    if is_running:
        restart(model_id)
        return True
    return False

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
            "file_gib": _file_gib(sect.get("model")),
        })
    # also expose ini-only models not yet known to a (possibly-down) router
    for name in ini:
        if name != "*" and name not in rmap:
            models.append({"id": name, "status": "offline", "failed": False,
                           "modalities": ["text"], "in_ini": True,
                           "settings": ini[name], "eff_ctx": ini[name].get("ctx-size", glob.get("ctx-size", "?")),
                           "file_gib": _file_gib(ini[name].get("model"))})
    models.sort(key=lambda m: (m["status"] != "loaded", m["id"]))
    return {"models": models, "global": glob}

def _file_gib(path):
    """Model file size in GiB, or None (missing path / file gone)."""
    try:
        return round(os.path.getsize(path) / 1024**3, 2) if path else None
    except OSError:
        return None

def _eff(rm, glob, key, flag):
    args = rm.get("status", {}).get("args", [])
    if flag in args:
        return args[args.index(flag) + 1]
    return glob.get(key, "?")

# ---------- unified model list (llama.cpp + vLLM) ----------
STATE_MAP = {"ready": "loaded", "loading": "loading", "starting": "loading",
             "failed": "offline", "stopped": "offline"}

def merge_vllm_models(base, vllm_status, vllm_ids, router_port):
    """Tag every existing (llama.cpp) row and append vLLM rows.
    base is model_state()'s dict; vllm_status is Manager.status();
    vllm_ids is vllm_registry.models()."""
    llama_ep = f"http://127.0.0.1:{router_port}"
    for m in base["models"]:
        m["backend"] = "llamacpp"
        if m.get("status") == "loaded":
            m["endpoint"] = llama_ep
    live = {i["model_id"]: i for i in vllm_status}
    for mid in vllm_ids:
        inst = live.get(mid)
        status = STATE_MAP.get(inst["state"], "offline") if inst else "offline"
        entry = vllm_registry.load().get(mid, {})
        row = {"id": mid, "backend": "vllm", "status": status,
               "failed": bool(inst and inst["state"] == "failed"),
               "modalities": ["text"], "in_ini": True,
               "settings": entry.get("settings", {}),
               "eff_ctx": vllm_registry.effective_settings(mid).get("max-model-len", "?"),
               "file_gib": round(entry.get("size_bytes", 0) / 1024**3, 2)
                           if entry.get("size_bytes") else None}
        if inst and status == "loaded":
            row["endpoint"] = inst["endpoint"]
        base["models"].append(row)
    return base

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

    def _vllm_gate(self, p):
        """vLLM rides on WSL2; short-circuit its routes on Linux/macOS."""
        if p.startswith("/api/vllm/") and not VLLM_SUPPORTED:
            if p == "/api/vllm/setup":   # the Setup tab probes this one
                self._send(200, {"supported": False, "wsl": {"present": False},
                                 "distros": [], "gpu": {"present": False},
                                 "vllm": {"present": False, "version": ""},
                                 "setup_job": {"running": False}, "setup_log": ""})
            else:
                self._send(400, {"error": "vLLM backend requires Windows + WSL2"})
            return True
        return False

    def do_GET(self):
        p = self.path.split("?")[0]
        qs = urllib.parse.parse_qs(self.path.split("?", 1)[1]) if "?" in self.path else {}
        force = qs.get("force", ["0"])[0] in ("1", "true")
        if self._vllm_gate(p):
            return
        if p in ("/", "/index.html"): return self._file("index.html", "text/html; charset=utf-8")
        if p == "/app.js":            return self._file("app.js", "application/javascript; charset=utf-8")
        if p == "/api/state":
            s = model_state()
            c = cfg()
            if VLLM_SUPPORTED:
                mgr = vllm_mgr()
                s = merge_vllm_models(s, mgr.status(), vllm_registry.models(), c["router_port"])
            else:   # still tags llama.cpp rows with backend + endpoint
                s = merge_vllm_models(s, [], [], c["router_port"])
            s["gpus"] = _gpu_telemetry(); s["config"] = c
            s["platform"] = osplat.current()
            s["vllm_supported"] = VLLM_SUPPORTED
            s["onboarding"] = {
                "server_bin_ok": bool(c.get("server_bin")) and os.path.exists(c["server_bin"]),
                "model_count": len(s["models"]),
            }
            return self._send(200, s)
        if p == "/api/schema":   return self._send(200, schema())
        if p == "/api/gpus":     return self._send(200, {"gpus": _gpu_telemetry()})
        if p == "/api/setup":
            return self._send(200, {"prereqs": prereqs.status(), "hardware": hardware.recommend()})
        if p == "/api/build/info":
            c = cfg()
            return self._send(200, {
                "current": BUILDER.current_commit(c["llama_src"]),
                "updates": BUILDER.check_updates(c["llama_src"], force=force),
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
        if p == "/api/vllm/log":
            return self._send(200, {"log": vllm_log_tail(400)})
        if p == "/api/vllm/setup":
            c = cfg()
            distro = c.get("wsl_distro") or wsl.default_distro()
            s = vllm_setup.status(distro)
            s["supported"] = True
            s["setup_job"] = VLLM_SETUP_JOB.progress()
            s["setup_log"] = VLLM_SETUP_JOB.tail(300)
            return self._send(200, s)
        if p == "/api/vllm/schema":
            return self._send(200, vllm_schema())
        if p == "/api/vllm/version":
            c = cfg()
            distro = c.get("wsl_distro") or wsl.default_distro()
            return self._send(200, {
                "installed": vllm_setup._vllm_version(distro),
                "latest": vllm_setup.latest_pypi_version(force=force),
            })
        if p == "/api/vllm/hub/progress":
            return self._send(200, vllm_dl().progress())
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or "{}") if n else {}
        p = self.path.split("?")[0]
        if self._vllm_gate(p):
            return

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
                inst = installed_repos(res, config.read_sections(),
                                       vllm_registry.models() if VLLM_SUPPORTED else [])
                return self._send(200, {"results": res, "vram_mib": total_vram_mib(),
                                        "installed": inst})
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

        if p == "/api/hub/cancel":
            return self._send(200, {"ok": DOWNLOADS.cancel()})

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

        if p == "/api/stats/reset":
            stats.TRACKER.reset()
            return self._send(200, {"ok": True})

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

        if p == "/api/vllm/load":
            mid = body.get("model", "")
            entry = vllm_registry.load().get(mid)
            if not entry:
                return self._send(400, {"error": f"unknown vLLM model: {mid}"})
            ref = entry.get("wsl_path") or entry.get("repo") or mid
            flags = vllm_ctl.settings_to_flags(vllm_registry.effective_settings(mid))
            ok, err = vllm_mgr().start(mid, ref, flags)
            return self._send(200 if ok else 400, {"ok": ok, "error": err})

        if p == "/api/vllm/unload":
            vllm_mgr().stop(body.get("model", ""))
            return self._send(200, {"ok": True})

        if p == "/api/vllm/setup/install":
            c = cfg()
            distro = body.get("distro") or c.get("wsl_distro") or wsl.default_distro()
            if body.get("distro"):
                c["wsl_distro"] = body["distro"]; config.save(c)
            ok = VLLM_SETUP_JOB.start(vllm_setup.install_script(), distro)
            return self._send(200, {"started": ok})

        if p == "/api/vllm/save":
            mid = body.get("model", "")
            settings = body.get("settings", {})
            mgr = vllm_mgr()
            running = any(i["model_id"] == mid and i["state"] in ("ready", "loading")
                          for i in mgr.status())
            def _restart(m):
                entry = vllm_registry.load().get(m, {})
                ref = entry.get("wsl_path") or entry.get("repo") or m
                mgr.stop(m)
                flags = vllm_ctl.settings_to_flags(vllm_registry.effective_settings(m))
                mgr.start(m, ref, flags)
            restarted = vllm_save(mid, settings, running, _restart)
            return self._send(200, {"ok": True, "restarted": restarted})

        if p == "/api/vllm/update":
            c = cfg()
            distro = c.get("wsl_distro") or wsl.default_distro()
            ok = VLLM_SETUP_JOB.start(vllm_setup.update_script(), distro)
            return self._send(200, {"started": ok})

        if p == "/api/vllm/hub/search":
            try:
                res = vllm_hub.search(body.get("query", ""), body.get("sort", "downloads"))
                inst = installed_repos(res, {}, vllm_registry.models())
                return self._send(200, {"results": res, "vram_mib": total_vram_mib(),
                                        "installed": inst})
            except Exception as e:
                return self._send(200, {"error": str(e), "results": []})

        if p == "/api/vllm/hub/info":
            try:
                return self._send(200, vllm_hub.repo_info(body.get("repo", ""), total_vram_mib()))
            except Exception as e:
                return self._send(200, {"error": str(e)})

        if p == "/api/vllm/hub/download":
            repo = body.get("repo", "")
            info = {}
            try:
                info = vllm_hub.repo_info(repo, total_vram_mib())
            except Exception:
                pass
            ok = vllm_dl().start(repo, int(body.get("size_bytes") or info.get("size_bytes") or 0))
            return self._send(200, {"started": ok})

        if p == "/api/vllm/hub/register":
            repo = body.get("repo", "")
            dl = vllm_dl()
            vllm_registry.upsert(repo, {
                "repo": repo, "wsl_path": dl.wsl_path(repo),
                "size_bytes": int(body.get("size_bytes") or 0),
                "quant": body.get("quant", "")})
            return self._send(200, {"ok": True, "added": repo})

        if p == "/api/vllm/delete":
            repo = body.get("model", "")
            ok, err = vllm_dl().delete(repo)
            if ok:
                vllm_registry.remove(repo)
            return self._send(200 if ok else 500, {"ok": ok, "error": err})

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
