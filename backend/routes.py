"""The LlamaForge JSON API: one named handler per route, plus the tables that
map a path to it.

Why this is a table and not an if-chain: every handler here is a plain function
of (request) -> (status, payload), so it can be called directly from a test with
no socket, no threads and no live router. The dispatch in server.py does nothing
but look the path up, which keeps HTTP plumbing and API behaviour separable.

Handler contract
----------------
    def handler(req) -> (status, payload) | (status, payload, content_type)

`req` is a Req: .body (parsed JSON for POST, {} for GET), .qs (parsed query
string, values already unwrapped to single strings), and .headers (lower-cased).
Returning a dict/list gets JSON-encoded; returning bytes/str needs a
content_type. Raising ApiError(status, message) produces {"error": message}.

Streaming responses (the Anthropic and OpenAI SSE proxies) are not in these
tables: they write to the socket themselves and stay in server.py.
"""
import json, os, subprocess, sys, urllib.request, urllib.error, urllib.parse

import config, argspec, hardware, osplat, prereqs, scanner, hub, router_ctl, stats
import autotune, anthropic_shim, agentsetup, wiki, docs
import vram_predict
import wsl, vllm_ctl, vllm_registry, vllm_setup, vllm_job, vllm_hub, vllm_download
import gguf, diag, backends
from builder import BuildManager

# vLLM is managed through WSL2, so the whole vLLM surface is Windows-only.
VLLM_SUPPORTED = osplat.IS_WIN

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB     = os.path.join(ROOT, "web")
LOGDIR  = os.path.join(ROOT, "logs")
# Each engine records the binary its own build produced (see _record_server_bin).
# Resolved at call time, so the helper can live further down with its siblings.
BUILDER_LLAMA   = BuildManager(LOGDIR, "build",
                               on_built=lambda p: _record_server_bin("server_bin", p))
BUILDER_IKLLAMA = BuildManager(LOGDIR, "build-ikllama",
                               on_built=lambda p: _record_server_bin("ik_llama_server_bin", p))

def _builder_for(target):
    return BUILDER_IKLLAMA if target == "ikllama" else BUILDER_LLAMA
DOWNLOADS = hub.DownloadManager()

VLLM_SETUP_JOB = vllm_job.WslJob(LOGDIR, "vllm-setup.log")

# The engine registry is built at the bottom of this module, once the helpers it
# depends on (model_state, router, cfg, vllm_mgr, ...) exist. Backends receive
# this module itself as their dependency bundle: it keeps them free of import
# cycles and lets a test hand in a stub with the same handful of functions.
REGISTRY = None


class ApiError(Exception):
    """Raise from a handler to return an error payload with a status."""
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


class Req:
    """One request, reduced to what handlers actually need."""
    __slots__ = ("body", "qs", "headers", "path")

    def __init__(self, body=None, qs=None, headers=None, path=""):
        self.body = body or {}
        self.qs = qs or {}
        self.headers = headers or {}
        self.path = path

    def q(self, name, default=""):
        """First value of a query parameter."""
        return self.qs.get(name, default)

    def flag(self, name):
        return str(self.qs.get(name, "")).lower() in ("1", "true", "yes")


# ---------------------------------------------------------------- shared state
_VLLM = None
_VLLM_DL = None
_SCHEMA = None          # cached knob schema (active engine)
_SCHEMA_KEY = None      # (server_bin, mtime) the cache was built from
_IK_SCHEMA = None       # cached schema for ik_llama binary
_IK_SCHEMA_KEY = None
_VLLM_SCHEMA = None


def cfg():          return config.load()
def router_base():  return f"http://127.0.0.1:{cfg()['router_port']}"


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


def _agent_endpoint(agent):
    c = cfg()
    if agent == "claude-code":
        return f"http://127.0.0.1:{c['panel_port']}"   # shim binds localhost only
    host = router_ctl.lan_ip() if c.get("router_host", "127.0.0.1") != "127.0.0.1" else "127.0.0.1"
    return f"http://{host}:{c['router_port']}/v1"


_AGENT_CONTEXT_FILE = {"claude-code": ".claude/CLAUDE.md",
                       "codex": ".codex/AGENTS.md", "pi": ".pi/AGENTS.md"}


def _wiki_export(body):
    agent = body.get("agent", "")
    path = body.get("path", "")
    composed = wiki.compose(body.get("profile", ""))
    if not path:
        rel = _AGENT_CONTEXT_FILE.get(agent)
        if not rel:
            return {"error": f"unknown agent: {agent}"}
        path = os.path.join(os.path.expanduser("~"), *rel.split("/"))
    return wiki.export_agent_file(path, composed)


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


def _cached_schema(bin_path, cache_holder):
    """Schema cache keyed on (binary path, mtime). Returns (schema, key)."""
    schema, key = cache_holder
    try:
        new_key = (bin_path, os.path.getmtime(bin_path))
    except OSError:
        new_key = (bin_path, None)
    if schema is None or key != new_key or schema.get("error"):
        schema = argspec.build_schema(bin_path)
        key = new_key
    return schema, key


def schema():
    """Knob schema for the active engine, cached per (server_bin, mtime)."""
    global _SCHEMA, _SCHEMA_KEY
    c = cfg()
    if c.get("active_engine") == "ikllama":
        return ik_schema()
    bin_ = c["server_bin"]
    _SCHEMA, _SCHEMA_KEY = _cached_schema(bin_, (_SCHEMA, _SCHEMA_KEY))
    return _SCHEMA


def ik_schema():
    """Knob schema specifically for ik_llama's binary."""
    global _IK_SCHEMA, _IK_SCHEMA_KEY
    bin_ = cfg().get("ik_llama_server_bin", "")
    if not bin_:
        return {"error": "ik_llama_server_bin not configured"}
    _IK_SCHEMA, _IK_SCHEMA_KEY = _cached_schema(bin_, (_IK_SCHEMA, _IK_SCHEMA_KEY))
    return _IK_SCHEMA


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


# ---------- auto-tune ----------
def _find_model(model_id):
    for m in model_state().get("models", []):
        if m.get("id") == model_id:
            return m
    return None


def _autotune_recommend(body):
    mid = body.get("model", "")
    intent = body.get("intent", "balanced")
    m = _find_model(mid)
    if not m:
        return {"error": f"unknown model: {mid}"}
    # model_state() rows normally nest the file path under settings.model;
    # fall back to a top-level "model" key so callers passing a flatter shape
    # (e.g. tests) still work.
    path = m.get("model") or (m.get("settings") or {}).get("model") or ""
    meta = gguf.metadata(path) or {}
    try:
        size = os.path.getsize(path)
    except OSError:
        size = None
    hw = {"gpus": hardware.detect_gpus(), "cpu": hardware.detect_cpu()}
    pred = None
    try:
        if cfg().get("vram_predict_enabled", True) and path:
            pred = vram_predict.predict_local(path, size_bytes=size, cfg=cfg())
    except Exception:
        pred = None
    rec = autotune.recommend(meta, hw, intent, size_bytes=size, prediction=pred)
    rec.update({"model": mid, "intent": intent})
    return rec


def _autotune_refine(body):
    mid = body.get("model", "")
    intent = body.get("intent", "balanced")
    base = body.get("knobs")
    m = _find_model(mid)
    if not m:
        return {"error": f"unknown model: {mid}"}
    # If no base knobs provided, generate them via recommend first.
    if not base:
        rec = _autotune_recommend({"model": mid, "intent": intent})
        if "error" in rec:
            return rec
        base = rec.get("knobs") or {}

    def load_fn(knobs):
        config.set_keys(mid, knobs)
        router("/models?reload=1")
        code, res = router("/models/load", "POST", {"model": mid})
        if code >= 400:
            raise RuntimeError((res or {}).get("error", "load failed"))

    def measure_fn():
        """Send a real completion request and measure tok/s (generation only, excludes prompt eval)."""
        import time
        prompt = "Write a Python function that computes the Fibonacci sequence iteratively. Explain your approach briefly."
        payload = {"model": mid, "prompt": prompt, "n_predict": 200, "stream": True}
        url = router_base() + "/completion"
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
        tokens = 0
        first_tok = None
        last_tok = None
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                for line in r:
                    line = line.decode().strip()
                    if not line or not line.startswith("data: "):
                        continue
                    try:
                        obj = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    if obj.get("stop"):
                        break
                    content = obj.get("content", "")
                    if content:
                        tokens += 1
                        now = time.monotonic()
                        if first_tok is None:
                            first_tok = now
                        last_tok = now
        except Exception as e:
            return 0.0
        if first_tok is None or last_tok is None or tokens < 10:
            return 0.0
        elapsed = last_tok - first_tok
        if elapsed < 0.01:
            return 0.0
        return round(tokens / elapsed, 1)

    out = autotune.refine(base, intent, load_fn, measure_fn)
    out["model"] = mid
    return out


# ---------- unified model list (llama.cpp + vLLM) ----------
STATE_MAP = {"ready": "loaded", "loading": "loading", "starting": "loading",
             "failed": "offline", "stopped": "offline"}


def merge_vllm_models(base, vllm_status, vllm_ids, router_port):
    """Deprecated: superseded by backends.Registry.state(), which asks each
    engine for its own rows instead of grafting one engine onto another. Kept
    because it is the documented shape of a model row."""
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


# ---------- anthropic shim ----------
def _resolve_anthropic_model(requested):
    ids = {m.get("id") for m in model_state().get("models", [])}
    if requested in ids:
        return requested
    return cfg().get("anthropic_default_model") or requested


def _shim_auth_ok(headers):
    c = cfg()
    if c.get("router_host", "127.0.0.1") == "127.0.0.1":
        return True
    key = c.get("router_api_key", "")
    if not key:
        return True
    if headers.get("x-api-key") == key:
        return True
    return headers.get("authorization", "") == f"Bearer {key}"


def _router_openai(oai_body, stream=False):
    """POST the translated body to the router's OpenAI chat endpoint.
    Non-stream: returns (status, dict). Stream: returns (status, response) where
    response is the open urllib object to iterate for SSE lines."""
    url = router_base() + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    key = cfg().get("router_api_key", "")
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url, data=json.dumps(oai_body).encode(),
                                 method="POST", headers=headers)
    if stream:
        try:
            resp = urllib.request.urlopen(req, timeout=600)
            return resp.status, resp
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read().decode())
            except Exception:
                return e.code, {"error": str(e)}
        except Exception as e:
            return 599, {"error": str(e)}
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, {"error": str(e)}
    except Exception as e:
        return 599, {"error": str(e)}


def _inject_openai_system(body, composed):
    if not composed:
        return body
    msgs = list(body.get("messages") or [])
    if msgs and msgs[0].get("role") == "system":
        merged = composed + "\n\n" + (msgs[0].get("content") or "")
        msgs = [{"role": "system", "content": merged}] + msgs[1:]
    else:
        msgs = [{"role": "system", "content": composed}] + msgs
    return {**body, "messages": msgs}


def _inject_anthropic_system(body, composed):
    if not composed:
        return body
    sys = body.get("system")
    if isinstance(sys, str) and sys:
        return {**body, "system": composed + "\n\n" + sys}
    if isinstance(sys, list):
        return {**body, "system": [{"type": "text", "text": composed}] + sys}
    return {**body, "system": composed}


def _anthropic_messages(body, headers):
    """Non-streaming /v1/messages: returns (status, anthropic_json)."""
    model = _resolve_anthropic_model(body.get("model", ""))
    body = _inject_anthropic_system(body, wiki.compose(wiki.active_profile(model)))
    oai = anthropic_shim.to_openai_request({**body, "model": model, "stream": False})
    status, data = _router_openai(oai, stream=False)
    if status >= 400:
        msg = data.get("error") if isinstance(data, dict) else str(data)
        if isinstance(msg, dict):
            msg = msg.get("message", "upstream error")
        return anthropic_shim.anthropic_error(status,
            anthropic_shim.error_type_for_status(status), msg or "upstream error")
    return 200, anthropic_shim.to_anthropic_response(data, model)


def _write_anthropic_stream(write, model, status, resp):
    """Translate a router streaming response into Anthropic SSE and write it.
    `resp` is either an open urllib response (status < 400, iterate for lines)
    or an error dict (status >= 400). `write(bytes)` sends to the client."""
    if status >= 400:
        msg = resp.get("error") if isinstance(resp, dict) else str(resp)
        if isinstance(msg, dict):
            msg = msg.get("message", "upstream error")
        write(anthropic_shim._sse("error", {"type": "error", "error": {
            "type": anthropic_shim.error_type_for_status(status),
            "message": msg or "upstream error"}}))
        return
    for event in anthropic_shim.stream_anthropic_events(resp, model):
        write(event)


def _apply_knobs_and_reload(mid, clean):
    """Write knobs to models.ini, then make the router pick them up. A loaded
    model has to be unloaded first - llama.cpp reads args at load time.
    Returns whether it had been running."""
    config.set_keys(mid, clean)
    st, data = router("/models")
    running = any(m["id"] == mid and m["status"]["value"] == "loaded"
                  for m in data.get("data", [])) if st == 200 else False
    if running:
        router("/models/unload", "POST", {"model": mid})
    router("/models?reload=1")
    return running


def _clean_settings(updates):
    """Normalize a knob map from the UI: blank means 'unset this key'."""
    clean = {}
    for k, v in (updates or {}).items():
        v = ("" if v is None else str(v)).strip()
        clean[k] = None if v == "" else v
    return clean


def _reconcile_preset_binding(mid, preset, engine=None):
    """Apply only preset defaults LlamaForge still owns for one model.

    A section value is user-owned when it predates the binding or differs from
    the last materialized snapshot.  Such values are never changed or removed.
    Returns the delta written to models.ini.
    """
    engine = engine or cfg().get("active_engine", "llamacpp")
    desired = {key: value for key, value in _clean_settings(preset).items()
               if value is not None}
    previous = config.get_binding_snapshot(mid, engine)
    path = config.ini_path(engine)
    current = config.read_sections(path).get(mid, {})
    updates, owned = {}, {}

    for key, old_value in previous.items():
        if current.get(key) != old_value:
            continue                    # edited or deleted manually: relinquish it
        if key not in desired:
            updates[key] = None
            continue
        new_value = desired[key]
        owned[key] = new_value
        if new_value != old_value:
            updates[key] = new_value

    for key, new_value in desired.items():
        if key in previous:
            continue
        if key not in current:
            updates[key] = new_value
            owned[key] = new_value

    if updates:
        if engine == cfg().get("active_engine", "llamacpp"):
            _apply_knobs_and_reload(mid, updates)
        else:
            config.set_keys(mid, updates, path)
    config.set_binding_snapshot(mid, owned, engine)
    return updates


def _register_ggufs_beside(paths):
    """Add scanner-derived entries to models.ini and reload the router."""
    entries = scanner.build_entries(paths)
    for e in entries:
        keys = {"model": e["model"]}
        if e.get("mmproj"):
            keys["mmproj"] = e["mmproj"]
        if e.get("embeddings"):
            keys["embeddings"] = "true"
        config.set_keys(e["id"], keys)
    config.apply_ctx_defaults()
    router("/models?reload=1")
    return entries


# =============================================================== GET handlers

def get_state(req):
    c = cfg()
    s = REGISTRY.state()
    s["gpus"] = _gpu_telemetry()
    s["config"] = _public_config(c)
    s["platform"] = osplat.current()
    s["vllm_supported"] = VLLM_SUPPORTED
    s["backends"] = [b.name for b in REGISTRY.enabled()]
    s["active_engine"] = c.get("active_engine", "llamacpp")
    s["config_error"] = config.LOAD_ERROR
    s["onboarding"] = {
        "server_bin_ok": bool(c.get("server_bin")) and os.path.exists(c["server_bin"]),
        "model_count": len(s["models"]),
        "ui_mode": c.get("ui_mode", "lite"),
        "onboarded": bool(c.get("onboarded", False)),
    }
    return 200, s


def _public_config(c):
    """config.json as the dashboard sees it. The router API key is deliberately
    included: the Client-config modal shows the exact curl the user needs, and
    the panel is same-origin and localhost-only."""
    return dict(c)


def get_schema(req):
    return 200, schema()


def get_gpus(req):
    return 200, {"gpus": _gpu_telemetry()}


def get_setup(req):
    return 200, {"prereqs": prereqs.status(), "hardware": hardware.recommend()}


def get_build_info(req):
    c = cfg()
    target = req.q("target") or "llamacpp"
    builder = _builder_for(target)
    if target == "ikllama":
        src = c.get("ik_llama_src", "")
        remote = c.get("ik_llama_git_remote", "https://github.com/ikawrakow/ik_llama.cpp")
        saved_flags = c.get("ik_llama_cmake_flags", {})
    else:
        src = c["llama_src"]
        remote = c.get("git_remote", "https://github.com/ggml-org/llama.cpp")
        saved_flags = c.get("cmake_flags", {})
    return 200, {
        "target": target,
        "current": builder.current_commit(src),
        "updates": builder.check_updates(src, force=req.flag("force")),
        "recommended_flags": hardware.recommend()["cmake_flags"],
        "saved_flags": saved_flags,
        "remote": remote,
    }


def get_build_log(req):
    target = req.q("target") or "llamacpp"
    builder = _builder_for(target)
    s = dict(builder.state)
    s["log"] = builder.tail(300)
    s["target"] = target
    return 200, s


def get_hub_progress(req):
    return 200, DOWNLOADS.progress()


def get_router_log(req):
    return 200, {"log": router_log_tail(400)}


def get_stats(req):
    return 200, stats.TRACKER.summary()


def get_scan_missing(req):
    ini = config.read_sections()
    st, data = router("/models")
    loaded = {m["id"] for m in data.get("data", [])
              if st == 200 and m.get("status", {}).get("value") == "loaded"}
    missing = [{"id": sec, "model": kv["model"], "loaded": sec in loaded}
               for sec, kv in ini.items()
               if sec != "*" and kv.get("model") and not os.path.exists(kv["model"])]
    return 200, {"missing": missing}


def get_network(req):
    c = cfg()
    return 200, {
        "host": c.get("router_host", "127.0.0.1"),
        "port": c["router_port"],
        "has_api_key": bool(c.get("router_api_key")),
        "lan_ip": router_ctl.lan_ip(),
        "router_running": router_ctl.is_running(c["router_port"]),
    }


def get_vllm_log(req):
    return 200, {"log": vllm_log_tail(400)}


def get_vllm_setup(req):
    c = cfg()
    distro = c.get("wsl_distro") or wsl.default_distro()
    s = vllm_setup.status(distro)
    s["supported"] = True
    s["setup_job"] = VLLM_SETUP_JOB.progress()
    s["setup_log"] = VLLM_SETUP_JOB.tail(300)
    return 200, s


def get_vllm_schema(req):
    return 200, vllm_schema()


def get_vllm_version(req):
    c = cfg()
    distro = c.get("wsl_distro") or wsl.default_distro()
    return 200, {
        "installed": vllm_setup._vllm_version(distro),
        "latest": vllm_setup.latest_pypi_version(force=req.flag("force")),
    }


def get_vllm_hub_progress(req):
    return 200, vllm_dl().progress()


def get_model_metadata(req):
    sect = config.read_sections().get(req.q("model"), {})
    mpath = sect.get("model")
    meta = gguf.metadata(mpath) if mpath else None
    return 200, {"metadata": meta or {}}


def get_model_diag(req):
    mid = req.q("model")
    ini = config.read_sections()
    merged = dict(ini.get("*", {}))
    merged.update(ini.get(mid, {}))
    return 200, {"diag": diag.diagnose(router_log_tail(120), merged)}


def get_presets(req):
    return 200, {"presets": config.get_presets()}


def get_agent_config(req):
    agent = req.q("agent")
    model = req.q("model")
    small = req.q("small") or None
    inject = req.flag("inject")
    c = cfg()
    if inject and agent in ("codex", "pi"):
        host = router_ctl.lan_ip() if c.get("router_host", "127.0.0.1") != "127.0.0.1" else "127.0.0.1"
        endpoint = f"http://{host}:{c['panel_port']}/v1"
    else:
        endpoint = _agent_endpoint(agent)
    try:
        out = agentsetup.generate(agent, endpoint, c.get("router_api_key", ""),
                                  model, small, inject)
    except ValueError as e:
        raise ApiError(400, str(e))
    return 200, out


def get_wiki_docs(req):
    return 200, {"docs": wiki.list_docs()}


def get_wiki_doc(req):
    name = req.q("name")
    return 200, {"name": name, "text": wiki.read_doc(name)}


def get_wiki_profiles(req):
    return 200, {"profiles": wiki.get_profiles()}


def get_wiki_preview(req):
    return 200, {"text": wiki.compose(req.q("profile"))}


def get_docs(req):
    return 200, docs.manifest()


def get_docs_page(req):
    pg = docs.page(req.q("slug"))
    if not pg:
        raise ApiError(404, "no such page")
    return 200, pg


# ============================================================== POST handlers

# ---- engine-agnostic model verbs -------------------------------------------
#
# These dispatch on the model's own backend, so a third engine needs an entry in
# backends.Registry rather than a duplicate of every route below. The
# engine-specific paths (/api/load, /api/vllm/load, ...) remain as aliases: they
# are what the shipped dashboard and any existing scripts call.

def _backend_for(req):
    mid = req.body.get("model", "")
    return mid, REGISTRY.for_model(mid, req.body.get("backend", ""))


def post_model_load(req):
    mid, backend = _backend_for(req)
    ok, err = backend.load(mid)
    return (200 if ok else 400), {"ok": ok, "error": err, "backend": backend.name}


def post_model_unload(req):
    mid, backend = _backend_for(req)
    ok, err = backend.unload(mid)
    return (200 if ok else 400), {"ok": ok, "error": err, "backend": backend.name}


def post_model_save(req):
    mid, backend = _backend_for(req)
    out = backend.save(mid, _clean_settings(req.body.get("settings", {})))
    return 200, {"ok": True, "backend": backend.name, **out}


def post_model_delete(req):
    mid, backend = _backend_for(req)
    try:
        ok, err = backend.delete(mid)
    except backends.Unsupported as e:
        raise ApiError(400, str(e))
    if ok:
        if backend.name in config.PRESET_ENGINES:
            _reconcile_preset_binding(mid, {}, backend.name)
            config.prune_binding(mid, backend.name)
    return (200 if ok else 500), {"ok": ok, "error": err, "backend": backend.name}


# ---- llama.cpp aliases (kept: this is what the dashboard calls today) -------

def post_save(req):
    mid = req.body.get("model")
    clean = _clean_settings(req.body.get("settings", {}))
    running = _apply_knobs_and_reload(mid, clean)
    return 200, {"ok": True, "was_running": running}


def post_load(req):
    code, res = router("/models/load", "POST", {"model": req.body.get("model")})
    return (200 if code == 200 else 400), res


def post_unload(req):
    code, res = router("/models/unload", "POST", {"model": req.body.get("model")})
    return (200 if code == 200 else 400), res


def post_unload_all(req):
    st, data = router("/models")
    loaded = [m["id"] for m in data.get("data", [])
              if st == 200 and m.get("id") != "default"
              and m.get("status", {}).get("value") in ("loaded", "loading")]
    for mid in loaded:
        router("/models/unload", "POST", {"model": mid})
    return 200, {"ok": True, "unloaded": loaded}


def post_autotune_recommend(req):
    return 200, _autotune_recommend(req.body)


def post_autotune_refine(req):
    return 200, _autotune_refine(req.body)


def post_presets_save(req):
    name = req.body.get("name", "")
    try:
        presets = config.save_preset(name, req.body.get("settings", {}))
    except ValueError as e:
        raise ApiError(400, str(e))
    # Re-sync every model bound to this preset: editing "coding" once updates
    # all models using it - the point of binding (issue #2).
    clean = _clean_settings(presets.get(name, {}))
    for engine in config.PRESET_ENGINES:
        for mid in config.bindings_for_preset(name, engine):
            _reconcile_preset_binding(mid, clean, engine)
    return 200, {"ok": True, "presets": presets}


def post_presets_delete(req):
    name = req.body.get("name", "")
    if name in config.get_presets():
        for engine in config.PRESET_ENGINES:
            for mid in config.bindings_for_preset(name, engine):
                _reconcile_preset_binding(mid, {}, engine)
    return 200, {"ok": config.delete_preset(name)}


def post_presets_bind(req):
    """Bind a preset as a model's default (name="" unbinds). Binding
    materializes only absent knobs; unbinding removes only unchanged values
    that LlamaForge previously materialized."""
    mid = (req.body.get("model", "") or "").strip()
    name = (req.body.get("name", "") or "").strip()
    engine = cfg().get("active_engine", "llamacpp")
    if not mid:
        raise ApiError(400, "model id is required")
    try:
        if name:
            preset = config.get_presets().get(name)
            if preset is None:
                raise ValueError(f"unknown preset: {name}")
            _reconcile_preset_binding(mid, preset, engine)
        else:
            _reconcile_preset_binding(mid, {}, engine)
        binds = config.bind_preset(mid, name, engine)
    except ValueError as e:
        raise ApiError(400, str(e))
    return 200, {"ok": True, "bindings": binds}


def post_presets_apply(req):
    mid = req.body.get("model", "")
    name = req.body.get("name", "")
    preset = config.get_presets().get(name)
    if preset is None:
        raise ApiError(400, f"unknown preset: {name}")
    # apply exactly like /api/save so a loaded model reloads with the knobs
    clean = _clean_settings(preset)
    running = _apply_knobs_and_reload(mid, clean)
    return 200, {"ok": True, "applied": list(clean), "was_running": running}


def post_build_start(req):
    c = cfg()
    target = req.body.get("target", "llamacpp")
    builder = _builder_for(target)
    if target == "ikllama":
        src = c.get("ik_llama_src", "")
        bdir = c.get("ik_llama_build_dir", "")
        flags = req.body.get("flags") or c.get("ik_llama_cmake_flags") or hardware.recommend()["cmake_flags"]
        config.update({"ik_llama_cmake_flags": flags})
    else:
        src = c["llama_src"]
        bdir = c["build_dir"]
        flags = req.body.get("flags") or c.get("cmake_flags") or hardware.recommend()["cmake_flags"]
        config.update({"cmake_flags": flags})
    # Answer an unset/bad path here rather than starting a build thread that can
    # only fail: the user gets the reason in the UI instead of a raw cmake error
    # in the build log ("No build directory specified for -B").
    bad = BuildManager.validate_paths(src, bdir)
    if bad:
        return 200, {"started": False, "target": target, "error": bad}
    ok = builder.start(src, bdir, flags, pull=req.body.get("pull", True))
    return 200, {"started": ok, "target": target}


def post_setup_install(req):
    ok, log = prereqs.install(req.body.get("tool", ""))
    return 200, {"ok": ok, "log": log}


def post_scan(req):
    roots = req.body.get("roots") or cfg().get("model_dirs") or None
    return 200, {"entries": scanner.scan(roots)}


def post_scan_apply(req):
    entries = req.body.get("entries", [])
    existing = config.read_sections()
    for e in entries:
        keys = {"model": e["model"]}
        # Always pass mmproj/embeddings so stale values are cleared on re-scan.
        keys["mmproj"] = e.get("mmproj") or None
        keys["embeddings"] = "true" if e.get("embeddings") else None
        # MTP wiring is ADDITIVE, unlike mmproj: spec-type is also how the user
        # selects ngram-* speculation, so clearing it on re-scan would wipe a
        # hand-set mode. Only fill these when the section doesn't already carry
        # its own value.
        sect = existing.get(e["id"], {})
        if e.get("draft_model") and not sect.get("spec-draft-model"):
            keys["spec-draft-model"] = e["draft_model"]
        if e.get("draft_mtp") and not sect.get("spec-type"):
            keys["spec-type"] = "draft-mtp"
        config.set_keys(e["id"], keys)
    config.apply_ctx_defaults()
    router("/models?reload=1")
    return 200, {"ok": True, "added": len(entries)}


def post_scan_prune(req):
    ids, removed = req.body.get("ids", []), []
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
            engine = cfg().get("active_engine", "llamacpp")
            _reconcile_preset_binding(mid, {}, engine)
            config.prune_binding(mid, engine)
    if removed:
        router("/models?reload=1")
    return 200, {"removed": removed}


def post_hub_search(req):
    try:
        res = hub.search(req.body.get("query", ""), req.body.get("sort", "downloads"))
        inst = installed_repos(res, config.read_sections(),
                               vllm_registry.models() if VLLM_SUPPORTED else [])
        return 200, {"results": res, "vram_mib": total_vram_mib(), "installed": inst}
    except Exception as e:
        return 200, {"error": str(e), "results": []}


def post_hub_files(req):
    try:
        repo = req.body.get("repo", "")
        listing = hub.files(repo, total_vram_mib())
        c = cfg()
        if c.get("vram_predict_enabled", True):
            hw = vram_predict.build_hardware(c)
            for f in listing.get("files", []):
                f["predict"] = vram_predict.predict_remote(
                    repo=repo, gguf_file=f.get("path"), size_bytes=f.get("size"),
                    cfg=c, hw=hw)
                # Prefer the offload-aware label over hub._fit()'s size-only
                # guess; keep the naive value only when physics can't decide.
                label = vram_predict.fit_label(f["predict"])
                if label != "unknown":
                    f["fit"] = label
        return 200, listing
    except Exception as e:
        return 200, {"error": str(e), "files": [], "mmproj": []}


def post_vram_predict(req):
    """Standalone 'will it run?' estimate for a repo + quant (Discover-independent).
    For GGUF repos whose config.json lacks geometry, fall back to the matching (or
    largest) GGUF file's size so the estimate still resolves instead of going unknown."""
    b = req.body or {}
    repo = b.get("repo", "")
    if not repo:
        return 200, {"error": "repo is required"}
    quant = b.get("quant", "q4_k_m")
    gguf_file = b.get("gguf_file")
    size_bytes = None
    try:
        ggufs = hub.files(repo, 0).get("files", [])
        if ggufs:
            qkey = quant.replace("_", "").replace("-", "").lower()
            match = next((f for f in ggufs
                          if qkey in f.get("path", "").replace("_", "").replace("-", "").lower()),
                         None)
            chosen = match or max(ggufs, key=lambda f: f.get("size", 0))
            size_bytes = chosen.get("size")
            gguf_file = gguf_file or chosen.get("path")
    except Exception:
        pass
    out = vram_predict.predict_remote(repo=repo, quant=quant, gguf_file=gguf_file,
                                      size_bytes=size_bytes, cfg=cfg())
    return 200, out


def post_hub_download(req):
    repo   = req.body.get("repo", "")
    first  = req.body.get("path", "")
    shards = int(req.body.get("shards", 1))
    paths  = hub.shard_paths(first, shards)
    if req.body.get("mmproj"):
        paths.append(req.body["mmproj"])
    dest = os.path.join(download_dir(), repo.replace("/", "--"))
    ok = DOWNLOADS.start(repo, paths, dest)
    return 200, {"started": ok, "dest": dest}


def post_hub_cancel(req):
    return 200, {"ok": DOWNLOADS.cancel()}


def post_hub_pause(req):
    return 200, {"ok": DOWNLOADS.pause()}


def post_hub_resume(req):
    return 200, {"ok": DOWNLOADS.resume()}


def post_hub_add(req):
    """Register a finished download in models.ini."""
    path = req.body.get("path", "")
    if not path or not os.path.exists(path):
        raise ApiError(400, "file not found")
    folder = os.path.dirname(path)
    entries = _register_ggufs_beside(
        [os.path.join(folder, f) for f in os.listdir(folder)
         if f.lower().endswith(".gguf")])
    return 200, {"ok": True, "added": [e["id"] for e in entries]}


def post_stats_reset(req):
    stats.TRACKER.reset()
    return 200, {"ok": True}


# Keys the dashboard may set through /api/config, with a validator each.
#
# This is an allowlist rather than a blanket `cfg.update(body)` because the
# panel is reachable by any page in the user's browser: an unfiltered merge let
# a request set `server_bin`, which argspec then executes as `<server_bin>
# --help` on the next /api/schema. Paths that name a program or a directory the
# backend reads (server_bin, llama_src, build_dir, models_ini, wiki_dir,
# docs_dir) are deliberately absent - those belong to bootstrap and config.json,
# not to the browser. Keys with their own route (presets, cmake_flags,
# router_host/router_api_key) are absent for the same reason: those routes carry
# extra behaviour this one must not bypass.
def _v_bool(v):  return bool(v) if isinstance(v, bool) else None
def _v_str(v):   return v if isinstance(v, str) else None
def _v_port(v):  return v if isinstance(v, int) and 1 <= v <= 65535 else None
def _v_mode(v):  return v if v in ("lite", "advanced") else None
def _v_theme(v): return v if v in ("", "light", "dark") else None
def _v_dirs(v):
    return v if isinstance(v, list) and all(isinstance(x, str) for x in v) else None

def _v_bandwidths(v):
    """{vram_bw,ram_bw,disk_bw} -> GB/s. Only those keys, each a positive number.
    An empty dict is valid and clears all overrides (back to presets/defaults)."""
    if not isinstance(v, dict):
        return None
    out = {}
    for k in ("vram_bw", "ram_bw", "disk_bw"):
        if k in v and v[k] is not None:
            n = v[k]
            if isinstance(n, bool) or not isinstance(n, (int, float)) or n <= 0:
                return None
            out[k] = float(n)
    return out


CONFIG_WRITABLE = {
    "ui_mode":                 _v_mode,
    "theme":                   _v_theme,
    "cvd":                     _v_bool,
    "onboarded":               _v_bool,
    "auto_load_model":         _v_str,
    "wsl_distro":              _v_str,
    "vllm_port":               _v_port,
    "model_dirs":              _v_dirs,
    "anthropic_default_model": _v_str,
    "anthropic_shim_enabled":  _v_bool,
    "vram_bandwidths":         _v_bandwidths,
    "vram_predict_enabled":    _v_bool,
}


def post_config(req):
    """Update user-facing settings. Unknown or ill-typed keys are refused and
    named in the response rather than silently dropped, so a UI change that
    needs a new key fails loudly instead of appearing to work."""
    accepted, rejected = {}, []
    for k, v in (req.body or {}).items():
        validator = CONFIG_WRITABLE.get(k)
        if validator is None:
            rejected.append(k)
            continue
        checked = validator(v)
        if checked is None:
            rejected.append(k)
            continue
        accepted[k] = checked
    if rejected and not accepted:
        raise ApiError(400, "not settable via /api/config: " + ", ".join(sorted(rejected)))
    c = config.update(accepted)
    out = {"ok": True, "config": _public_config(c), "applied": sorted(accepted)}
    if rejected:
        out["rejected"] = sorted(rejected)
    return 200, out


def _active_server_bin(c=None):
    """Return the server binary path for the currently active engine."""
    c = c or cfg()
    if c.get("active_engine") == "ikllama":
        return c.get("ik_llama_server_bin", "")
    return c.get("server_bin", "")


def _record_server_bin(key, path):
    """Point `key` at the binary a finished build produced. Returns True if
    config.json changed.

    Only fills a gap or repairs a path that isn't there: bootstrap's pre-build
    guess (`bin/llama-server`) never exists on MSVC, so it gets corrected, while
    a path the user set deliberately - and that resolves - is left alone. This
    runs on a build thread, so it goes through config.update()'s lock rather
    than load/mutate/save.
    """
    current = (cfg().get(key) or "").strip()
    if current and os.path.exists(current):
        return False
    config.update({key: path})
    return True


def post_network(req):
    c = cfg()
    host = req.body.get("host", "127.0.0.1")
    api_key = req.body.get("api_key")
    if api_key is None:
        api_key = c.get("router_api_key", "")   # field left blank -> keep existing key
    c = config.update({"router_host": host, "router_api_key": api_key})
    sbin = _active_server_bin(c)
    ini = config.ini_path()
    ok, err = router_ctl.restart(sbin, ini, c["router_port"],
                                 host, api_key, LOGDIR)
    return (200 if ok else 500), {"ok": ok, "error": err, "host": host}


def post_engine_switch(req):
    """Switch the active engine (llamacpp / ikllama) and restart the router.

    Validate the binary BEFORE persisting. `active_engine` steers ini_path(),
    schema() and the model list, so writing it for an engine that cannot start
    leaves the panel pointed at nothing - and the failure only shows up later,
    somewhere else."""
    engine = req.body.get("engine", "llamacpp")
    if engine not in backends.LLAMA_FAMILY:
        raise ApiError(400, f"unknown engine: {engine}")
    c = cfg()
    current = c.get("active_engine", "llamacpp")
    sbin = _active_server_bin(dict(c, active_engine=engine))
    if not sbin or not os.path.exists(sbin):
        return 200, {"ok": False, "active_engine": current,
                     "error": f"binary not found: {sbin or '(unset)'} — build {engine} first"}
    if not router_ctl.supports_router_mode(sbin):
        # ik_llama.cpp forked before router mode; it rejects --models-preset and
        # serves one model per process. Switching anyway would kill the router.
        return 200, {"ok": False, "active_engine": current,
                     "error": f"{engine} has no router mode (its llama-server rejects "
                              f"--models-preset), so LlamaForge cannot drive it as the "
                              f"router. Staying on {current}."}
    c = config.update({"active_engine": engine})
    ok, err = router_ctl.restart(sbin, config.ini_path(), c["router_port"],
                                 c.get("router_host", "127.0.0.1"),
                                 c.get("router_api_key", ""), LOGDIR)
    return 200, {"ok": ok, "active_engine": engine, "error": err}


def post_vllm_load(req):
    mid = req.body.get("model", "")
    ok, err = REGISTRY.get("vllm").load(mid)
    return (200 if ok else 400), {"ok": ok, "error": err}


def post_vllm_unload(req):
    REGISTRY.get("vllm").unload(req.body.get("model", ""))
    return 200, {"ok": True}


def post_vllm_setup_install(req):
    c = cfg()
    distro = req.body.get("distro") or c.get("wsl_distro") or wsl.default_distro()
    if req.body.get("distro"):
        config.update({"wsl_distro": req.body["distro"]})
    ok = VLLM_SETUP_JOB.start(vllm_setup.install_script(), distro)
    return 200, {"started": ok}


def post_vllm_save(req):
    out = REGISTRY.get("vllm").save(req.body.get("model", ""),
                                    req.body.get("settings", {}))
    return 200, {"ok": True, "restarted": out["restarted"]}


def post_vllm_update(req):
    c = cfg()
    distro = c.get("wsl_distro") or wsl.default_distro()
    ok = VLLM_SETUP_JOB.start(vllm_setup.update_script(), distro)
    return 200, {"started": ok}


def post_vllm_hub_search(req):
    try:
        res = vllm_hub.search(req.body.get("query", ""), req.body.get("sort", "downloads"))
        inst = installed_repos(res, {}, vllm_registry.models())
        return 200, {"results": res, "vram_mib": total_vram_mib(), "installed": inst}
    except Exception as e:
        return 200, {"error": str(e), "results": []}


def post_vllm_hub_info(req):
    try:
        return 200, vllm_hub.repo_info(req.body.get("repo", ""), total_vram_mib())
    except Exception as e:
        return 200, {"error": str(e)}


def post_vllm_hub_download(req):
    repo = req.body.get("repo", "")
    info = {}
    try:
        info = vllm_hub.repo_info(repo, total_vram_mib())
    except Exception:
        pass
    ok = vllm_dl().start(repo, int(req.body.get("size_bytes") or info.get("size_bytes") or 0))
    return 200, {"started": ok}


def post_vllm_hub_register(req):
    repo = req.body.get("repo", "")
    try:
        wsl_path = vllm_dl().wsl_path(repo)
    except ValueError as e:
        raise ApiError(400, str(e))
    vllm_registry.upsert(repo, {
        "repo": repo, "wsl_path": wsl_path,
        "size_bytes": int(req.body.get("size_bytes") or 0),
        "quant": req.body.get("quant", "")})
    return 200, {"ok": True, "added": repo}


def post_vllm_delete(req):
    ok, err = REGISTRY.get("vllm").delete(req.body.get("model", ""))
    return (200 if ok else 500), {"ok": ok, "error": err}


def post_count_tokens(req):
    if not cfg().get("anthropic_shim_enabled", True):
        raise ApiError(404, "not found")
    if not _shim_auth_ok(req.headers):
        st, err = anthropic_shim.anthropic_error(401, "authentication_error",
                                                 "invalid x-api-key")
        return st, err
    return 200, {"input_tokens": anthropic_shim.count_tokens_estimate(req.body)}


def post_agent_apply(req):
    agent = req.body.get("agent", "")
    model = req.body.get("model", "")
    small = req.body.get("small") or None
    try:
        out = agentsetup.apply(agent, os.path.expanduser("~"),
                               _agent_endpoint(agent),
                               cfg().get("router_api_key", ""), model, small)
    except ValueError as e:
        raise ApiError(400, str(e))
    return 200, out


def post_wiki_doc(req):
    try:
        wiki.write_doc(req.body.get("name", ""), req.body.get("text", ""))
    except ValueError as e:
        raise ApiError(400, str(e))
    return 200, {"ok": True, "docs": wiki.list_docs()}


def post_wiki_doc_delete(req):
    try:
        ok = wiki.delete_doc(req.body.get("name", ""))
    except ValueError as e:
        raise ApiError(400, str(e))
    return 200, {"ok": ok, "docs": wiki.list_docs()}


def post_wiki_profile(req):
    try:
        profs = wiki.save_profile(req.body.get("name", ""), req.body.get("docs", []),
                                  req.body.get("description", ""))
    except ValueError as e:
        raise ApiError(400, str(e))
    return 200, {"ok": True, "profiles": profs}


def post_wiki_profile_delete(req):
    return 200, {"ok": wiki.delete_profile(req.body.get("name", "")),
                 "profiles": wiki.get_profiles()}


def post_wiki_active(req):
    wiki.set_active(req.body.get("model", ""), req.body.get("profile", ""))
    return 200, {"ok": True}


def post_wiki_export(req):
    out = _wiki_export(req.body)
    return (400 if out.get("error") else 200), out


# =================================================================== the tables

GET_ROUTES = {
    "/api/state":             get_state,
    "/api/schema":            get_schema,
    "/api/gpus":              get_gpus,
    "/api/setup":             get_setup,
    "/api/build/info":        get_build_info,
    "/api/build/log":         get_build_log,
    "/api/hub/progress":      get_hub_progress,
    "/api/router/log":        get_router_log,
    "/api/stats":             get_stats,
    "/api/scan/missing":      get_scan_missing,
    "/api/network":           get_network,
    "/api/vllm/log":          get_vllm_log,
    "/api/vllm/setup":        get_vllm_setup,
    "/api/vllm/schema":       get_vllm_schema,
    "/api/vllm/version":      get_vllm_version,
    "/api/vllm/hub/progress": get_vllm_hub_progress,
    "/api/model/metadata":    get_model_metadata,
    "/api/model/diag":        get_model_diag,
    "/api/presets":           get_presets,
    "/api/agent/config":      get_agent_config,
    "/api/wiki/docs":         get_wiki_docs,
    "/api/wiki/doc":          get_wiki_doc,
    "/api/wiki/profiles":     get_wiki_profiles,
    "/api/wiki/preview":      get_wiki_preview,
    "/api/docs":              get_docs,
    "/api/docs/page":         get_docs_page,
}

POST_ROUTES = {
    # engine-agnostic (dispatch on the model's backend)
    "/api/models/load":         post_model_load,
    "/api/models/unload":       post_model_unload,
    "/api/models/save":         post_model_save,
    "/api/models/delete":       post_model_delete,
    # llama.cpp-specific aliases
    "/api/save":                post_save,
    "/api/load":                post_load,
    "/api/unload":              post_unload,
    "/api/unload_all":          post_unload_all,
    "/api/autotune/recommend":  post_autotune_recommend,
    "/api/autotune/refine":     post_autotune_refine,
    "/api/presets/save":        post_presets_save,
    "/api/presets/bind":        post_presets_bind,
    "/api/presets/delete":      post_presets_delete,
    "/api/presets/apply":       post_presets_apply,
    "/api/build/start":         post_build_start,
    "/api/setup/install":       post_setup_install,
    "/api/scan":                post_scan,
    "/api/scan/apply":          post_scan_apply,
    "/api/scan/prune":          post_scan_prune,
    "/api/hub/search":          post_hub_search,
    "/api/hub/files":           post_hub_files,
    "/api/vram/predict":        post_vram_predict,
    "/api/hub/download":        post_hub_download,
    "/api/hub/cancel":          post_hub_cancel,
    "/api/hub/pause":           post_hub_pause,
    "/api/hub/resume":          post_hub_resume,
    "/api/hub/add":             post_hub_add,
    "/api/stats/reset":         post_stats_reset,
    "/api/config":              post_config,
    "/api/network":             post_network,
    "/api/engine/switch":       post_engine_switch,
    "/api/vllm/load":           post_vllm_load,
    "/api/vllm/unload":         post_vllm_unload,
    "/api/vllm/setup/install":  post_vllm_setup_install,
    "/api/vllm/save":           post_vllm_save,
    "/api/vllm/update":         post_vllm_update,
    "/api/vllm/hub/search":     post_vllm_hub_search,
    "/api/vllm/hub/info":       post_vllm_hub_info,
    "/api/vllm/hub/download":   post_vllm_hub_download,
    "/api/vllm/hub/register":   post_vllm_hub_register,
    "/api/vllm/delete":         post_vllm_delete,
    "/v1/messages/count_tokens": post_count_tokens,
    "/api/agent/apply":         post_agent_apply,
    "/api/wiki/doc":            post_wiki_doc,
    "/api/wiki/doc/delete":     post_wiki_doc_delete,
    "/api/wiki/profile":        post_wiki_profile,
    "/api/wiki/profile/delete": post_wiki_profile_delete,
    "/api/wiki/active":         post_wiki_active,
    "/api/wiki/export":         post_wiki_export,
}


# Built last: backends.Registry captures this module as its dependency bundle,
# so every helper it reaches for must already be defined.
REGISTRY = backends.Registry(sys.modules[__name__])
