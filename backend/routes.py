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
import json, os, re, subprocess, sys, threading, time, urllib.request, urllib.error, urllib.parse

import config, argspec, hardware, osplat, prereqs, scanner, hub, router_ctl, stats, telemetry
import autotune, anthropic_shim, agentsetup, clientsetup, network_policy, wiki, docs
import feed, selfupdate, appinstall, profiles, recipes, gallery, starters
import vram_predict
import wsl, vllm_ctl, vllm_registry, vllm_setup, vllm_job, vllm_hub, vllm_download
import gguf, diag, backends, prebuilt, version, slots, slotctl, slotproc, builds, compat
import mcp_server, piinstall, tts, logfiles, loadprog, gamemode, footprint, elastic, slotkeep
import claudelaunch
from builder import BuildManager
from embers import panel as embers_panel
import forge_panel

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
# Its own manager: the shared one registers finished files into models.ini,
# which would offer the speech model to the router as a chat model.
TTS_DOWNLOADS = hub.DownloadManager()
TTS = tts.Speaker()
APP_UPDATE = selfupdate.UpdateJob(ROOT)

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

# Saving router-affecting config and restarting the process is one transaction.
# ThreadingHTTPServer may run network and engine mutations concurrently; without
# this boundary, the file and the live router can end up describing different
# settings.  RLock keeps it safe for future lifecycle helpers to compose.
_ROUTER_LIFECYCLE_LOCK = threading.RLock()


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
    return logfiles.tail_lines(path, n)   # never the whole file (issue #26)


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


def vram_snapshot_mib():
    """(total, used) VRAM across GPUs from one nvidia-smi call.  `used` counts
    everything on the cards right now: other apps and any loaded model."""
    gpus = [g for g in _gpu_telemetry() if "total" in g]
    return sum(g["total"] for g in gpus), sum(g.get("used", 0) for g in gpus)


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


def _agent_endpoint_for(agent, inject, c=None):
    c = c or cfg()
    if agent == "claude-code":
        return f"http://127.0.0.1:{c['panel_port']}"
    if inject:
        return f"http://127.0.0.1:{c['panel_port']}/v1"
    return _llama_client_endpoint(c) + "/v1"


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
    c = cfg()
    url = f"http://127.0.0.1:{c['router_port']}" + path
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    key = network_policy.effective_key(c)
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url, data=data, method=method,
                                 headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        try: return e.code, json.loads(e.read().decode())
        except Exception: return e.code, {"error": str(e)}
    except Exception as e:
        return 599, {"error": str(e)}


def reload_presets():
    """Make the router re-read models.ini; (status, body) of the request that
    did it. Builds disagree on the route: current llama.cpp reloads on
    GET /models?reload=1 and has no POST /models/reload (404), while builds
    that keep GET /models read-only reload on the POST and answer the query
    form with a plain listing. Try the POST, fall back to the query form when
    the route isn't there, so every build reloads exactly once."""
    status, out = router("/models/reload", "POST", {})
    if status in (404, 405):
        status, out = router("/models?reload=1")
    if status == 599:
        out = {"error": f"could not reach the router to reload models.ini: {out.get('error')}"}
    return status, out


def _sse_target():
    c = cfg()
    key = network_policy.effective_key(c)
    return (f"http://127.0.0.1:{c['router_port']}/models/sse",
            {"Authorization": "Bearer " + key} if key else {})


LOADPROG = loadprog.LoadProgress(_sse_target)


def gpus():
    return hardware.detect_gpus_verbose() if hasattr(hardware, "detect_gpus_verbose") else _gpu_telemetry()


def _gpu_telemetry():
    if osplat.IS_MAC:
        return osplat.mac_gpu_telemetry()
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"], text=True, timeout=8,
            creationflags=router_ctl.CREATE_NO_WINDOW if osplat.IS_WIN else 0)
    except Exception as e:
        return [{"error": str(e)}]
    def num(v):
        try:
            return int(v)
        except ValueError:            # "[N/A]" on cards that don't report it
            return None
    res = []
    for ln in out.strip().splitlines():
        f = [x.strip() for x in ln.split(",")]
        if len(f) >= 6:
            index, used, total = num(f[0]), num(f[2]), num(f[3])
            if None in (index, used, total):
                continue              # a GPU without memory figures tells us nothing
            res.append({"index": index, "name": f[1], "used": used,
                        "total": total, "util": num(f[4]), "temp": num(f[5])})
    return res


# nvidia-smi startup is most of /api/state's response time (and is slower under
# WSL2).  A ten-second snapshot keeps the dashboard live without launching a
# process for every browser poll or every open tab.
_GPU_TELEMETRY = telemetry.TimedCache(_gpu_telemetry, ttl=10)


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
    _overlay_procs(models)
    parked = config.vision_parked()
    for m in models:
        if m["id"] in parked:
            m["vision_off"] = True    # projector parked; the editor offers it back
    LOADPROG.attach(models)
    models.sort(key=lambda m: (m["status"] != "loaded", m["id"]))
    return {"models": models, "global": glob}


def _overlay_procs(models):
    """A model pinned to another build runs in its own process: the router
    lists it as unloaded, so its row takes the process's state and port."""
    procs = PROCS.status()
    pins = cfg().get("model_builds") or {}
    for m in models:
        p = procs.get(m["id"])
        if p:
            s = slotctl.proc_status(p)
            m.update(status=s["value"], failed=bool(s.get("failed")), endpoint=p["endpoint"],
                     process={"port": p["port"], "pid": p["pid"], "exit_code": p.get("exit_code")})
        if isinstance(pins, dict) and pins.get(m["id"]):
            m["build"] = pins[m["id"]]


def _file_gib(path):
    """Model file size in GiB, or None (missing path / file gone)."""
    try:
        return round(gguf.total_size(path) / 1024**3, 2) if path else None
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
        size = gguf.total_size(path)
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

    was_running = (_model_status(mid) or {}).get("value") in slotctl.RUNNING
    main = SLOTS._main()
    # beside another main it's tuned where it would run: as a worker
    role = "main" if main == mid or not (main or was_running) else "worker"
    before = config.read_sections(raw=True).get(mid, {})
    touched = set()

    def load_fn(knobs):
        clean = _clean_settings(knobs)              # blank = unset (left to --fit)
        touched.update(clean)
        _apply_knobs_and_reload(mid, clean)
        _load_and_wait(mid, role)

    out = autotune.refine(base, intent, load_fn, lambda: _measure_tok_s(mid))
    # benchmarking isn't saving: models.ini goes back to what the user wrote and
    # the model to how it was found; the UI offers the winner as unsaved changes
    if touched:
        _apply_knobs_and_reload(mid, {k: before.get(k) for k in touched})
        if was_running:
            try:
                _load_and_wait(mid, role)
            except RuntimeError as e:
                out["restore_error"] = str(e)
    out["model"] = mid
    return out


def _model_base(mid):
    """Where mid answers: its own process's port, or the router's."""
    return (PROCS.status().get(mid) or {}).get("endpoint") or router_base()


def _measure_tok_s(mid):
    """Send a real completion request and measure tok/s (generation only, excludes prompt eval)."""
    prompt = "Write a Python function that computes the Fibonacci sequence iteratively. Explain your approach briefly."
    payload = {"model": mid, "prompt": prompt, "n_predict": 200, "stream": True}
    url = _model_base(mid) + "/completion"
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    key = network_policy.effective_key(cfg())
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers=headers)
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
    except Exception:
        return 0.0
    if first_tok is None or last_tok is None or tokens < 10:
        return 0.0
    elapsed = last_tok - first_tok
    if elapsed < 0.01:
        return 0.0
    return round(tokens / elapsed, 1)


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
    key = network_policy.effective_key(cfg())
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
    /models/unload only signals the process, so wait for it to stop: then the
    reload never sees it running, whatever this router build does with a
    running model's changed preset. On a pool the slot manager unloads it, so
    the per-GPU books stay right. Returns whether it had been running."""
    config.set_keys(mid, clean)
    return _reload_preset(mid)


def _reload_preset(mid):
    """Make the router pick up mid's models.ini section as it is now: unload
    it first when it's running, then reload_presets(). Whether it had been running."""
    running = (_model_status(mid) or {}).get("value") in slotctl.RUNNING
    if running:
        _unload_and_wait(mid)
    reload_presets()
    return running


def _unload_and_wait(mid):
    """(status, body) once mid has stopped. On a pool, or for a model in its own
    process, the slot manager does it (it keeps the per-GPU books); otherwise
    /models/unload, which only signals the process, then a wait."""
    if _slots_on() or PROCS.has(mid):
        return SLOTS.unload(mid)
    code, out = router("/models/unload", "POST", {"model": mid})
    if code != 200:
        return code, {"ok": False, "error": _router_error(out, "unload failed")}
    _wait_status(mid, lambda s: s.get("value") not in slotctl.RUNNING + ("loading",),
                 slotctl.UNLOAD_TIMEOUT_S)
    return 200, {"ok": True}


_sleep = time.sleep


def _model_status(mid):
    """mid's status: its own process's if it has one, else the router's; {} if
    the router doesn't list it, None if it isn't answering."""
    p = PROCS.status().get(mid)
    if p:
        return slotctl.proc_status(p)
    st, data = router("/models")
    if st != 200 or not isinstance(data, dict):
        return None
    return next((m.get("status") or {} for m in data.get("data", [])
                 if m.get("id") == mid), {})


def _wait_status(mid, done, timeout):
    """Poll once a second until done(status); that status, or None on timeout."""
    for _ in range(timeout):
        s = _model_status(mid)
        if s is not None and done(s):
            return s
        _sleep(1)
    return None


def _router_error(res, default):
    e = res.get("error") if isinstance(res, dict) else None
    if isinstance(e, dict):
        e = e.get("message")
    return e or default


def _load_and_wait(mid, role):
    """Load mid and return once it serves; RuntimeError says why it didn't.
    /models/load only starts a load - a request sent before it finishes would
    measure a model that's still loading."""
    if _slotted(mid):
        status, out = SLOTS.load(mid, role, False, wait=True)
        if status != 200:
            raise RuntimeError(_slot_error(out) or "load failed")
        return
    _stop_procs()
    code, res = router("/models/load", "POST", {"model": mid})
    if code >= 400:
        raise RuntimeError(_router_error(res, "load failed"))
    s = _wait_status(mid, lambda s: s.get("failed") or s.get("value") in slotctl.RUNNING,
                     slotctl.LOAD_TIMEOUT_S)
    if s is None:
        raise RuntimeError(f"still loading after {slotctl.LOAD_TIMEOUT_S} s")
    if s.get("failed"):
        raise RuntimeError(f"the router reports the load failed (exit code "
                           f"{s.get('exit_code', '?')}); see the router log")


def _clean_settings(updates):
    """Normalize a knob map from the UI: blank means 'unset this key'."""
    clean = {}
    for k, v in (updates or {}).items():
        v = ("" if v is None else str(v)).strip()
        clean[k] = None if v == "" else v
    return clean


def _persist_scanned_entries(entries):
    """Persist scanner fields and reconcile only LlamaForge-owned MTP keys."""
    existing = config.read_sections()
    parked = config.vision_parked()
    for e in entries:
        mmproj = e.get("mmproj") or None
        if e["id"] in parked:                 # vision turned off: keep it off
            config.repark_vision(e["id"], mmproj)
            mmproj = None
        keys = {
            "model": e["model"],
            "mmproj": mmproj,
            "embeddings": "true" if e.get("embeddings") else None,
        }
        desired_mtp = {
            "spec-draft-model": e.get("draft_model"),
            "spec-type": "draft-mtp" if e.get("draft_mtp") else None,
        }
        keys.update(config.reconcile_mtp_autowire(
            e["id"], existing.get(e["id"]), desired_mtp))
        config.set_keys(e["id"], keys)
    return entries


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
    entries = _persist_scanned_entries(scanner.build_entries(paths))
    config.apply_ctx_defaults()
    reload_presets()
    return entries


# =============================================================== GET handlers

def get_state(req):
    c = cfg()
    s = REGISTRY.state()
    s["gpus"] = _GPU_TELEMETRY.get()
    s["ram"] = osplat.ram_usage()
    s["config"] = _public_config(c)
    s["platform"] = osplat.current()
    s["vllm_supported"] = VLLM_SUPPORTED
    s["backends"] = [b.name for b in REGISTRY.enabled()]
    s["active_engine"] = c.get("active_engine", "llamacpp")
    s["chat_port"] = c.get("chat_port", 8091)
    s["config_error"] = config.LOAD_ERROR
    s["version"] = version.VERSION
    s["slots"] = _slots_state(c)
    s["game_mode"] = GAMEMODE.status()
    s["elastic"] = ELASTIC.status()
    s["onboarding"] = {
        "server_bin_ok": bool(c.get("server_bin")) and os.path.exists(c["server_bin"]),
        "model_count": len(s["models"]),
        "ui_mode": c.get("ui_mode", "lite"),
        "onboarded": bool(c.get("onboarded", False)),
    }
    return 200, s


def _public_config(c):
    """Allowlisted browser state. Secret material is available only through
    deliberate client/agent/network POST actions."""
    return network_policy.public_config(c)


def get_schema(req):
    return 200, schema()


def get_gpus(req):
    return 200, {"gpus": _GPU_TELEMETRY.get(), "ram": osplat.ram_usage()}


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


def get_starters(req):
    vram = total_vram_mib()
    return 200, {"vram_mib": vram, "starters": starters.pick(vram)}


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


def _network_status(c, running=None):
    assessment = network_policy.assess(
        c.get("router_host", "127.0.0.1"),
        c.get("router_api_key", ""))
    if running is None:
        running = router_ctl.is_running(c["router_port"])
    out = assessment.public()
    out.update({
        "port": c["router_port"],
        "lan_ip": router_ctl.lan_ip(),
        "router_running": bool(running),
        "listener_status": "listening" if running else "not_listening",
    })
    return out


def get_network(req):
    return 200, _network_status(cfg())


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


def get_feed(req):
    """"New this week": model support llama.cpp just merged (marked against the
    running engine's build) and whether a newer LlamaForge release is out."""
    force = req.flag("force")
    out = {"engine_build": feed.engine_build(cfg().get("server_bin", ""))}
    try:
        out["engine_news"] = feed.engine_news(feed.llama_releases(force), out["engine_build"])
    except Exception as e:
        out["engine_news"], out["engine_error"] = [], str(e)
    installed = appinstall.installed_version(ROOT)
    try:
        out["app"] = feed.app_update(installed, feed.app_latest(force))
    except Exception as e:
        out["app"] = {"installed": installed, "available": False, "error": str(e)}
    out["app"]["managed"] = installed is not None
    out["app"]["job"] = APP_UPDATE.progress()
    return 200, out


def post_app_update(req):
    tag = (req.body or {}).get("tag", "")
    if not APP_UPDATE.start(tag):
        return 200, {"ok": False, "error": "an update is already running"}
    return 200, {"ok": True}


def post_app_restart(req):
    if APP_UPDATE.progress()["state"] != "done":
        return 200, {"ok": False, "error": "nothing to restart for"}
    selfupdate.restart(ROOT, osplat.current())
    return 200, {"ok": True}


def get_vllm_hub_progress(req):
    return 200, vllm_dl().progress()


def get_model_metadata(req):
    """The editor's GGUF card: header facts, and which llama.cpp family can
    load the file (compat.py), read when a row opens, never on the poll."""
    mid = req.q("model")
    sect = config.read_sections().get(mid, {})
    mpath = sect.get("model")
    meta = gguf.metadata(mpath) if mpath else None
    c = cfg()
    pins = c.get("model_builds") if isinstance(c.get("model_builds"), dict) else {}
    opts = builds.options(c, _installs(c))
    ok = compat.for_model(mpath) if mpath else dict(compat.UNKNOWN)
    # an ik older than the file crashes on it instead of refusing: ask its checkout
    ids = ok.get("ids") or []
    stale = compat.missing(ids, compat.build_types(c.get("ik_llama_src"))) if ids else []
    ok["advice"] = compat.advice(ok["class"], ok["types"], c.get("active_engine", "llamacpp"),
                                 build=pins.get(mid, ""),
                                 ik_built=any(o["ref"] == builds.IK for o in opts),
                                 ik_missing=stale)
    # a model family newer than the build: "unknown model architecture" at load
    ok["arch_advice"] = compat.arch_advice((meta or {}).get("architecture"),
                                           compat.build_archs(builds.source_dir(mid, c, _installs(c))))
    return 200, {"metadata": meta or {}, "compat": ok,
                 "builds": {"options": [{k: o[k] for k in ("ref", "label", "router")} for o in opts],
                            "pinned": pins.get(mid, "")}}


def get_model_diag(req):
    mid = req.q("model")
    ini = config.read_sections()
    merged = dict(ini.get("*", {}))
    merged.update(ini.get(mid, {}))
    # A model in its own process logs to its own file; on the router a load
    # prints ~50 lines of args before the child says anything, so the tail
    # must reach back past the spawn line of the last attempt.
    log = PROCS.log_tail(mid, 800) if PROCS.has(mid) else router_log_tail(800)
    return 200, {"diag": diag.diagnose(log, merged, model=mid)}


def get_presets(req):
    return 200, {"presets": config.get_presets()}


def _resolve_model_row(mid, hint=""):
    if not isinstance(mid, str) or not mid.strip():
        raise ApiError(400, "model is required")
    mid = mid.strip()
    if not isinstance(hint, str):
        raise ApiError(400, "backend must be a string")
    if hint in backends.LLAMA_FAMILY:
        hint = REGISTRY.active_engine()
    elif hint and hint != "vllm":
        raise ApiError(400, f"unknown backend: {hint}")

    rows = [row for row in REGISTRY.state().get("models", [])
            if row.get("id") == mid]
    if hint:
        rows = [row for row in rows if row.get("backend") == hint]
    if not rows:
        raise ApiError(400, f"unknown model/backend: {mid}/{hint or 'unspecified'}")
    if len(rows) != 1:
        raise ApiError(409, f"model ownership is ambiguous; supply backend for {mid}")
    return rows[0], rows[0].get("backend", "")


def _llama_client_endpoint(c):
    assessment = network_policy.assess(
        c.get("router_host", "127.0.0.1"), c.get("router_api_key", ""))
    if assessment.access_scope == "local":
        host = "127.0.0.1"
    elif assessment.access_scope == "lan":
        host = router_ctl.lan_ip()
        if not host:
            raise ApiError(503, "LAN IP is not available; use local-only or repair networking")
    else:
        raise ApiError(409, "repair the legacy Network Access configuration first")
    return f"http://{host}:{c['router_port']}"


def post_client_config(req):
    body = req.body or {}
    unknown = set(body) - {"model", "backend"}
    if unknown:
        raise ApiError(400, "unsupported client-config fields: " + ", ".join(sorted(unknown)))
    row, backend = _resolve_model_row(body.get("model"), body.get("backend", ""))
    c = cfg()
    if backend in backends.LLAMA_FAMILY and row.get("process"):
        endpoint = row["endpoint"]          # its own build's process: this machine only
        api_key = network_policy.effective_key(c)
    elif backend in backends.LLAMA_FAMILY:
        endpoint = _llama_client_endpoint(c)
        api_key = network_policy.effective_key(c)
    elif backend == "vllm":
        live = next((item for item in vllm_mgr().status()
                     if item.get("model_id") == row["id"]
                     and item.get("state") == "ready"), None)
        if not live or not live.get("endpoint"):
            raise ApiError(400, f"vLLM model {row['id']} is not ready; load it first")
        endpoint = live["endpoint"]
        api_key = ""
    else:
        raise ApiError(400, f"unsupported backend: {backend}")
    shell = "powershell" if osplat.IS_WIN else "posix"
    try:
        out = clientsetup.generate(endpoint, api_key, row["id"], backend, shell)
    except Exception:
        raise ApiError(500, "client configuration could not be generated") from None
    out["model_loaded"] = row.get("status") == "loaded"
    return 200, out


def _resolve_agent_request(body):
    if not isinstance(body, dict):
        raise ApiError(400, "agent configuration must be an object")
    allowed = {"agent", "model", "backend", "small", "inject"}
    unknown = set(body) - allowed
    if unknown:
        raise ApiError(400, "unsupported agent-config fields: " + ", ".join(sorted(unknown)))
    agent = body.get("agent", "")
    if not isinstance(agent, str):
        raise ApiError(400, "agent must be a string")
    if agent not in agentsetup.AGENTS:
        raise ApiError(400, f"unknown agent: {agent}")
    inject = body.get("inject")
    if not isinstance(inject, bool):
        raise ApiError(400, "inject must be a boolean")
    if agent == "claude-code" and inject:
        raise ApiError(400, "Claude Code always uses the local Anthropic endpoint")

    backend = body.get("backend", "")
    active = REGISTRY.active_engine()
    if backend != active or backend not in backends.LLAMA_FAMILY:
        raise ApiError(400, f"agent setup requires the active llama backend: {active}")
    row, resolved_backend = _resolve_model_row(body.get("model"), backend)
    if resolved_backend != active:
        raise ApiError(400, f"model is not owned by active backend: {row['id']}")

    small = body.get("small", "")
    if not isinstance(small, str):
        raise ApiError(400, "small must be a model id")
    small = small or None
    if small and agent != "claude-code":
        raise ApiError(400, "small is only supported for Claude Code")
    if small:
        small_row, small_backend = _resolve_model_row(small, backend)
        if small_backend != active:
            raise ApiError(400, f"small model is not owned by active backend: {small_row['id']}")
        small = small_row["id"]

    c = cfg()
    return {
        "agent": agent,
        "model": row["id"],
        "backend": active,
        "small": small,
        "inject": inject,
        "endpoint": _agent_endpoint_for(agent, inject, c),
        "api_key": network_policy.effective_key(c),
    }


def post_agent_config(req):
    target = _resolve_agent_request(req.body)
    try:
        out = agentsetup.generate(
            target["agent"], target["endpoint"], target["api_key"],
            target["model"], target["small"], target["inject"])
    except Exception:
        raise ApiError(500, "agent configuration could not be generated") from None
    return 200, out


# ---- one-click Claude Code on a folder (claudelaunch.py) ----

def _claude_cli(c):
    """The configured claude_bin if it exists, else the usual install spots."""
    override = c.get("claude_bin") or ""
    if override and os.path.isfile(override):
        return override
    return claudelaunch.locate()


def _claude_ctx(row, state):
    """Live n_ctx of a loaded model (never autoloads), else the configured ctx-size."""
    if row.get("status") == "loaded":
        q = urllib.parse.quote(row["id"], safe="")
        try:
            st, props = router(f"/props?model={q}&autoload=false", timeout=5)
            n = ((props or {}).get("default_generation_settings") or {}).get("n_ctx") \
                if st == 200 and isinstance(props, dict) else None
            if n:
                return int(n)
        except Exception:
            pass
    raw = (row.get("settings") or {}).get("ctx-size") \
        or (state.get("global") or {}).get("ctx-size")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _claude_assess(mid):
    """Resolve a model in the active llama backend and run the readiness checks."""
    active = REGISTRY.active_engine()
    if active not in backends.LLAMA_FAMILY:
        raise ApiError(400, f"Claude Code needs the active llama backend: {active}")
    row, backend = _resolve_model_row(mid, active)
    if backend != active:
        raise ApiError(400, f"model is not owned by active backend: {row['id']}")
    c = cfg()
    state = REGISTRY.state()
    cli = _claude_cli(c)
    small = claudelaunch.choose_small(
        [r for r in state.get("models", []) if r.get("backend") == active], row["id"])
    items, ready = claudelaunch.checks(
        cli, bool(c.get("anthropic_shim_enabled", True)), row.get("status") == "loaded",
        _claude_ctx(row, state), row.get("settings"), row["id"], small)
    return {"c": c, "row": row, "cli": cli, "small": small, "checks": items, "ready": ready}


def get_claude_check(req):
    a = _claude_assess(req.q("model"))
    return 200, {
        "model": a["row"]["id"],
        "cli": {"path": a["cli"], "version": claudelaunch.version(a["cli"]) if a["cli"] else ""},
        "small": a["small"],
        "checks": a["checks"],
        "ready": a["ready"],
        "recent": a["c"].get("claude_recent_dirs") or [],
        "install_hint": claudelaunch.INSTALL_HINT[claudelaunch.os_name()],
    }


def post_claude_pick_folder(req):
    argv = claudelaunch.pick_folder_argv(claudelaunch.os_name(), python=sys.executable)
    if not argv:
        return 200, {"path": "", "unsupported": True}
    return 200, {"path": claudelaunch.pick_folder(argv)}


def post_claude_launch(req):
    body = req.body
    if not isinstance(body, dict):
        raise ApiError(400, "launch request must be an object")
    unknown = set(body) - {"model", "folder"}
    if unknown:
        raise ApiError(400, "unsupported launch fields: " + ", ".join(sorted(unknown)))
    folder = claudelaunch.clean_folder(body.get("folder"))
    if not folder:
        raise ApiError(400, "folder is required")
    folder = os.path.abspath(os.path.expanduser(folder))
    if not os.path.isdir(folder):
        raise ApiError(400, f"not a folder: {folder}")
    a = _claude_assess(body.get("model"))
    if not a["ready"]:
        raise ApiError(409, "; ".join(i["text"] for i in a["checks"] if i["level"] == "error"))
    c = a["c"]
    try:
        plan = claudelaunch.launch(
            a["cli"], folder, dict(os.environ), _agent_endpoint_for("claude-code", False, c),
            network_policy.effective_key(c) or "llamaforge", a["row"]["id"], a["small"])
    except OSError as e:
        raise ApiError(500, f"couldn't open a terminal: {e.strerror or e}") from None
    config.mutate(lambda cc: cc.__setitem__(
        "claude_recent_dirs", claudelaunch.remember(cc.get("claude_recent_dirs"), folder)))
    return 200, {"launched": plan["kind"] != "manual", "command": plan["command"]}


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
    if backend.name == "llamacpp" and _slotted(mid):
        status, out = _slot_load(req.body, mid)
        return status, dict(out, error=_slot_error(out), backend=backend.name)
    if backend.name == "llamacpp":
        _stop_procs()
    ok, err = backend.load(mid)
    return (200 if ok else 400), {"ok": ok, "error": err, "backend": backend.name}


def post_model_unload(req):
    mid, backend = _backend_for(req)
    if backend.name == "llamacpp" and (_slots_on() or PROCS.has(mid)):
        status, out = SLOTS.unload(mid)
        return status, dict(out, error=out.get("error", ""), backend=backend.name)
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
            config.prune_vision(mid, backend.name)
    return (200 if ok else 500), {"ok": ok, "error": err, "backend": backend.name}


def post_model_unregister(req):
    """Remove a llama-family registry entry without deleting its GGUF file."""
    mid, backend = _backend_for(req)
    if not mid or mid == "*":
        raise ApiError(400, "a model id is required")
    if backend.name not in config.PRESET_ENGINES:
        raise ApiError(400, "vLLM models use Delete; unregister is for GGUF registries")
    path = config.ini_path(backend.name)
    if mid not in config.read_sections(path):
        raise ApiError(404, "model is not registered")

    status, data = router("/models")
    loaded = any(
        m.get("id") == mid and (m.get("status") or {}).get("value") == "loaded"
        for m in data.get("data", []) if status == 200
    )
    if loaded:
        unload_status, unload_out = router("/models/unload", "POST", {"model": mid})
        if unload_status != 200:
            detail = (unload_out or {}).get("error", "router refused to unload it")
            raise ApiError(409, f"model is still loaded: {detail}")
    if not config.remove_section(mid, path):
        raise ApiError(409, "model registry changed; reload and try again")
    _reconcile_preset_binding(mid, {}, backend.name)
    config.prune_binding(mid, backend.name)
    config.prune_vision(mid, backend.name)
    reload_presets()
    return 200, {"ok": True, "backend": backend.name}


# ---- llama.cpp aliases (kept: this is what the dashboard calls today) -------

def post_save(req):
    mid = req.body.get("model")
    clean = _clean_settings(req.body.get("settings", {}))
    running = _apply_knobs_and_reload(mid, clean)
    return 200, {"ok": True, "was_running": running}


def post_model_vision(req):
    """Vision on/off for one model; the projector file is never touched."""
    mid, on = req.body.get("model"), req.body.get("on")
    if not mid or not isinstance(mid, str) or mid == "*":
        raise ApiError(400, "a model id is required")
    if not isinstance(on, bool):
        raise ApiError(400, "on must be true or false")
    try:
        out = config.set_vision(mid, on)
    except ValueError as e:
        raise ApiError(400, str(e))
    running = _apply_knobs_and_reload(mid, {})
    return 200, {"ok": True, "vision": out["vision"], "was_running": running}


def post_load(req):
    if _slotted(req.body.get("model")):
        status, out = _slot_load(req.body, req.body.get("model"))
        # the shape the router answers in, which is what the dashboard reads
        res = dict(out, success=bool(out.get("ok")))
        if not out.get("ok"):
            res["error"] = {"message": _slot_error(out)}
        return status, res
    _stop_procs()
    code, res = router("/models/load", "POST", {"model": req.body.get("model")})
    return (200 if code == 200 else 400), res


def post_unload(req):
    if _slots_on() or PROCS.has(req.body.get("model")):
        status, out = SLOTS.unload(req.body.get("model"))
        res = dict(out, success=bool(out.get("ok")))
        if not out.get("ok"):
            res["error"] = {"message": out.get("error") or "unload failed"}
        return status, res
    code, res = router("/models/unload", "POST", {"model": req.body.get("model")})
    return (200 if code == 200 else 400), res


def _router_loaded_ids(states=("loaded", "loading", "sleeping")):
    st, data = router("/models")
    return [m["id"] for m in data.get("data", [])
            if st == 200 and m.get("id") != "default"
            and m.get("status", {}).get("value") in states]


def post_unload_all(req):
    loaded = _router_loaded_ids()
    slotted = _slots_on()
    for mid in loaded:
        if slotted:
            SLOTS.unload(mid)
        else:
            router("/models/unload", "POST", {"model": mid})
    procs = [m for m in PROCS.status() if m not in loaded]
    for mid in procs:
        SLOTS.unload(mid)
    return 200, {"ok": True, "unloaded": loaded + procs}


# ---- multi-model slots -------------------------------------------------------

def _slots_on(c=None):
    """Loads go through the planner only on a router LlamaForge started with a
    pool. The runner's router (or one started before the setting was turned on)
    holds one model and evicts by itself; plans for a pool that isn't running
    would be fiction. ik_llama has no router mode at all."""
    c = c or cfg()
    return (bool(c.get("multi_model")) and c.get("active_engine", "llamacpp") == "llamacpp"
            and router_ctl.running_pool(LOGDIR) is not None)


# ---- per-model builds (builds.py) and their processes (slotproc.py) ---------

def _installs(c=None):
    return prebuilt.list_installs(ENGINES_DIR, _active_server_bin(c))


def pinned_bin(mid):
    """(server_bin, "") when mid runs in its own process, (None, "") when the
    router serves it, (None, why) for a pin that can't be honoured. Lists the
    installs only for a model that has a pin: the dashboard poll asks this."""
    c = cfg()
    pins = c.get("model_builds")
    if not isinstance(pins, dict) or not pins.get(mid):
        return None, ""
    return builds.pinned_bin(mid, c, _installs(c))


def _slotted(mid):
    """Loads of mid go through the slot manager: on a pool, or when it runs on
    a build of its own (its own process, single-model mode too). A pin that
    can't be honoured goes there as well, to be refused - never to the router,
    whose build may be the reason for the pin."""
    if _slots_on():
        return True
    sbin, err = pinned_bin(mid) if mid else (None, "")
    return bool(sbin or err)


def _stop_procs():
    """Single-model mode: a router load replaces whatever runs in its own process."""
    for mid in list(PROCS.status()):
        SLOTS.unload(mid)


def post_model_build(req):
    """Pin a model to a build: {model, build}; build "" runs it on the router.
    A model that is up is unloaded, as a knob save does: it runs on the old
    build until it loads again."""
    body = req.body or {}
    mid, ref = body.get("model"), body.get("build", "")
    if not isinstance(mid, str) or not mid.strip() or mid.strip() == "*":
        raise ApiError(400, "a model id is required")
    if mid not in config.read_sections():
        raise ApiError(404, f"unknown model: {mid}")
    c = cfg()
    err = builds.validate(ref, c, _installs(c))
    if err:
        raise ApiError(400, err)
    pins = c.get("model_builds") if isinstance(c.get("model_builds"), dict) else {}
    if pins.get(mid, "") == ref:
        return 200, {"ok": True, "build": ref, "changed": False, "was_running": False}
    running = (_model_status(mid) or {}).get("value") in slotctl.RUNNING + ("loading",)
    if running:
        code, out = _unload_and_wait(mid)
        if code != 200:
            raise ApiError(409, f"{mid} is still up on its old build: "
                                f"{out.get('error') or 'unload failed'}")

    def apply(conf):
        p = conf.get("model_builds") if isinstance(conf.get("model_builds"), dict) else {}
        if ref:
            p[mid] = ref
        else:
            p.pop(mid, None)
        conf["model_builds"] = p
    config.mutate(apply)
    return 200, {"ok": True, "build": ref, "changed": True, "was_running": running}


def _slot_role(v):
    if v in (None, ""):
        return "main"
    if v not in ("main", "worker"):
        raise ApiError(400, "role must be main or worker")
    return v


def _slot_load(body, mid):
    if not mid:
        raise ApiError(400, "model is required")
    return SLOTS.load(mid, _slot_role(body.get("role")), bool(body.get("evict")), wait=True)


def _slot_error(out):
    if out.get("ok"):
        return ""
    return out.get("error") or out.get("reason") or "load refused"


def get_slots(req):
    c = cfg()
    on = _slots_on(c)
    return 200, {"enabled": on, "multi_model": bool(c.get("multi_model")),
                 "pool": router_ctl.running_pool(LOGDIR),
                 "restart_needed": bool(c.get("multi_model")) and not on,
                 "main": SLOTS._main(), "loaded": SLOTS.loaded() if on else []}


def _slots_state(c):
    """The slots part of /api/state: roles, where our loads went, and the
    settings. Built for the 4-second poll - config and two small files, never
    the planner or nvidia-smi."""
    on = _slots_on(c)
    llama = c.get("active_engine", "llamacpp") == "llamacpp"
    s = c.get("slots") if isinstance(c.get("slots"), dict) else {}
    main = s.get("main") if on and isinstance(s.get("main"), str) else ""
    cap = c.get("slot_cap")
    head = c.get("slot_headroom_mib")
    return {"enabled": on, "engine_ok": llama, "main": main,
            "restart_needed": bool(c.get("multi_model")) and llama and not on,
            "devices": SLOTS.devices() if on else {},
            "footprints": SLOTS.footprints() if on else {},
            "settings": {"multi_model": bool(c.get("multi_model")),
                         "slot_cap": cap if _v_int(slots.CAP_MIN, slots.CAP_MAX)(cap)
                         is not None else slots.CAP_DEFAULT,
                         "slot_headroom_mib": head if _v_int(0, 32768)(head) is not None
                         else slots.DEFAULT_HEADROOM_MIB,
                         "slot_autoload": bool(c.get("slot_autoload"))},
            "cap_range": [slots.CAP_MIN, slots.CAP_MAX]}


def post_slots_apply(req):
    """Restart the router with the pool the settings ask for (the one running
    was started single, by run.ps1 / run.sh or before the setting)."""
    with _ROUTER_LIFECYCLE_LOCK:
        restarted, err = _sync_router_pool(cfg())
    return (500 if err else 200), {"ok": not err, "restarted": restarted, "error": err or ""}


def get_slots_plan(req):
    mid = req.q("model")
    if not mid:
        raise ApiError(400, "model is required")
    return 200, SLOTS.plan(mid, _slot_role(req.q("role") or "worker"))


def get_mcp_setup(req):
    """Client configs for LlamaForge's MCP server (backend/mcp_server.py)."""
    info = mcp_server.setup_info()
    info["pi"] = bool(mcp_server.pirun.locate(cfg().get("pi_bin") or ""))
    return 200, info


def get_pi_status(req):
    return 200, piinstall.status(cfg().get("pi_bin") or "")


def _pi_job(action):
    if not piinstall.start(action):
        raise ApiError(409, piinstall.BUSY)
    return 200, {"started": True}


def post_pi_install(req):
    return _pi_job("install")


def post_pi_remove(req):
    return _pi_job("remove")


def post_slots_main(req):
    SLOTS.set_main(req.body.get("model") or "")
    return 200, {"ok": True, "main": req.body.get("model") or ""}


def post_autotune_recommend(req):
    return 200, _autotune_recommend(req.body)


def post_autotune_refine(req):
    return 200, _autotune_refine(req.body)


def post_presets_save(req):
    name = (req.body.get("name", "") or "").strip()
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


def post_profiles_save(req):
    try:
        profs = config.save_profile(req.body.get("name", ""), req.body.get("profile"))
    except ValueError as e:
        raise ApiError(400, str(e))
    return 200, {"ok": True, "profiles": profs}


def post_profiles_delete(req):
    return 200, {"ok": config.delete_profile(req.body.get("name", ""))}


def _wait_router(timeout=90):
    """After a restart the router takes a moment to bind; loads sent before
    that fail with connection refused."""
    end = time.time() + timeout
    while time.time() < end:
        if router("/models", timeout=5)[0] == 200:
            return True
        time.sleep(1)
    return False


def post_profiles_launch(req):
    """Switch to the profile's engine build if it pins one, apply its preset,
    then load its model. Each step reports which one failed."""
    name = req.body.get("name", "")
    prof = config.get_profiles().get(name)
    if prof is None:
        raise ApiError(404, f"unknown profile: {name}")
    try:
        p = profiles.plan(prof, prebuilt.list_installs(ENGINES_DIR, cfg().get("server_bin", "")),
                          config.get_presets())
    except ValueError as e:
        return 200, {"ok": False, "step": "plan", "error": str(e)}
    if prof.get("source") == "recipe" and p["settings"] is not None:
        # its preset may have been edited since the import - same gate again
        try:
            p["settings"] = recipes.clean(p["settings"], _recipe_knobs())[0]
        except ApiError as e:
            return 200, {"ok": False, "step": "plan", "error": e.message}
    if p["switch_bin"]:
        ok, err = _activate_prebuilt(p["switch_bin"])
        if not ok:
            return 200, {"ok": False, "step": "engine", "error": err}
        if not _wait_router():
            return 200, {"ok": False, "step": "engine",
                         "error": "the router didn't come back on the pinned build - see the router log"}
    if p["settings"] is not None:
        _apply_knobs_and_reload(p["model"], p["settings"])
    ok, err = REGISTRY.for_model(p["model"], p["backend"]).load(p["model"])
    return 200, {"ok": ok, "step": "load", "error": err,
                 "switched_engine": bool(p["switch_bin"]), "model": p["model"]}


def post_profiles_export(req):
    """A saved profile as a shareable recipe (see recipes.py)."""
    name = req.body.get("name", "")
    prof = config.get_profiles().get(name)
    if prof is None:
        raise ApiError(404, f"unknown profile: {name}")
    sections = config.read_sections()
    section = sections.get(prof["model"])
    if section is None:
        raise ApiError(400, f"{prof['model']} is no longer in models.ini")
    # what actually runs: the [*] defaults under the model's own section
    section = {**sections.get("*", {}), **section}
    try:
        recipe = recipes.export(name, prof, section, config.get_presets(),
                                prebuilt.list_installs(ENGINES_DIR, cfg().get("server_bin", "")),
                                _known_knobs())
    except ValueError as e:
        raise ApiError(400, str(e))
    return 200, {"ok": True, "recipe": recipe}


def get_recipes_gallery(req):
    """Community recipes (recipes/ in the repo), live from GitHub or bundled."""
    files, source = gallery.files(os.path.join(ROOT, "recipes"), force=req.flag("force"))
    return 200, {"source": source,
                 "recipes": gallery.entries(files, config.read_sections(), _known_knobs())}


def _known_knobs():
    """recipes.knob_index of the live schema, or None when there is none
    (router binary missing): then only canonical allowlisted names pass."""
    try:
        return recipes.knob_index(schema()) or None
    except Exception:
        return None


def _recipe_knobs():
    """The schema a recipe must be checked against before it touches this
    machine; without one an alias can't be told from an unknown flag."""
    known = _known_knobs()
    if known is None:
        raise ApiError(409, "recipes need the llama-server knob list - "
                            "set up a llama.cpp build first (Build / Update)")
    return known


def _start_recipe_download(model):
    """Fetch the recipe's GGUF (all shards) from its repo through the normal
    download manager; the UI imports again once it finishes."""
    if not model["hf_repo"]:
        return "this model isn't on this machine and the recipe doesn't say where it came from"
    try:
        listing = hub.files(model["hf_repo"])
    except Exception as e:
        return f"couldn't list {model['hf_repo']} on Hugging Face: {e}"
    want = model["file"].lower()
    f = next((f for f in listing.get("files", [])
              if os.path.basename(f["path"]).lower() == want), None)
    if f is None:
        return f"{model['file']} isn't in {model['hf_repo']} any more"
    paths = hub.shard_paths(f["path"], f.get("shards", 1))
    if f.get("mtp"):
        paths.append(f["mtp"])
    dest = os.path.join(download_dir(), model["hf_repo"].replace("/", "--"))
    if not DOWNLOADS.start(model["hf_repo"], paths, dest):
        return "another download is running - try again when it finishes"
    return ""


def post_profiles_import(req):
    """Turn a recipe into a preset + profile on this machine. When its model
    isn't here, {missing} (and with download=true, start fetching it)."""
    try:
        r = recipes.parse(req.body.get("recipe"), _recipe_knobs())
    except ValueError as e:
        raise ApiError(400, str(e))
    local = recipes.match_local(r["model"], config.read_sections())
    if not local:
        out = {"ok": False, "missing": r["model"], "dropped": r["dropped"]}
        if req.body.get("download"):
            err = _start_recipe_download(r["model"])
            out.update(downloading=not err, error=err)
        return 200, out
    preset = ""
    if r["settings"]:
        preset = recipes.unique_name(r["name"], config.get_presets())
        config.save_preset(preset, r["settings"])
    engine = recipes.match_engine(r["engine"], prebuilt.list_installs(ENGINES_DIR, cfg().get("server_bin", "")))
    name = recipes.unique_name(r["name"], config.get_profiles())
    config.save_profile(name, {"model": local, "backend": "llamacpp", "preset": preset, "engine": engine,
                               "source": "recipe"})
    wanted = r["engine"] and not engine
    return 200, {"ok": True, "name": name, "model": local, "preset": preset, "engine": engine,
                 "engine_missing": f"{r['engine']['tag']} {r['engine']['variant']}".strip() if wanted else "",
                 "dropped": r["dropped"]}


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
    roots = (req.body.get("roots") or None) if "roots" in req.body \
        else (cfg().get("model_dirs") or None)
    return 200, {"entries": scanner.scan(roots)}


def post_scan_apply(req):
    entries = req.body.get("entries", [])
    _persist_scanned_entries(entries)
    config.apply_ctx_defaults()
    reload_presets()
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
            config.prune_vision(mid, engine)
    if removed:
        reload_presets()
    return 200, {"removed": removed}


def post_hub_search(req):
    try:
        res = hub.search(req.body.get("query", ""), req.body.get("sort", "downloads"))
        inst = installed_repos(res, config.read_sections(),
                               vllm_registry.models() if VLLM_SUPPORTED else [])
        total, used = vram_snapshot_mib()
        return 200, {"results": res, "vram_mib": total, "vram_used_mib": used, "installed": inst}
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
        return 200, {"error": str(e), "files": [], "mmproj": [], "mtp": []}


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
    if req.body.get("mtp"):
        paths.append(req.body["mtp"])
    dest = os.path.join(download_dir(), repo.replace("/", "--"))
    ok = DOWNLOADS.start(repo, paths, dest)
    return 200, {"started": ok, "dest": dest}


def post_hub_cancel(req):
    return 200, {"ok": DOWNLOADS.cancel()}


def post_hub_pause(req):
    return 200, {"ok": DOWNLOADS.pause()}


def post_hub_resume(req):
    return 200, {"ok": DOWNLOADS.resume()}


def _register_download(path):
    """Register a finished download (and its folder's shards/mmproj) in
    models.ini. Returns the model ids added, the downloaded file's first so
    "Load & Chat" loads what was just fetched."""
    folder = os.path.dirname(path)
    entries = _register_ggufs_beside(
        [os.path.join(folder, f) for f in os.listdir(folder)
         if f.lower().endswith(".gguf")])
    same = lambda e: os.path.normcase(os.path.abspath(e["model"])) == os.path.normcase(os.path.abspath(path))
    return [e["id"] for e in sorted(entries, key=lambda e: not same(e))]


DOWNLOADS.on_done = _register_download


def post_hub_add(req):
    """Register a finished download in models.ini (kept for older panels;
    downloads now register themselves when they finish)."""
    path = req.body.get("path", "")
    if not path or not os.path.exists(path):
        raise ApiError(400, "file not found")
    return 200, {"ok": True, "added": _register_download(path)}


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
def _v_skin(v): return v if v in ("", "stowage", "hearth", "classic") else None
def _v_dirs(v):
    return v if isinstance(v, list) and all(isinstance(x, str) for x in v) else None
def _v_int(lo, hi):
    return lambda v: v if isinstance(v, int) and not isinstance(v, bool) and lo <= v <= hi else None

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
    "skin":                    _v_skin,
    "onboarded":               _v_bool,
    "auto_load_model":         _v_str,
    "wsl_distro":              _v_str,
    "vllm_port":               _v_port,
    "model_dirs":              _v_dirs,
    "anthropic_default_model": _v_str,
    "anthropic_shim_enabled":  _v_bool,
    "vram_bandwidths":         _v_bandwidths,
    "vram_predict_enabled":    _v_bool,
    "multi_model":             _v_bool,
    "slot_cap":                _v_int(slots.CAP_MIN, slots.CAP_MAX),
    "slot_headroom_mib":       _v_int(0, 32768),
    "slot_autoload":           _v_bool,
    "game_mode":               _v_bool,
    "game_mode_keep":          _v_dirs,     # model ids that stay loaded
    "game_mode_games":         _v_dirs,     # extra game executables
    "game_mode_restore":       _v_bool,
    "elastic_vram":            _v_bool,
    "elastic_grow":            _v_bool,
    "elastic_keep_conversation": _v_bool,
    "elastic_headroom_mib":    _v_int(512, 16384),
}
# what the router is started with: changing one restarts it (and unloads models)
_POOL_KEYS = ("multi_model", "slot_cap", "slot_autoload")


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
    if any(k in accepted for k in _POOL_KEYS):
        with _ROUTER_LIFECYCLE_LOCK:
            restarted, err = _sync_router_pool(c)
        out["router"] = {"restarted": restarted, "error": err or ""}
    if accepted.get("multi_model") is False:
        SLOTS.unplace_all()                 # models.ini back the way the user wrote it
    return 200, out


def _active_server_bin(c=None):
    """Return the server binary path for the currently active engine."""
    c = c or cfg()
    if c.get("active_engine") == "ikllama":
        return c.get("ik_llama_server_bin", "")
    return c.get("server_bin", "")


def _router_pool(c, sbin):
    """router_ctl's pool for config `c` on binary `sbin` (None = single)."""
    return slots.router_pool(c, bool(sbin) and router_ctl.supports_no_autoload(sbin))


def _sync_router_pool(c):
    """Restart a running router whose pool isn't the one `c` asks for.
    (restarted, error). Callers hold _ROUTER_LIFECYCLE_LOCK."""
    sbin = _active_server_bin(c)
    if not sbin or not os.path.exists(sbin) or not router_ctl.is_running(c["router_port"]):
        return False, ""
    want = _router_pool(c, sbin)
    if want == router_ctl.running_pool(LOGDIR):
        return False, ""
    ok, err = router_ctl.restart(sbin, config.ini_path(), c["router_port"],
                                 c.get("router_host", "127.0.0.1"),
                                 c.get("router_api_key", ""), LOGDIR,
                                 c.get("router_local_key", ""), want,
                                 slot_save=_el_slot_save(c))
    return bool(ok), err


def reconcile_router_pool(wait_s=20):
    """Startup check: run.ps1 / run.sh start the router one-model-at-a-time;
    with multi_model on (or just turned off) restart it with the right pool,
    before anything is loaded. True when a restart was attempted."""
    c = cfg()
    if not c.get("multi_model"):
        SLOTS.unplace_all()             # turned off while LlamaForge was down
        if router_ctl.running_pool(LOGDIR) is None:
            return False                # single mode, single router: nothing to do
    deadline = time.monotonic() + wait_s   # the runner's router may still be binding
    while not router_ctl.is_running(c["router_port"]) and time.monotonic() < deadline:
        time.sleep(0.5)
    with _ROUTER_LIFECYCLE_LOCK:
        restarted, err = _sync_router_pool(cfg())
    if err:
        print(f"  WARNING: router restart for multi-model failed ({err})")
    return restarted or bool(err)


def reconcile_router_auth():
    """Startup check: restart a router that runs unkeyed or with a stale key.
    The runners leave an already-listening router alone, so without this an
    upgrade would keep the old open router until the next reboot. True when a
    restart was attempted."""
    c = cfg()
    if router_ctl.auth_state(c["router_port"],
                             network_policy.effective_key(c)) not in ("open", "mismatch"):
        return False
    sbin = _active_server_bin(c)
    if not sbin or not os.path.exists(sbin):
        return False
    with _ROUTER_LIFECYCLE_LOCK:
        ok, err = router_ctl.restart(sbin, config.ini_path(), c["router_port"],
                                     c.get("router_host", "127.0.0.1"),
                                     c.get("router_api_key", ""), LOGDIR,
                                     c.get("router_local_key", ""), _router_pool(c, sbin),
                                     slot_save=_el_slot_save(c))
    print("  router restarted with API-key auth" if ok
          else f"  WARNING: router auth restart failed ({err})")
    return True


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


def _network_error(error, secret):
    text = str(error or "")
    return text.replace(secret, "[redacted]") if secret else text


def post_network(req):
    with _ROUTER_LIFECYCLE_LOCK:
        return _post_network_locked(req)


def _post_network_locked(req):
    current = cfg()
    try:
        mutation = network_policy.apply_request(current, req.body or {})
    except ValueError as e:
        raise ApiError(400, str(e))
    except Exception:
        raise ApiError(500, "network change could not be prepared") from None

    try:
        c = config.update({
            "router_host": mutation.router_host,
            "router_api_key": mutation.router_api_key,
        })
    except Exception:
        raise ApiError(500, "network settings could not be saved") from None
    try:
        sbin = _active_server_bin(c)
        ok, error = router_ctl.restart(
            sbin, config.ini_path(), c["router_port"],
            mutation.router_host, mutation.router_api_key, LOGDIR,
            c.get("router_local_key", ""), _router_pool(c, sbin),
            slot_save=_el_slot_save(c))
    except Exception as exc:
        ok, error = False, exc
    running = router_ctl.is_running(c["router_port"])
    out = _network_status(c, running)
    out.update({
        "ok": bool(ok),
        "saved": True,
        "restart_status": ("failed" if not ok else
                           "running" if running else "starting"),
    })
    if error:
        out["error"] = _network_error(error, mutation.router_api_key)
    if mutation.generated_api_key is not None:
        out["generated_api_key"] = mutation.generated_api_key
    return (200 if ok else 500), out


def post_engine_switch(req):
    with _ROUTER_LIFECYCLE_LOCK:
        return _post_engine_switch_locked(req)


def _post_engine_switch_locked(req):
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
                                 c.get("router_api_key", ""), LOGDIR,
                                 c.get("router_local_key", ""), _router_pool(c, sbin),
                                 slot_save=_el_slot_save(c))
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
        total, used = vram_snapshot_mib()
        return 200, {"results": res, "vram_mib": total, "vram_used_mib": used, "installed": inst}
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
    target = _resolve_agent_request(req.body if req.body is not None else {})
    try:
        out = agentsetup.apply(
            target["agent"], os.path.expanduser("~"), target["endpoint"],
            target["api_key"], target["model"], target["small"])
    except Exception:
        raise ApiError(500, "agent configuration could not be applied") from None
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



# ============================================================ prebuilt engine
# Official ggml-org binaries: the no-compiler path (see prebuilt.py).

ENGINES_DIR = os.path.join(ROOT, "engines", "llama.cpp")


def _activate_prebuilt(sbin):
    """Point llama.cpp at a prebuilt binary and (re)start the router on it.

    Unlike _record_server_bin this overwrites: installing or picking a build
    is an explicit user choice. Runs on the installer thread, hence update()."""
    with _ROUTER_LIFECYCLE_LOCK:
        c = config.update({"server_bin": sbin, "active_engine": "llamacpp"})
        try:
            ok, err = router_ctl.restart(sbin, config.ini_path(), c["router_port"],
                                         c.get("router_host", "127.0.0.1"),
                                         c.get("router_api_key", ""), LOGDIR,
                                         c.get("router_local_key", ""), _router_pool(c, sbin),
                                         slot_save=_el_slot_save(c))
        except Exception as e:
            ok, err = False, str(e)
        return ok, err


PREBUILT = prebuilt.Installer(ROOT, LOGDIR, on_installed=_activate_prebuilt)
def _protected_installs():
    """Install dirs pruning must keep: the ones a launch profile or a model pins."""
    installs = prebuilt.list_installs(ENGINES_DIR)
    return sorted(set(profiles.pinned_dirs(config.get_profiles(), installs))
                  | set(builds.protected_dirs(cfg(), installs)))


PREBUILT.protected = _protected_installs
_PREBUILT_CACHE = {}            # channel -> (expires_at, result)
_PREBUILT_LOCK = threading.Lock()
PREBUILT_TTL, PREBUILT_FAIL_TTL = 900, 60


def _prebuilt_latest(channel, force=False):
    now = time.time()
    with _PREBUILT_LOCK:
        hit = _PREBUILT_CACHE.get(channel)
        if hit and hit[0] > now and not force:
            return hit[1]
    plat, gpus, driver = prebuilt._detect()
    try:
        rel = prebuilt.resolve(channel, plat=plat, gpus=gpus, driver=driver)
        pick = prebuilt.choose(rel["assets"], plat, gpus, driver)
        size = sum(int((a or {}).get("size") or 0) for a in (pick["bin"], pick["cudart"]))
        out = {"ok": True, "tag": rel["tag"], "label": rel["label"],
               "published": rel["published"], "variant": pick["variant"],
               "reason": pick["reason"], "alternatives": pick["alternatives"],
               "download_bytes": size}
        ttl = PREBUILT_TTL
    except Exception as e:
        out, ttl = {"ok": False, "error": str(e)}, PREBUILT_FAIL_TTL
    out.update(platform="-".join(plat), driver_cuda=".".join(map(str, driver)) if driver else "",
               gpus=[g.get("name", "") for g in gpus])
    with _PREBUILT_LOCK:
        _PREBUILT_CACHE[channel] = (now + ttl, out)
    return out


def _build_num(tag):
    m = re.match(r"^b(\d+)$", tag or "")
    return int(m.group(1)) if m else -1


def get_engine_prebuilt(req):
    channel = req.q("channel") or cfg().get("prebuilt_channel", "nightly")
    if channel not in prebuilt.CHANNELS:
        raise ApiError(400, f"unknown channel: {channel}")
    active = cfg().get("server_bin", "")
    installs = prebuilt.list_installs(ENGINES_DIR, active)
    latest = _prebuilt_latest(channel, force=req.flag("force"))
    current = next((i for i in installs if i["active"]), None)
    return 200, {
        "channel": channel,
        "latest": latest,
        "installs": installs,
        "active_bin": active,
        "using_prebuilt": bool(current),
        # bNNNN tags compare numerically; anything else never claims "newer"
        "update_available": bool(latest.get("ok") and current and
                                 _build_num(latest["tag"]) > _build_num(current.get("tag"))),
    }


def get_engine_prebuilt_status(req):
    s = PREBUILT.progress()
    s["log"] = PREBUILT.tail(120)
    return 200, s


def post_engine_prebuilt_install(req):
    channel = req.body.get("channel") or "nightly"
    if channel not in prebuilt.CHANNELS:
        raise ApiError(400, f"unknown channel: {channel}")
    variant = req.body.get("variant") or None
    config.update({"prebuilt_channel": channel})
    started = PREBUILT.start(channel, variant)
    with _PREBUILT_LOCK:
        _PREBUILT_CACHE.clear()
    return 200, {"started": started}


def post_engine_prebuilt_cancel(req):
    return 200, {"cancelled": PREBUILT.cancel()}


def post_engine_prebuilt_use(req):
    """Switch to an already-installed build (update rollback). Only directories
    list_installs() reports are accepted - never an arbitrary path."""
    want = os.path.normcase(os.path.abspath(req.body.get("dir") or ""))
    inst = next((i for i in prebuilt.list_installs(ENGINES_DIR)
                 if os.path.normcase(os.path.abspath(i["dir"])) == want), None)
    if not inst or not inst.get("server_bin"):
        raise ApiError(404, "no such installed build")
    ok, err = _activate_prebuilt(inst["server_bin"])
    return 200, {"ok": ok, "error": err, "server_bin": inst["server_bin"]}


# ============================================================ text-to-speech
# llama.cpp's llama-tts behind an OpenAI-style /v1/audio/speech (see tts.py).

def _openai_err(status, message, kind):
    return status, {"error": {"message": message, "type": kind}}


def _speak(body):
    """-> (status, audio bytes, content type); raises ValueError / TtsError."""
    r = tts.parse_request(body)
    audio = TTS.speak(cfg(), r)
    return 200, audio, ("audio/pcm" if r["format"] == "pcm" else "audio/wav")


def post_audio_speech(req):
    """POST /v1/audio/speech for clients (Open WebUI, scripts, agents)."""
    if not _shim_auth_ok(req.headers):
        return _openai_err(401, "invalid key", "authentication_error")
    try:
        return _speak(req.body)
    except ValueError as e:
        return _openai_err(400, str(e), "invalid_request_error")
    except tts.TtsError as e:
        kind = "invalid_request_error" if e.status == 400 else "server_error"
        return _openai_err(e.status, str(e), kind)


def post_tts_speak(req):
    """The panel's own Speak button: same engine, panel-style errors."""
    try:
        return _speak(req.body)
    except ValueError as e:
        raise ApiError(400, str(e))
    except tts.TtsError as e:
        raise ApiError(e.status, str(e))


def get_tts_status(req):
    s = tts.status(cfg())
    s["busy"] = TTS.busy
    s["download"] = TTS_DOWNLOADS.progress()
    c = cfg()   # /v1/audio/speech asks for the router key when the router is exposed
    s["needs_key"] = c.get("router_host", "127.0.0.1") != "127.0.0.1" and bool(c.get("router_api_key"))
    return 200, s


def post_tts_get(req):
    """Download a catalog speech model (default: Qwen3-TTS 1.7B, Q8_0 + mmproj)."""
    e = tts.catalog_entry(req.body.get("id") or tts.CATALOG[0]["id"])
    if not e:
        raise ApiError(400, "unknown speech model")
    c = cfg()
    started = TTS_DOWNLOADS.start(e["repo"], list(e["files"]),
                                  tts.entry_dir(tts.models_dir(c), e))
    if started and e.get("voice"):
        # A 640 KB clip from another repo: fetched beside the job, best effort.
        def fetch(folder=tts.voices_dir(c), voice=e["voice"]):
            try:
                tts.fetch_voice(folder, voice)
            except Exception:
                pass
        threading.Thread(target=fetch, daemon=True).start()
    return 200, {"started": started}


def post_tts_voice(req):
    import base64, binascii
    try:
        data = base64.b64decode(req.body.get("wav_b64") or "", validate=True)
    except (binascii.Error, ValueError):
        raise ApiError(400, "wav_b64 is not base64")
    try:
        tts.save_voice(tts.voices_dir(cfg()), req.body.get("name"), data)
    except ValueError as e:
        raise ApiError(400, str(e))
    return 200, {"voices": tts.list_voices(tts.voices_dir(cfg()))}


def post_tts_voice_delete(req):
    tts.delete_voice(tts.voices_dir(cfg()), req.body.get("name"))
    return 200, {"voices": tts.list_voices(tts.voices_dir(cfg()))}


# =================================================================== the tables

def _embers(fn):
    """An embers.panel handler as a route: its Error becomes our ApiError."""
    def handler(req):
        try:
            return fn(req, cfg())
        except embers_panel.Error as e:
            raise ApiError(e.status, e.message)
    return handler

GET_ROUTES = {
    "/api/state":             get_state,
    "/api/schema":            get_schema,
    "/api/gpus":              get_gpus,
    "/api/setup":             get_setup,
    "/api/build/info":        get_build_info,
    "/api/build/log":         get_build_log,
    "/api/engine/prebuilt":   get_engine_prebuilt,
    "/api/engine/prebuilt/status": get_engine_prebuilt_status,
    "/api/hub/progress":      get_hub_progress,
    "/api/starters":          get_starters,
    "/api/router/log":        get_router_log,
    "/api/stats":             get_stats,
    "/api/scan/missing":      get_scan_missing,
    "/api/network":           get_network,
    "/api/vllm/log":          get_vllm_log,
    "/api/vllm/setup":        get_vllm_setup,
    "/api/vllm/schema":       get_vllm_schema,
    "/api/vllm/version":      get_vllm_version,
    "/api/vllm/hub/progress": get_vllm_hub_progress,
    "/api/feed":              get_feed,
    "/api/recipes/gallery":   get_recipes_gallery,
    "/api/app/update":        lambda req: (200, APP_UPDATE.progress()),
    "/api/model/metadata":    get_model_metadata,
    "/api/model/diag":        get_model_diag,
    "/api/presets":           get_presets,
    "/api/wiki/docs":         get_wiki_docs,
    "/api/wiki/doc":          get_wiki_doc,
    "/api/wiki/profiles":     get_wiki_profiles,
    "/api/wiki/preview":      get_wiki_preview,
    "/api/docs":              get_docs,
    "/api/docs/page":         get_docs_page,
    "/api/slots":             get_slots,
    "/api/slots/plan":        get_slots_plan,
    "/api/mcp/setup":         get_mcp_setup,
    "/api/pi/status":         get_pi_status,
    "/api/tts/status":        get_tts_status,
    "/api/tts/progress":      lambda req: (200, TTS_DOWNLOADS.progress()),
    "/api/embers":            _embers(embers_panel.get_embers),
    "/api/embers/templates":  _embers(embers_panel.get_templates),
    "/api/embers/brief":      _embers(embers_panel.get_brief),
    "/api/embers/pages":      _embers(embers_panel.get_pages),
    "/api/embers/page":       _embers(embers_panel.get_page),
    "/api/embers/raw":        _embers(embers_panel.get_raw),
    "/api/embers/log":        _embers(embers_panel.get_log),
    "/api/claude/check":      get_claude_check,
}

POST_ROUTES = {
    "/v1/audio/speech":         post_audio_speech,
    "/api/tts/speak":           post_tts_speak,
    "/api/tts/get":             post_tts_get,
    "/api/tts/voice":           post_tts_voice,
    "/api/tts/voice/delete":    post_tts_voice_delete,
    "/api/client/config":       post_client_config,
    # engine-agnostic (dispatch on the model's backend)
    "/api/models/load":         post_model_load,
    "/api/models/unload":       post_model_unload,
    "/api/models/save":         post_model_save,
    "/api/models/delete":       post_model_delete,
    "/api/models/unregister":   post_model_unregister,
    # llama.cpp-specific aliases
    "/api/save":                post_save,
    "/api/model/vision":        post_model_vision,
    "/api/load":                post_load,
    "/api/unload":              post_unload,
    "/api/unload_all":          post_unload_all,
    "/api/autotune/recommend":  post_autotune_recommend,
    "/api/autotune/refine":     post_autotune_refine,
    "/api/presets/save":        post_presets_save,
    "/api/presets/bind":        post_presets_bind,
    "/api/presets/delete":      post_presets_delete,
    "/api/presets/apply":       post_presets_apply,
    "/api/profiles/save":       post_profiles_save,
    "/api/profiles/delete":     post_profiles_delete,
    "/api/profiles/launch":     post_profiles_launch,
    "/api/profiles/export":     post_profiles_export,
    "/api/profiles/import":     post_profiles_import,
    "/api/build/start":         post_build_start,
    "/api/setup/install":       post_setup_install,
    "/api/pi/install":          post_pi_install,
    "/api/pi/remove":           post_pi_remove,
    "/api/scan":                post_scan,
    "/api/scan/apply":          post_scan_apply,
    "/api/scan/prune":          post_scan_prune,
    "/api/hub/search":          post_hub_search,
    "/api/app/update":          post_app_update,
    "/api/app/restart":         post_app_restart,
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
    "/api/engine/prebuilt/install": post_engine_prebuilt_install,
    "/api/engine/prebuilt/cancel":  post_engine_prebuilt_cancel,
    "/api/engine/prebuilt/use":     post_engine_prebuilt_use,
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
    "/api/agent/config":        post_agent_config,
    "/api/agent/apply":         post_agent_apply,
    "/api/claude/pick-folder":  post_claude_pick_folder,
    "/api/claude/launch":       post_claude_launch,
    "/api/wiki/doc":            post_wiki_doc,
    "/api/wiki/doc/delete":     post_wiki_doc_delete,
    "/api/wiki/profile":        post_wiki_profile,
    "/api/wiki/profile/delete": post_wiki_profile_delete,
    "/api/wiki/active":         post_wiki_active,
    "/api/wiki/export":         post_wiki_export,
    "/api/slots/main":          post_slots_main,
    "/api/slots/apply":         post_slots_apply,
    "/api/model/build":         post_model_build,
    "/api/embers/create":       _embers(embers_panel.post_create),
    "/api/embers/update":       _embers(embers_panel.post_update),
    "/api/embers/delete":       _embers(embers_panel.post_delete),
    "/api/embers/run":          _embers(embers_panel.post_run),
    "/api/embers/cancel":       _embers(embers_panel.post_cancel),
    "/api/embers/folder":       _embers(embers_panel.post_folder),
    "/api/embers/ask":          _embers(embers_panel.post_ask),
    "/api/embers/forge":        _embers(embers_panel.post_forge),
    "/api/embers/push/test":    _embers(embers_panel.post_push_test),
}


# Built last: backends.Registry captures this module as its dependency bundle,
# so every helper it reaches for must already be defined.
REGISTRY = backends.Registry(sys.modules[__name__])
SLOTS = slotctl.SlotManager(sys.modules[__name__], os.path.join(ROOT, "footprints.json"))
# Models pinned to another build, each in its own llama-server. Adopting the
# ones a previous panel left running is server.main()'s job (PROCS.reconcile()).
PROCS = slotproc.Manager(LOGDIR)


def _proc_endpoints():
    """{model: endpoint} of the process slots that are up, for the stats poller."""
    return {mid: s["endpoint"] for mid, s in PROCS.status().items() if s.get("state") == "ready"}


stats.TRACKER.proc_source = _proc_endpoints


def _game_loaded():
    """Router models plus process-slot models that are up."""
    loaded = _router_loaded_ids()
    return loaded + [m for m in PROCS.status() if m not in loaded]


def _game_unload(mid):
    status, res = post_unload(Req({"model": mid}))
    return status == 200 and res.get("success", True) is not False


def _game_load(mid):
    status, _ = post_load(Req({"model": mid}))
    return status == 200


# Game mode: server.main() starts its loop; it only looks at processes while on.
GAMEMODE = gamemode.GameMode(cfg, gamemode.detect, _game_loaded, _game_unload, _game_load)


# ---- Elastic VRAM (elastic.py plans, slotkeep.py keeps the conversation) -----
#
# Concurrency: every helper below that touches models.ini or elastic_placed runs
# with ELASTIC._run held - tick() holds it around tidy() and every swap, and
# grow_now() holds it around its swap. A swap unloads the model and loads it
# again; tidy() would read the model as "not loaded" in between and put the
# user's keys back under it, but it can't run then: it only runs inside tick().

ELASTIC_DIR = router_ctl.slot_save_dir(LOGDIR)
_EL_LAYOUTS = {}


def _el_slot_save(c):
    """Whether the router gets a slot save folder: only while Elastic VRAM is
    on and keeps conversations. Read when the router (re)starts; toggling
    either never restarts it (that would unload the model being kept)."""
    return bool(c.get("elastic_vram") and c.get("elastic_keep_conversation", True))


def _el_layout(path):
    """gguf.layout(path), read again only when the file changes: a tick comes
    every 5 s and a big model has thousands of tensors."""
    try:
        mtime = os.path.getmtime(path)
    except (OSError, TypeError, ValueError):
        return None
    got = _EL_LAYOUTS.get(path)
    if not got or got[0] != mtime:
        got = _EL_LAYOUTS[path] = (mtime, gguf.layout(path))
    return got[1]


def _el_placed(c=None):
    p = (c or cfg()).get("elastic_placed")
    return p if isinstance(p, dict) else {}


def _el_record(mid, rec):
    def fn(c):
        p = c.get("elastic_placed") if isinstance(c.get("elastic_placed"), dict) else {}
        if rec:
            p[mid] = rec
        else:
            p.pop(mid, None)
        c["elastic_placed"] = p
    config.mutate(fn)


def _el_ini():
    """Elastic only resizes llama.cpp router models, so it reads and writes the
    llama.cpp models.ini whichever engine is active."""
    return config.ini_path("llamacpp")


def _el_llamacpp(c=None):
    return (c or cfg()).get("active_engine", "llamacpp") == "llamacpp"


def _el_pending(old, new):
    """The record to hold while models.ini is rewritten: `old`, plus the keys
    `new` adds that `old` doesn't have. Every key then matches the ini whether
    the write lands or not: a key `old` has keeps its on-disk value until the
    write succeeds, and an added key that never got written reads as a user
    edit to plan_keys, which leaves the user's value alone."""
    old, new = old or {}, new or {}
    keys, was = dict(old.get("keys") or {}), dict(old.get("was") or {})
    for k, v in (new.get("keys") or {}).items():
        if k not in keys:
            keys[k] = v
            was[k] = (new.get("was") or {}).get(k)
    return _el_with_room({"keys": keys, "was": was}, old.get("room")) if keys else None


def _el_with_room(rec, room):
    """rec plus the free MiB per GPU the model had at the user's own settings
    before its first shrink (the controller grows back to that). It lives
    under the record's "room", which plan_keys never reads as an ini key, so
    it goes exactly when the record goes: back at the top, give-up, tidy."""
    if rec and room:
        rec["room"] = {str(g): int(v) for g, v in room.items()}
    return rec


def _el_room(rec):
    """A record's room with the GPU ids the snapshot uses (JSON made them
    strings), or None."""
    room = (rec or {}).get("room")
    if not isinstance(room, dict) or not room:
        return None
    try:
        return {int(g) if str(g).isdigit() else g: int(v) for g, v in room.items()}
    except (TypeError, ValueError):
        return None


def _el_spelt(section, keys):
    """A step's keys (canonical names) under the spelling `section` already
    uses - what plan_keys writes - so the settings we predict and key
    footprints by are the ones the load will actually read."""
    _, rec = elastic.plan_keys(section or {}, None, keys)
    return (rec or {}).get("keys") or {}


def _el_user_settings(mid, secs, rec):
    """mid's effective settings as the user wrote them: our keys taken back out."""
    sec = dict(secs.get(mid) or {})
    for k, v in ((rec or {}).get("keys") or {}).items():
        if sec.get(k) == v:
            w = (rec.get("was") or {}).get(k)
            if w is None:
                sec.pop(k, None)
            else:
                sec[k] = str(w).split(";", 1)[0].strip()
    return {**(secs.get("*") or {}), **sec}


def _el_step_settings(mid, secs, rec, keys):
    """mid's effective settings at a step with `keys`."""
    return {**_el_user_settings(mid, secs, rec), **_el_spelt(secs.get(mid), keys)}


def _el_gpus():
    """{gpu: {"used", "total"}} in MiB, straight from nvidia-smi."""
    if osplat.IS_MAC:                 # unified memory: nothing to move layers to
        return {}
    return {g["index"]: {"used": g["used"], "total": g["total"]}
            for g in _gpu_telemetry() if "index" in g and "used" in g and "total" in g}


def _el_snapshot():
    """The controller's look, every 5 s: nothing (no nvidia-smi run) while
    Elastic is off and nothing of ours is left to put back."""
    c = cfg()
    if not c.get("elastic_vram") and not _el_placed(c):
        return {}
    return _el_gpus()


def _el_measured(m, gpus):
    """One footprint out of Store.measured()'s {devices: {gpu: MiB}}: the one
    for the GPUs we see, else the largest."""
    if not m:
        return None
    return m.get(tuple(sorted(gpus))) or max(m.values(), key=lambda fp: sum(fp.values()))


def _el_models(gpus):
    """The loaded router models Elastic may resize, each with its ladder, every
    step's footprint (measured when a load has measured it) and the step it's
    on now. Models on a build of their own and vLLM are out of scope."""
    c = cfg()
    placed = _el_placed(c)
    if not c.get("elastic_vram") and not placed:
        return []
    if not _el_llamacpp(c):
        return []
    secs = config.read_sections(_el_ini())
    build = SLOTS.build_id(_active_server_bin(c))
    out = []
    for mid in _router_loaded_ids(("loaded", "sleeping")):   # not one still loading
        if PROCS.has(mid) or mid not in secs:
            continue
        rec = placed.get(mid)
        user = _el_user_settings(mid, secs, rec)
        lay = _el_layout(user.get("model") or "")
        steps = elastic.ladder(user, lay)
        if not steps:
            continue
        measured, predicted = [], []
        for st in steps:
            s = _el_step_settings(mid, secs, rec, st["keys"])
            measured.append(_el_measured(SLOTS.store.measured(footprint.key(mid, build, s)), gpus))
            predicted.append(int(footprint.predict(s, lay)["need_mib"]))
        out.append({"id": mid, "steps": steps, "at": _el_at(secs.get(mid) or {}, rec, steps),
                    "fps": elastic.footprints(measured, predicted, gpus), "room": _el_room(rec)})
    return out


def _el_at(section, rec, steps):
    """The step a model is on. Only our keys still in effect count (one the
    user has since edited is theirs). None: at the top. Rec keys carry the
    section's spelling (gpu-layers), ladder keys are canonical (n-gpu-layers),
    so both are compared normalized. Keys of ours that match no step (the
    ladder moved: a new build, an edited [*]) mean resized to somewhere
    unknown: call it the smallest step, which the controller can only grow
    from, so a grow, or turning Elastic off, takes it back to the top."""
    ours = footprint._norm({k: v for k, v in ((rec or {}).get("keys") or {}).items()
                            if section.get(k) == v})
    if not ours:
        return 0
    return next((i for i, st in enumerate(steps) if footprint._norm(st["keys"]) == ours),
                len(steps) - 1)


def _el_kv_mib(mid):
    secs = config.read_sections(_el_ini())
    s = {**(secs.get("*") or {}), **(secs.get(mid) or {})}
    p = footprint.predict(s, _el_layout(s.get("model") or ""))
    return int(p.get("kv_gpu_mib", 0) + p.get("kv_ram_mib", 0))


def _el_save(mid):
    notes = [] if slotkeep.wait_idle(router, mid) else ["a reply in progress was cut off"]
    files, why = slotkeep.save(router, mid, ELASTIC_DIR, _el_kv_mib(mid))
    if why:
        notes.append(why)
    return {"files": files, "note": "; ".join(notes)}


def _el_swap(mid, step, room=None):
    """Move mid to `step`: write the keys (recorded so they can be put back; a
    user edit wins, along with `room`, else the room already recorded), reload
    the preset, load. "" on success, else why not. On
    a single-model router, measure the step's footprint around the load - the
    pool's slot manager records its own.

    Record and models.ini are kept in step as far as two files can be: the
    write is one atomic file replace, bracketed by _el_pending (before) and
    the new record (after). A write that fails puts the old record back; a
    record that fails after the write puts the ini's old values back, best
    effort. A key can only leak if that rollback fails too. The unload and
    reload come after, so they can't strand a key."""
    if not _el_llamacpp():
        return "Elastic VRAM only resizes llama.cpp router models"
    ini = _el_ini()
    old = _el_placed().get(mid)
    sec = config.read_sections(ini, raw=True).get(mid) or {}
    writes, rec = elastic.plan_keys(sec, old, step["keys"])
    rec = _el_with_room(rec, room or (old or {}).get("room"))
    if writes:
        _el_record(mid, _el_pending(old, rec))
        try:
            config.set_keys(mid, writes, path=ini)
        except Exception:
            _el_record(mid, old)
            raise
    try:
        _el_record(mid, rec)
    except Exception:
        if writes:                                # the pending record still covers
            try:                                  # the old values: put them back
                config.set_keys(mid, {k: sec.get(k) for k in writes}, path=ini)
            except Exception:
                pass
        raise
    _reload_preset(mid)                           # unloads first when it's running
    pool = _slots_on()
    before = None if pool else _el_gpus()         # not the gated look: a grow to the
    try:                                          # top has just cleared the record
        _load_and_wait(mid, "main" if SLOTS._main() == mid else "worker")
    except RuntimeError as e:
        return str(e) or "load failed"
    if before:
        _el_measure(mid, rec, step, before, _el_gpus())
    return ""


def _el_measure(mid, rec, step, before, after):
    """Record what the load of `step` took on each GPU - unless it's far from
    the prediction (half to one and a half times it), which means other apps
    grew or shrank during the load and the delta isn't the model's."""
    fp = {g: after[g]["used"] - before[g]["used"] for g in after if g in before}
    devices = sorted(g for g, v in fp.items() if v >= footprint.NOISE_MIB)
    s = _el_step_settings(mid, config.read_sections(_el_ini()), rec, step["keys"])
    lay = _el_layout(s.get("model") or "")
    if lay:
        need = int(footprint.predict(s, lay).get("need_mib") or 0)
        took = sum(fp[g] for g in devices)
        if need > 0 and not need * 0.5 <= took <= need * 1.5:
            return
    SLOTS.store.record(footprint.key(mid, SLOTS.build_id(_active_server_bin()), s),
                       devices, fp)


def _el_restore(mid, handle):
    try:
        return slotkeep.restore(router, mid, handle["files"])
    finally:
        slotkeep.drop(ELASTIC_DIR, handle["files"])


def _el_drop(handle):
    slotkeep.drop(ELASTIC_DIR, (handle or {}).get("files") or [])


def _el_giveup(mid):
    """Even the smallest step didn't load: the user's keys go back and the
    model stays unloaded until they load it. The record goes only once the
    keys are back, so a failed write leaves it for tidy(). On another engine
    nothing is touched: tidy() puts the keys back once llama.cpp is active."""
    if not _el_llamacpp():
        return
    ini = _el_ini()
    writes, _ = elastic.plan_keys(config.read_sections(ini, raw=True).get(mid) or {},
                                  _el_placed().get(mid), {})
    if writes:
        config.set_keys(mid, writes, path=ini)
    _el_record(mid, None)
    _reload_preset(mid)


def _el_tidy():
    """A resized model that's no longer loaded (unloaded by hand, by Game
    mode, by a router restart) gets the user's keys back. Each model's keys go
    in one models.ini write, and its record goes only after that write, so a
    failure partway leaves the rest recorded for the next tick; whatever was
    written still reaches the router. On another engine the llama.cpp router
    isn't the one answering: the records wait until it's back."""
    c = cfg()
    placed = _el_placed(c)
    if not placed or not _el_llamacpp(c):
        return
    st, data = router("/models")          # never raises: 599 when it's down
    if st != 200 or not isinstance(data, dict):
        return
    up = {m.get("id") for m in data.get("data", [])
          if (m.get("status") or {}).get("value") in ("loaded", "loading", "sleeping")}
    gone = [mid for mid in placed if mid not in up]
    if not gone:
        return
    ini = _el_ini()
    raw = config.read_sections(ini, raw=True)
    wrote = False
    try:
        for mid in gone:
            writes, _ = elastic.plan_keys(raw.get(mid) or {}, placed[mid], {})
            if writes:
                config.set_keys(mid, writes, path=ini)
                wrote = True
            _el_record(mid, None)
    finally:
        if wrote:
            reload_presets()


class _ElasticIO:
    """The controller's view of the world. Methods look the helpers up on
    every call, so tests can patch them on this module."""
    def snapshot(self):           return _el_snapshot()
    def models(self, gpus):       return _el_models(gpus)
    def game_active(self):        return bool(GAMEMODE.status().get("active"))
    def tidy(self):               return _el_tidy()
    def save(self, mid):          return _el_save(mid)
    def swap(self, mid, step, room=None): return _el_swap(mid, step, room)
    def restore(self, mid, h):    return _el_restore(mid, h)
    def drop(self, h):            return _el_drop(h)
    def giveup(self, mid):        return _el_giveup(mid)


# Elastic VRAM: server.main() starts its loop; idle unless elastic_vram is on.
ELASTIC = elastic.Elastic(cfg, _ElasticIO())


def get_elastic(req):
    return 200, ELASTIC.status()


def post_elastic_grow(req):
    """Grow a resized model back now. The swap takes seconds (a 27B about 20),
    so it runs in the background; the event log says how it went."""
    mid = str(req.body.get("model") or "")
    st = ELASTIC.status()
    if mid and st.get("swapping") == mid:
        return 409, {"ok": False, "error": f"{mid} is being resized right now"}
    if not any(m["id"] == mid and m["at"] > 0 for m in st["models"]):
        return 400, {"ok": False, "error": f"{mid or 'that model'} isn't resized"}
    threading.Thread(target=ELASTIC.safe_grow, args=(mid,), daemon=True,
                     name="elastic-grow").start()
    return 200, {"ok": True}


# The route tables are built above this section and hold the functions
# themselves, so these two join them here.
GET_ROUTES["/api/elastic"] = get_elastic
POST_ROUTES["/api/elastic/grow"] = post_elastic_grow
FORGE = forge_panel.Panel(sys.modules[__name__])
FORGE.register(GET_ROUTES, POST_ROUTES)
embers_panel.MAIN_FN = lambda: SLOTS._main()   # Ask lists the pool's main model first


class EmbersPool:
    """The pool as the embers scheduler may use it: worker loads beside the
    user's models, never an eviction, never a role change."""

    def active(self):
        return _slots_on()

    def plan(self, mid):
        return SLOTS.plan(mid, "worker")

    def load(self, mid):
        return SLOTS.load(mid, "worker", False, wait=True, keep_role=True)

    def unload(self, mid):
        return SLOTS.unload(mid)

    def touch(self, mid):
        SLOTS.touch(mid)

    def main(self):
        return SLOTS._main()

    def own_build(self, mid):
        """Embers go through the router; a model in its own process isn't there."""
        sbin, err = pinned_bin(mid)
        if err:
            return f"{mid} is pinned to a build that can't be used: {err}"
        if sbin:
            return (f"{mid} runs on its own build, outside the router embers use; "
                    f"pick another model for this ember, or set {mid} back to the "
                    f"router's build")
        return ""
