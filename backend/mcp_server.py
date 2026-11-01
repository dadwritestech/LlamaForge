"""LlamaForge as an MCP server.

Claude Code, Codex or any MCP client drives LlamaForge with it - what's loaded,
load/unload, fit checks, Hugging Face search and downloads - and can hand a
whole task to pi running on a loaded local model:

    claude mcp add --scope user llamaforge -- python <root>/backend/mcp_server.py

stdio transport, newline-delimited JSON-RPC, stdlib only. It is a thin client
over the panel's HTTP API on 127.0.0.1:<panel_port>: the panel's guard admits a
loopback caller that sends no Origin, so there is no token and no new way in.
It reads config.json (never writes it) for the panel port, pi's location and
the router key that ask/pi_run need to reach a model's endpoint.

Speaks both protocol eras: the initialize handshake (2024-11-05 .. 2025-11-25)
and the stateless 2026-07-28 revision, where every request carries its version
and client capabilities in params._meta.
"""
import json
import os
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request

import config
import network_policy
import pirun
import version

LEGACY = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
MODERN = ("2026-07-28",)
SUPPORTED = MODERN + LEGACY
K_VER = "io.modelcontextprotocol/protocolVersion"
K_CAPS = "io.modelcontextprotocol/clientCapabilities"
K_SERVER = "io.modelcontextprotocol/serverInfo"
SERVER_INFO = {"name": "llamaforge", "title": "LlamaForge", "version": version.VERSION,
               "websiteUrl": "https://github.com/dadwritestech/LlamaForge"}
CAPABILITIES = {"tools": {"listChanged": False}}
LIST_TTL_MS = 3_600_000          # the tool list only changes with an upgrade

PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS = -32700, -32600, -32601, -32602
UNSUPPORTED_VERSION = -32022

INSTRUCTIONS = (
    "LlamaForge runs local LLMs (llama.cpp; vLLM on Windows) on this machine's GPUs. "
    "status shows what is loaded and where; list_models what is installed. "
    "load_model loads one (fit_check first in multi-model mode; in single-model mode "
    "a load replaces the running model). search_models, list_files and download_model "
    "fetch new GGUFs from Hugging Face. ask sends one prompt to a loaded model. "
    "pi_run hands a whole task to pi - an open-source coding agent - running on a "
    "loaded local model in a directory you choose: use it to offload grunt work "
    "(summaries, boilerplate, bulk edits, first-pass reviews) to local compute. "
    "Local models are smaller than you: give pi_run small, concrete, checkable tasks. "
    "forge_start runs Auto-Forge, which benchmarks setups of one model on these GPUs and "
    "finds the fastest; it unloads every model for the run, so ask the user first. "
    "forge_status follows it; forge_apply writes the winner only when the user agrees.")

HUB_LIMIT = 20
FORGE_INTENTS = ("coding", "chat", "long", "agent")
FORGE_TOP = 5
_BOARD_KEYS = ("label", "score", "code", "prose", "summary", "prefill", "vram_mib", "ctx",
               "finalist", "sanity")
_PROXYLESS = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class ToolError(Exception):
    """A tool failed in a way the calling model can read and act on."""


class PanelError(ToolError):
    """An HTTP call (panel or model endpoint) failed."""


def _log(msg):
    sys.stderr.write(f"llamaforge-mcp: {msg}\n")
    sys.stderr.flush()


def _http(method, url, body=None, timeout=30, key="", what="the LlamaForge panel"):
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if key:
        headers["Authorization"] = "Bearer " + key
    r = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        # never through a proxy: everything here is on 127.0.0.1
        with _PROXYLESS.open(r, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            err = json.loads(raw).get("error")
            err = err.get("message") if isinstance(err, dict) else err
        except (ValueError, AttributeError):
            err = None
        raise PanelError(f"HTTP {e.code} from {what}: {err or raw[:300].decode('utf-8', 'replace')}")
    except (urllib.error.URLError, OSError) as e:
        reason = getattr(e, "reason", e)
        # a timeout is an OSError too, but it means busy, not down
        if isinstance(e, TimeoutError) or isinstance(reason, TimeoutError):
            raise PanelError(f"{what} did not answer within {timeout}s")
        raise PanelError(f"can't reach {what} at {url.split('?')[0]} ({reason})")
    try:
        return json.loads(raw or b"{}")
    except ValueError:
        raise PanelError(f"{what} did not answer with JSON")


class Panel:
    """The panel's HTTP API, on the port config.json names."""

    def __init__(self, cfg=config.load):
        self.cfg = cfg

    def _url(self, path, params=None):
        port = int(self.cfg().get("panel_port") or 8090)
        q = ("?" + urllib.parse.urlencode(params)) if params else ""
        return f"http://127.0.0.1:{port}{path}{q}"

    def _call(self, method, path, body, params, timeout):
        try:
            return _http(method, self._url(path, params), body, timeout)
        except PanelError as e:
            if str(e).startswith("can't reach"):
                raise PanelError(f"the LlamaForge panel isn't running ({e}); start LlamaForge "
                                 "(run.ps1 / run.sh) and try again")
            raise

    def get(self, path, params=None, timeout=30):
        return self._call("GET", path, None, params, timeout)

    def post(self, path, body=None, timeout=60):
        return self._call("POST", path, body or {}, None, timeout)


def http_chat(url, body, key, timeout):
    return _http("POST", url, body, timeout, key=key, what="the model endpoint")


# ---------------------------------------------------------------- arguments
def _prop(t, desc, **kw):
    return dict(type=t, description=desc, **kw)


def _schema(props=None, required=()):
    s = {"type": "object", "properties": props or {}, "additionalProperties": False}
    if required:
        s["required"] = list(required)
    return s


_TYPES = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
}


def check_args(schema, args):
    """Validate against the tool's inputSchema; nulls count as absent."""
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ToolError("invalid arguments: expected an object")
    args = {k: v for k, v in args.items() if v is not None}
    props = schema.get("properties", {})
    for k in args:
        if k not in props:
            raise ToolError(f"invalid arguments: unknown argument {k!r} "
                            f"(takes: {', '.join(props) or 'nothing'})")
    for k in schema.get("required", []):
        if k not in args or args[k] == "":
            raise ToolError(f"invalid arguments: {k} is required")
    for k, v in args.items():
        p = props[k]
        if not _TYPES[p["type"]](v):
            raise ToolError(f"invalid arguments: {k} must be a {p['type']}")
        if "enum" in p and v not in p["enum"]:
            raise ToolError(f"invalid arguments: {k} must be one of {', '.join(p['enum'])}")
        if "minimum" in p and v < p["minimum"]:
            raise ToolError(f"invalid arguments: {k} must be at least {p['minimum']}")
        if "maximum" in p and v > p["maximum"]:
            raise ToolError(f"invalid arguments: {k} must be at most {p['maximum']}")
    return args


# ---------------------------------------------------------------- model rows
def _v1(endpoint):
    e = (endpoint or "").rstrip("/")
    return e if e.endswith("/v1") else e + "/v1"


def _ctx(v):
    try:
        n = int(v)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _rows(state):
    return [m for m in state.get("models") or [] if isinstance(m, dict) and m.get("id")]


def _is_up(m):
    return m.get("status") == "loaded" and bool(m.get("endpoint"))


def find_row(state, mid):
    for m in _rows(state):
        if m["id"] == mid:
            return m
    raise ToolError(f"no model named {mid!r} in LlamaForge (list_models shows the installed ones)")


def pick_model(state, name=None):
    """The model ask/pi_run use: the named one if it is loaded; else a worker
    (multi-model mode keeps the main for the user); else whatever is loaded."""
    if name:
        row = find_row(state, name)
        if not _is_up(row):
            raise ToolError(f"{name} is not loaded ({row.get('status')}); call load_model first")
        return row
    up = [m for m in _rows(state) if _is_up(m)]
    if not up:
        raise ToolError("no model is loaded; call load_model first "
                        "(list_models shows what is installed)")
    slots = state.get("slots") or {}
    main = slots.get("main") if slots.get("enabled") else ""
    workers = [m for m in up if m["id"] != main] if main else []
    return (workers or up)[0]


def _summary(m, slots):
    on, main = bool(slots.get("enabled")), slots.get("main") or ""
    out = {"id": m["id"], "role": ("main" if m["id"] == main else "worker") if on else None,
           "endpoint": _v1(m["endpoint"]) if m.get("endpoint") else None,
           "ctx": _ctx(m.get("eff_ctx")), "backend": m.get("backend")}
    if m.get("build"):
        out["build"] = m["build"]
    return out


class _Answer:
    """A tool result whose text is prose (a model's answer), not the JSON."""

    def __init__(self, text, data, error=False):
        self.text, self.data, self.error = text, data, error


class _Ctx:
    """Per-call cancel flag and progress reporting."""

    def __init__(self, server, token):
        self.cancel = threading.Event()
        self._server, self._token, self._n = server, token, 0
        self._lock = threading.Lock()

    def progress(self, msg):
        if self._token is None or self.cancel.is_set():
            return
        with self._lock:
            self._n += 1
            n = self._n
        self._server._send({"jsonrpc": "2.0", "method": "notifications/progress",
                            "params": {"progressToken": self._token, "progress": n,
                                       "message": str(msg)}})


# ---------------------------------------------------------------- the server
class Server:
    def __init__(self, panel=None, cfg=config.load, chat=http_chat, runner=pirun.run,
                 send=None, threaded=True, poll_s=1.0):
        self.cfg = cfg
        self.panel = panel or Panel(cfg)
        self.chat, self.runner = chat, runner
        self.send = send
        self.threaded, self.poll_s = threaded, poll_s
        self._inflight = {}
        self._threads = []
        self._lock = threading.Lock()
        self.tools = self._tool_table()
        self._by_name = {t["name"]: t for t in self.tools}

    # ------------------------------------------------------------ transport
    def _send(self, msg):
        if msg is not None and self.send:
            self.send(msg)

    @staticmethod
    def _error(id_, code, message, data=None):
        err = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        return {"jsonrpc": "2.0", "id": id_, "error": err}

    @staticmethod
    def _result(id_, result, modern):
        if modern:
            result = dict(result, resultType="complete")
            result["_meta"] = dict(result.get("_meta") or {}, **{K_SERVER: SERVER_INFO})
        return {"jsonrpc": "2.0", "id": id_, "result": result}

    def handle_line(self, line):
        if not line.strip():
            return None
        try:
            msg = json.loads(line)
        except ValueError:
            return self._error(None, PARSE_ERROR, "parse error")
        return self.handle(msg)

    def handle(self, msg):
        """One parsed message -> its response, or None (a notification, or a
        tools/call answered later from its own thread)."""
        if isinstance(msg, list):
            return self._error(None, INVALID_REQUEST, "batches are not supported")
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
            return self._error(msg.get("id") if isinstance(msg, dict) else None,
                               INVALID_REQUEST, "not a JSON-RPC 2.0 message")
        method = msg.get("method")
        if not isinstance(method, str):
            return None                          # a response; we never ask the client anything
        if "id" not in msg:
            self._notification(method, msg.get("params"))
            return None
        id_ = msg["id"]
        if isinstance(id_, bool) or not isinstance(id_, (str, int)):
            return self._error(None, INVALID_REQUEST, "id must be a string or an integer")
        params = msg.get("params")
        if params is None:
            params = {}
        if not isinstance(params, dict):
            return self._error(id_, INVALID_PARAMS, "params must be an object")
        meta = params.get("_meta") if isinstance(params.get("_meta"), dict) else {}
        modern = False
        if K_VER in meta:
            v = meta[K_VER]
            if v in MODERN:
                if not isinstance(meta.get(K_CAPS), dict):
                    return self._error(id_, INVALID_PARAMS, f"_meta[{K_CAPS}] is required")
                modern = True
            elif v not in LEGACY:
                return self._error(id_, UNSUPPORTED_VERSION, "Unsupported protocol version",
                                   {"supported": list(SUPPORTED), "requested": v})
        return self._request(id_, method, params, meta, modern)

    def _notification(self, method, params):
        if method == "notifications/cancelled" and isinstance(params, dict):
            with self._lock:
                ctx = self._inflight.get(params.get("requestId"))
            if ctx:
                ctx.cancel.set()

    def _request(self, id_, method, params, meta, modern):
        if method == "initialize":
            asked = params.get("protocolVersion")
            return self._result(id_, {
                "protocolVersion": asked if asked in LEGACY else LEGACY[0],
                "capabilities": CAPABILITIES, "serverInfo": SERVER_INFO,
                "instructions": INSTRUCTIONS}, False)
        if method == "ping" and not modern:
            return self._result(id_, {}, False)
        if method == "server/discover":
            if not modern:
                return self._error(id_, INVALID_PARAMS,
                                   f"server/discover needs _meta[{K_VER}] = {MODERN[0]}")
            return self._result(id_, {
                "supportedVersions": list(SUPPORTED), "capabilities": CAPABILITIES,
                "instructions": INSTRUCTIONS, "ttlMs": LIST_TTL_MS, "cacheScope": "public"}, True)
        if method == "tools/list":
            res = {"tools": [{k: t[k] for k in ("name", "title", "description",
                                                 "inputSchema", "annotations")}
                             for t in self.tools]}
            if modern:
                res.update(ttlMs=LIST_TTL_MS, cacheScope="public")
            return self._result(id_, res, modern)
        if method == "tools/call":
            return self._tools_call(id_, params, meta, modern)
        return self._error(id_, METHOD_NOT_FOUND, f"method not found: {method}")

    def _tools_call(self, id_, params, meta, modern):
        tool = self._by_name.get(params.get("name"))
        if not tool:
            return self._error(id_, INVALID_PARAMS, f"unknown tool: {params.get('name')!r}")
        ctx = _Ctx(self, meta.get("progressToken"))
        with self._lock:
            self._inflight[id_] = ctx

        def work():
            try:
                res = self._run_tool(tool, params.get("arguments"), ctx)
            finally:
                with self._lock:
                    self._inflight.pop(id_, None)
            # a cancelled request gets no response at all
            return None if ctx.cancel.is_set() else self._result(id_, res, modern)

        if not self.threaded:
            return work()
        th = threading.Thread(target=lambda: self._send(work()), daemon=True,
                              name=f"mcp-{tool['name']}")
        with self._lock:
            self._threads = [t for t in self._threads if t.is_alive()] + [th]
        th.start()
        return None

    def _run_tool(self, tool, arguments, ctx):
        try:
            out = tool["fn"](check_args(tool["inputSchema"], arguments), ctx)
        except ToolError as e:
            return {"content": [{"type": "text", "text": str(e)}], "isError": True}
        except Exception as e:  # report, don't die: the client keeps the session
            _log(traceback.format_exc())
            return {"content": [{"type": "text", "text": f"internal error in {tool['name']}: {e}"}],
                    "isError": True}
        if isinstance(out, _Answer):
            return {"content": [{"type": "text", "text": out.text},
                                {"type": "text", "text": json.dumps(out.data, indent=2)}],
                    "structuredContent": out.data, "isError": out.error}
        return {"content": [{"type": "text", "text": json.dumps(out, indent=2)}],
                "structuredContent": out, "isError": False}

    def shutdown(self, grace_s=2.0):
        """stdin closed: let quick calls finish, cancel the rest (pi is killed)."""
        end = time.monotonic() + grace_s
        with self._lock:
            threads = list(self._threads)
        for th in threads:
            th.join(max(0.0, end - time.monotonic()))
        with self._lock:
            for ctx in self._inflight.values():
                ctx.cancel.set()
        for th in threads:
            th.join(15)

    # ------------------------------------------------------------ tools
    def _tool_table(self):
        model = _prop("string", "Model id, as list_models shows it.")
        ro = {"readOnlyHint": True, "openWorldHint": False}
        hub = {"readOnlyHint": True, "openWorldHint": True}
        t = [
            ("status", "Status", "What LlamaForge is running right now: version, engine, the "
             "loaded models (role, OpenAI-compatible endpoint, context size), models still "
             "loading, GPU memory and multi-model mode.", _schema(), ro, self.t_status),
            ("list_models", "List models", "Every model installed in LlamaForge with its "
             "status, size on disk, context size, engine and pinned build.",
             _schema(), ro, self.t_list_models),
            ("load_model", "Load a model", "Load an installed model and wait until it is "
             "serving (or fails, with the diagnosed cause). In single-model mode this replaces "
             "the running model. In multi-model mode role=worker (default) loads it beside "
             "what runs if it fits (see fit_check); evict=true may unload workers to make room; "
             "role=main replaces the user's main model - only do that when asked.",
             _schema({"model": model,
                      "role": _prop("string", "Multi-model mode only.", enum=["worker", "main"]),
                      "evict": _prop("boolean", "Multi-model mode: may unload other workers "
                                     "to make room (default false)."),
                      "wait_s": _prop("integer", "How long to wait for it (default 300).",
                                      minimum=1, maximum=1800)}, ["model"]),
             {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True,
              "openWorldHint": False}, self.t_load),
            ("unload_model", "Unload a model", "Unload a loaded model and free its memory.",
             _schema({"model": model}, ["model"]),
             {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True,
              "openWorldHint": False}, self.t_unload),
            ("fit_check", "Will it fit?", "Multi-model mode: would this model fit beside "
             "what is loaded, on which GPUs, and what would have to be evicted.",
             _schema({"model": model,
                      "role": _prop("string", "Default worker.", enum=["worker", "main"])},
                     ["model"]), ro, self.t_fit),
            ("stats", "Throughput", "Per-model token throughput and request counts, "
             "scraped from the router's /metrics.", _schema(), ro, self.t_stats),
            ("diagnose", "Diagnose a load", "Why a model failed to load (known causes with a "
             "suggested fix) plus the tail of the router log.",
             _schema({"model": model,
                      "log_lines": _prop("integer", "Log lines to return (default 60).",
                                         minimum=1, maximum=400)}, ["model"]),
             ro, self.t_diagnose),
            ("search_models", "Search Hugging Face", "Search Hugging Face for GGUF repos "
             f"(top {HUB_LIMIT}); marks repos already installed.",
             _schema({"query": _prop("string", "Search words; empty = most popular."),
                      "sort": _prop("string", "Default downloads.",
                                    enum=["downloads", "likes", "lastModified", "trending"])}),
             hub, self.t_search),
            ("list_files", "List GGUF files", "The GGUF files in a Hugging Face repo: size, "
             "shard count and whether each fits this machine's GPUs; plus mmproj (vision) "
             "and MTP (speculative) companions.",
             _schema({"repo": _prop("string", "Repo id, e.g. unsloth/Qwen3-8B-GGUF.")},
                     ["repo"]), hub, self.t_files),
            ("download_model", "Download a model", "Start downloading a GGUF (path and shards "
             "from list_files) into LlamaForge; it registers itself when done, ready for "
             "load_model. One download at a time; follow it with download_progress.",
             _schema({"repo": _prop("string", "Repo id."),
                      "path": _prop("string", "File path (the first shard of a set)."),
                      "shards": _prop("integer", "Shard count from list_files (default 1).",
                                      minimum=1, maximum=999),
                      "mmproj": _prop("string", "Also fetch this mmproj file (vision)."),
                      "mtp": _prop("string", "Also fetch this MTP file.")}, ["repo", "path"]),
             {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": True},
             self.t_download),
            ("download_progress", "Download progress", "Progress of the current or last "
             "download.", _schema(), ro, self.t_progress),
            ("ask", "Ask a local model", "One prompt to a loaded local model, no tools; "
             "returns its answer. Defaults to a loaded worker in multi-model mode, else the "
             "loaded model.",
             _schema({"prompt": _prop("string", "The prompt."),
                      "model": _prop("string", "A loaded model id (default: see above)."),
                      "system": _prop("string", "Optional system prompt."),
                      "max_tokens": _prop("integer", "Answer budget, thinking included "
                                          "(default 4096).", minimum=1, maximum=65536),
                      "temperature": _prop("number", "Sampling temperature.", minimum=0,
                                           maximum=2)}, ["prompt"]),
             {"readOnlyHint": True, "openWorldHint": False}, self.t_ask),
            ("pi_run", "Run a task with pi", "Hand a whole task to pi - Mario Zechner's "
             "open-source coding agent (MIT, https://github.com/earendil-works/pi) - running "
             "on a model loaded in LlamaForge. pi works in cwd with the chosen tools: read = "
             "read/grep/find/ls (default; changes nothing), edit = + edit/write, full = + bash "
             "(it can run commands). Blocks until pi is done and returns its final answer, the "
             "tools it used, turns and token usage. Model defaults to a loaded worker in "
             "multi-model mode, else the loaded model. Needs pi installed "
             "(Setup -> pi coding agent -> Install pi, or npm install -g "
             "@earendil-works/pi-coding-agent).",
             _schema({"task": _prop("string", "What pi should do. Self-contained: pi sees "
                                    "only this and the files in cwd."),
                      "model": _prop("string", "A loaded model id (default: see above)."),
                      "cwd": _prop("string", "Directory pi works in (default: where this MCP "
                                   "server was started, usually your project)."),
                      "tools": _prop("string", "Tool set (default read).", enum=list(pirun.TOOLSETS)),
                      "timeout_s": _prop("integer", "Give up after this long (default 600).",
                                         minimum=10, maximum=pirun.MAX_TIMEOUT)}, ["task"]),
             {"readOnlyHint": False, "destructiveHint": True, "openWorldHint": False},
             self.t_pi),
            ("forge_start", "Auto-Forge a model", "Start Auto-Forge on an installed model: it "
             "benchmarks GPU placements, KV cache types and speculative drafters in a sandbox "
             "and finds the fastest setup measured on this machine. While it runs it unloads "
             "every loaded model (they come back when it ends) and has the GPUs to itself: a "
             "quick run takes minutes to half an hour, a thorough one can take over an hour. "
             "Ask the user before starting. Follow it with forge_status.",
             _schema({"model": model,
                      "intent": _prop("string", "What the model is for; sets how speeds are "
                                      "weighed (default coding).", enum=list(FORGE_INTENTS)),
                      "ctx": _prop("integer", "Context every setup must hold (default 32768).",
                                   minimum=512, maximum=1 << 20),
                      "thorough": _prop("boolean", "Try more setups; takes longer (default false)."),
                      "allow_lossy_kv": _prop("boolean", "Also try 8/4-bit KV cache: faster or "
                                              "more context, answers may change slightly."),
                      "drop_vision": _prop("boolean", "Also try without the vision projector.")},
                     ["model"]),
             {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False},
             self.t_forge_start),
            ("forge_status", "Auto-Forge progress", "Progress of an Auto-Forge run: stage, "
             "setups done, the top five so far, and once it is done the before/after tokens a "
             "second with a postable markdown summary.",
             _schema({"model": _prop("string", "Model id (default: the last run).")}),
             ro, self.t_forge_status),
            ("forge_apply", "Apply the Auto-Forge winner", "Write the winning setup into the "
             "model's models.ini section and reload it. Only when the user asked for it; "
             "forge_undo puts the old values back.", _schema({"model": model}, ["model"]),
             {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True,
              "openWorldHint": False}, self.t_forge_apply),
            ("forge_undo", "Undo Auto-Forge", "Put back the values forge_apply replaced.",
             _schema({"model": model}, ["model"]),
             {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False},
             self.t_forge_undo),
        ]
        return [{"name": n, "title": ti, "description": d, "inputSchema": s, "annotations": a,
                 "fn": f} for n, ti, d, s, a, f in t]

    def _state(self):
        st = self.panel.get("/api/state")
        if not isinstance(st, dict):
            raise ToolError("the panel sent an unexpected /api/state")
        return st

    def t_status(self, a, ctx):
        st = self._state()
        slots = st.get("slots") or {}
        rows = _rows(st)
        return {"version": st.get("version"), "engine": st.get("active_engine"),
                "multi_model": bool(slots.get("enabled")),
                "main": (slots.get("main") or None) if slots.get("enabled") else None,
                "loaded": [_summary(m, slots) for m in rows if _is_up(m)],
                "loading": [m["id"] for m in rows if m.get("status") == "loading"],
                "installed": len(rows), "gpus": st.get("gpus") or [],
                "config_error": st.get("config_error")}

    def t_list_models(self, a, ctx):
        out = []
        for m in _rows(self._state()):
            row = {"id": m["id"], "status": m.get("status"), "failed": bool(m.get("failed")),
                   "backend": m.get("backend"), "ctx": _ctx(m.get("eff_ctx")),
                   "size_gib": m.get("file_gib"), "modalities": m.get("modalities") or ["text"]}
            if m.get("build"):
                row["build"] = m["build"]
            if _is_up(m):
                row["endpoint"] = _v1(m["endpoint"])
            out.append(row)
        return {"models": out}

    def t_load(self, a, ctx):
        mid, wait = a["model"], a.get("wait_s", 300)
        st = self._state()
        row = find_row(st, mid)
        slots = st.get("slots") or {}
        if _is_up(row):
            return dict(_summary(row, slots), status="loaded", already_loaded=True)
        body = {"model": mid, "backend": row.get("backend") or "llamacpp",
                # the panel's own default is main: an agent must not replace the user's
                "role": a.get("role", "worker"), "evict": bool(a.get("evict", False))}
        ctx.progress(f"loading {mid}")
        t0 = time.monotonic()
        out = self.panel.post("/api/models/load", body, timeout=wait + 60)
        if out.get("ok") is False or out.get("error"):
            why = out.get("error") or out.get("reason") or "load refused"
            raise ToolError(f"{mid} was not loaded: {why}")
        seen_loading, polls = False, 0
        while True:
            if ctx.cancel.is_set():
                raise ToolError("cancelled")
            st = self._state()
            row, polls = find_row(st, mid), polls + 1
            status = row.get("status")
            if _is_up(row):
                return dict(_summary(row, st.get("slots") or {}), status="loaded",
                            elapsed_s=round(time.monotonic() - t0, 1))
            if status == "loading":
                seen_loading = True
            # a failed flag left from an earlier attempt clears once the router
            # starts this one; give it a few polls to show "loading"
            elif row.get("failed") and (seen_loading or polls >= 3):
                raise ToolError(self._load_failure(mid))
            elif seen_loading:
                raise ToolError(f"{mid} stopped loading (now {status}); diagnose shows the log")
            elapsed = time.monotonic() - t0
            if elapsed >= wait:
                raise ToolError(f"{mid} is still loading after {wait}s; LlamaForge carries on - "
                                "call status to see when it is up")
            ctx.progress(f"loading {mid} ({int(elapsed)}s)")
            ctx.cancel.wait(self.poll_s)

    def _load_failure(self, mid):
        try:
            d = self.panel.get("/api/model/diag", {"model": mid}).get("diag")
        except ToolError:
            d = None
        if isinstance(d, dict) and d.get("error"):
            tip = f" Suggestion: {d['suggestion']}" if d.get("suggestion") else ""
            return f"{mid} failed to load: {d['error']}.{tip}"
        return f"{mid} failed to load; no known cause in the log - diagnose shows its tail"

    def t_unload(self, a, ctx):
        row = find_row(self._state(), a["model"])
        if row.get("status") not in ("loaded", "loading"):
            return {"model": row["id"], "unloaded": False, "note": "it was not loaded"}
        out = self.panel.post("/api/models/unload",
                              {"model": row["id"], "backend": row.get("backend") or "llamacpp"})
        if out.get("ok") is False or out.get("error"):
            raise ToolError(f"{row['id']} was not unloaded: {out.get('error') or 'refused'}")
        return {"model": row["id"], "unloaded": True}

    def t_fit(self, a, ctx):
        return self.panel.get("/api/slots/plan", {"model": a["model"],
                                                  "role": a.get("role", "worker")})

    def t_stats(self, a, ctx):
        return self.panel.get("/api/stats")

    def t_diagnose(self, a, ctx):
        mid = a["model"]
        diag = self.panel.get("/api/model/diag", {"model": mid}).get("diag")
        log = self.panel.get("/api/router/log").get("log") or ""
        lines = log.splitlines()[-a.get("log_lines", 60):]
        return {"model": mid, "diag": diag, "log_tail": "\n".join(lines)}

    @staticmethod
    def _hub_ok(out, what):
        if not isinstance(out, dict):
            raise ToolError(f"{what} failed: unexpected answer")
        if out.get("error"):
            raise ToolError(f"{what} failed: {out['error']}")
        return out

    def t_search(self, a, ctx):
        out = self._hub_ok(self.panel.post("/api/hub/search", {
            "query": a.get("query", ""), "sort": a.get("sort", "downloads")}, timeout=60),
            "Hugging Face search")
        have = set(out.get("installed") or [])
        res = [{"repo": r.get("repo"), "downloads": r.get("downloads"), "likes": r.get("likes"),
                "updated": r.get("updated"), "gated": bool(r.get("gated")),
                "installed": r.get("repo") in have}
               for r in (out.get("results") or [])[:HUB_LIMIT]]
        return {"results": res, "vram_mib": out.get("vram_mib")}

    def t_files(self, a, ctx):
        out = self._hub_ok(self.panel.post("/api/hub/files", {"repo": a["repo"]}, timeout=120),
                           "listing the repo")
        files = []
        for f in out.get("files") or []:
            row = {"path": f.get("path"), "size_gib": round((f.get("size") or 0) / 1024**3, 2),
                   "shards": f.get("shards", 1), "fit": f.get("fit")}
            if f.get("mtp"):
                row["mtp"] = f["mtp"]
            files.append(row)
        return {"repo": a["repo"], "files": files,
                "mmproj": [m.get("path") for m in out.get("mmproj") or []],
                "mtp": [m.get("path") for m in out.get("mtp") or []]}

    def t_download(self, a, ctx):
        body = {k: a[k] for k in ("repo", "path", "shards", "mmproj", "mtp") if k in a}
        out = self._hub_ok(self.panel.post("/api/hub/download", body), "the download")
        if not out.get("started"):
            raise ToolError("another download is running; download_progress shows it")
        return {"started": True, "dest": out.get("dest"), "repo": a["repo"], "path": a["path"]}

    def t_progress(self, a, ctx):
        return self.panel.get("/api/hub/progress")

    def t_ask(self, a, ctx):
        row = pick_model(self._state(), a.get("model"))
        budget = a.get("max_tokens", 4096)
        msgs = ([{"role": "system", "content": a["system"]}] if a.get("system") else [])
        msgs.append({"role": "user", "content": a["prompt"]})
        body = {"model": row["id"], "messages": msgs, "max_tokens": budget, "stream": False}
        if "temperature" in a:
            body["temperature"] = a["temperature"]
        t0 = time.monotonic()
        key = network_policy.effective_key(self.cfg())
        out = self.chat(_v1(row["endpoint"]) + "/chat/completions", body, key, 900)
        choice = ((out or {}).get("choices") or [{}])[0] or {}
        msg = choice.get("message") or {}
        text = msg.get("content") if isinstance(msg.get("content"), str) else ""
        text = text.strip()
        if not text:
            if choice.get("finish_reason") == "length":
                raise ToolError(f"{row['id']} spent all {budget} max_tokens before answering "
                                "(it was still thinking); ask again with a larger max_tokens")
            raise ToolError(f"{row['id']} returned an empty answer")
        return _Answer(text, {"model": row["id"], "finish_reason": choice.get("finish_reason"),
                              "usage": out.get("usage") or {},
                              "elapsed_s": round(time.monotonic() - t0, 1)})

    def t_pi(self, a, ctx):
        row = pick_model(self._state(), a.get("model"))
        c = self.cfg()
        ctx.progress(f"pi is starting on {row['id']}")
        r = self.runner(task=a["task"], model=row["id"], endpoint=row["endpoint"],
                        key=network_policy.effective_key(c), cwd=a.get("cwd") or os.getcwd(),
                        tools=a.get("tools", "read"),
                        timeout=a.get("timeout_s", pirun.DEFAULT_TIMEOUT),
                        ctx=row.get("eff_ctx"), cancel=ctx.cancel, progress=ctx.progress,
                        pi_bin=c.get("pi_bin") or "")
        data = {k: v for k, v in r.items() if k != "text"}
        text = r.get("text") or ""
        if not r.get("ok"):
            msg = f"pi_run failed: {r.get('error') or 'unknown error'}"
            return _Answer(msg + (f"\n\nPartial output:\n{text}" if text else ""), data, error=True)
        return _Answer(text or "(pi finished without a final message)", data)

    def t_forge_start(self, a, ctx):
        body = {"model": a["model"], "intent": a.get("intent", "coding"),
                "ctx": a.get("ctx", 32768), "time": "thorough" if a.get("thorough") else "quick",
                "allow_loss": bool(a.get("allow_lossy_kv")), "drop_vision": bool(a.get("drop_vision"))}
        out = self.panel.post("/api/forge/start", body)
        return {"started": bool(out.get("ok")), "model": a["model"], "req": out.get("req"),
                "next": "call forge_status every minute or two until running is false"}

    def t_forge_status(self, a, ctx):
        st = self.panel.get("/api/forge/status", {"model": a["model"]} if a.get("model") else None)
        board = [{k: r[k] for k in _BOARD_KEYS if k in r} for r in (st.get("leaderboard") or [])]
        out = {k: st.get(k) for k in ("model", "stage", "running", "progress", "loading",
                                      "error", "req")}
        out["leaderboard"] = board[:FORGE_TOP]
        out["failed"] = len(st.get("failed") or [])
        out["applied"] = bool(st.get("applied"))
        res = st.get("result") or {}
        if not st.get("running") and res.get("winner"):
            try:      # share keeps paths, keys and hosts out: safe to hand to the agent
                sh = self.panel.get("/api/forge/share", {"model": st.get("model") or a.get("model")})
                out["share"] = {k: sh.get(k) for k in ("before", "after", "x", "setup", "settings",
                                                       "lossless", "drops_vision", "markdown")}
            except ToolError:
                pass
        elif not st.get("running") and res:
            out["no_winner"] = res.get("reason") or "no setup beat the current one"
        return out

    def t_forge_apply(self, a, ctx):
        out = self.panel.post("/api/forge/apply", {"model": a["model"]})
        return {"applied": bool(out.get("ok")), "updates": out.get("updates") or {}}

    def t_forge_undo(self, a, ctx):
        out = self.panel.post("/api/forge/undo", {"model": a["model"]})
        return {"undone": bool(out.get("ok"))}


# ---------------------------------------------------------------- stdio
def serve(inp, out, server=None):
    """Read requests line by line until EOF; one JSON message per output line."""
    lock = threading.Lock()

    def send(msg):
        line = json.dumps(msg, ensure_ascii=False, separators=(",", ":"))
        with lock:
            try:
                out.write(line + "\n")
                out.flush()
            except (OSError, ValueError):   # client gone
                pass

    server = server or Server()
    server.send = send
    for line in inp:
        try:
            resp = server.handle_line(line)
        except Exception:
            _log(traceback.format_exc())
            resp = None
        if resp is not None:
            send(resp)
    server.shutdown()


def _console_python(exe):
    """pythonw has no stdio of its own; MCP clients need the console build."""
    d, name = os.path.split(exe or "")
    if name.lower().startswith("pythonw"):
        cand = os.path.join(d, name[:6] + name[7:])
        if os.path.isfile(cand):
            return cand
    return exe


def setup_info(python=None, script=None):
    """Ready-to-paste client configs for this install. No secrets: the server
    reads the router key from config.json itself."""
    python = _console_python(python or sys.executable)
    script = script or os.path.abspath(__file__)
    q = lambda s: '"' + s.replace('"', '\\"') + '"'
    # TOML basic strings escape like JSON strings
    toml = "\n".join([
        "[mcp_servers.llamaforge]",
        f"command = {json.dumps(python)}",
        f"args = [{json.dumps(script)}]",
        "# pi_run can work for minutes; Codex's default tool timeout is 60s",
        "tool_timeout_sec = 900",
    ])
    return {
        "python": python, "script": script,
        "claude": f"claude mcp add --scope user llamaforge -- {q(python)} {q(script)}",
        "codex_toml": toml,
        "json": json.dumps({"mcpServers": {"llamaforge": {"command": python, "args": [script]}}},
                           indent=2),
        "tools": [t["name"] for t in Server(panel=object(), cfg=dict).tools],
    }


def main():
    for s in (sys.stdin, sys.stdout):
        try:
            s.reconfigure(encoding="utf-8", newline="\n" if s is sys.stdout else None)
        except (AttributeError, ValueError):
            pass
    out = sys.stdout
    sys.stdout = sys.stderr      # a stray print() must never corrupt the protocol
    serve(sys.stdin, out)


if __name__ == "__main__":
    main()
