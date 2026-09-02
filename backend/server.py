"""LlamaForge backend: one local HTTP server that powers the whole GUI.

This module is only plumbing - parse the request, decide whether to trust it,
look the path up in routes.py, write the response. All API behaviour lives in
routes.py, where each handler is a plain function a test can call directly.

Two exceptions stay here: the Anthropic and OpenAI streaming proxies, which
write SSE to the socket themselves rather than returning a payload.

Pure Python stdlib.
"""
import json, os, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import config, wiki, anthropic_shim
import routes
from routes import ApiError, Req

# Everything else is reached as routes.<name> rather than imported by name: a
# `from routes import cfg` binds the function object here, so a test patching
# routes.cfg would not affect this module.

# ---------------------------------------------------------------- trust model
#
# The dashboard binds 127.0.0.1, which keeps it off the network but NOT out of
# reach: every page the user browses can send requests to it, and these routes
# rebuild llama.cpp, install packages and edit configuration. Two checks close
# that gap.
#
# 1. Origin. A cross-site request either carries an Origin naming the attacker's
#    site, or (for form posts) carries none while declaring a form content type.
#    Same-origin fetches from our own page always send an Origin we recognise.
# 2. Host. Blocks DNS rebinding, where an attacker's hostname is re-pointed at
#    127.0.0.1 so the browser treats their page as same-origin with ours.
#
# Content-Type matters because a <form> can only send three types, none of them
# application/json - requiring JSON on state-changing routes means an attacker's
# page cannot forge one without a preflight it will fail.
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "[::1]", "::1"}
MAX_MANAGEMENT_JSON_BODY_BYTES = 4 * 1024 * 1024
MAX_PROXY_JSON_BODY_BYTES = 64 * 1024 * 1024
_BODY_ERROR = object()


def _post_body_limit(path):
    if path in ("/v1/messages", "/v1/chat/completions"):
        return MAX_PROXY_JSON_BODY_BYTES
    return MAX_MANAGEMENT_JSON_BODY_BYTES


def _host_ok(host_header, port):
    """True when the Host names this loopback service."""
    if not host_header:
        return False
    host = host_header.strip()
    if host.startswith("["):                       # [::1]:8090
        addr, _, tail = host.partition("]")
        addr, got_port = addr + "]", tail.lstrip(":")
    else:
        addr, _, got_port = host.partition(":")
    if got_port and got_port != str(port):
        return False
    return addr in ALLOWED_HOSTS


def _origin_ok(origin, port):
    """True when Origin is absent (a same-origin GET, curl, or an agent client)
    or names this same service."""
    if not origin:
        return True
    if origin == "null":                           # sandboxed iframe / file://
        return False
    try:
        u = urllib.parse.urlsplit(origin)
    except ValueError:
        return False
    if u.scheme not in ("http", "https"):
        return False
    return _host_ok(u.netloc, port)


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a): pass

    # ------------------------------------------------------------- responding
    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)): body = json.dumps(body).encode()
        elif isinstance(body, str):        body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        if ctype.startswith("application/json"):
            self.send_header("Cache-Control", "no-store")
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, name, ctype):
        path = os.path.join(routes.WEB, name)
        if not os.path.exists(path):
            return self._send(404, {"error": "not found"})
        with open(path, "rb") as f:
            self._send(200, f.read(), ctype)

    def _headers_lower(self):
        return {k.lower(): v for k, v in self.headers.items()}

    def _body_error(self, code, message):
        self.close_connection = True
        self._send(code, {"error": message})
        return _BODY_ERROR

    def _read_json_body(self, path):
        if self.headers.get("Transfer-Encoding") is not None:
            return self._body_error(400, "Transfer-Encoding is not supported")
        lengths = self.headers.get_all("Content-Length") or []
        if not lengths:
            return self._body_error(411, "Content-Length is required")
        if len(lengths) != 1:
            return self._body_error(400, "invalid Content-Length")
        raw = lengths[0]
        if not raw or not raw.isascii() or not raw.isdecimal():
            return self._body_error(400, "invalid Content-Length")
        size = int(raw)
        if size > _post_body_limit(path):
            return self._body_error(413, "request body too large")
        data = self.rfile.read(size)
        if len(data) != size:
            return self._body_error(400, "incomplete request body")
        try:
            body = json.loads(data or b"{}")
        except (UnicodeDecodeError, ValueError):
            return self._body_error(400, "invalid JSON body")
        if not isinstance(body, dict):
            return self._body_error(400, "body must be a JSON object")
        return body

    # ---------------------------------------------------------------- guards
    def _guard(self, method):
        """Reject anything that isn't this dashboard's own page talking to it.
        Returns True when the request has been answered and must not proceed."""
        port = routes.cfg()["panel_port"]
        if not _host_ok(self.headers.get("Host", ""), port):
            if method == "POST":
                self.close_connection = True
            self._send(403, {"error": "bad Host header"})
            return True
        if not _origin_ok(self.headers.get("Origin", ""), port):
            if method == "POST":
                self.close_connection = True
            self._send(403, {"error": "cross-origin request refused"})
            return True
        if method == "POST":
            # A cross-site <form> can only send urlencoded/multipart/text-plain.
            # Requiring JSON means a forged post needs a preflight it can't pass.
            ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype and ctype != "application/json":
                self.close_connection = True
                self._send(415, {"error": "Content-Type must be application/json"})
                return True
        return False

    def _vllm_gate(self, p):
        """vLLM rides on WSL2; short-circuit its routes on Linux/macOS."""
        if p.startswith("/api/vllm/") and not routes.VLLM_SUPPORTED:
            if p == "/api/vllm/setup":   # the Setup tab probes this one
                self._send(200, {"supported": False, "wsl": {"present": False},
                                 "distros": [], "gpu": {"present": False},
                                 "vllm": {"present": False, "version": ""},
                                 "setup_job": {"running": False}, "setup_log": ""})
            else:
                self._send(400, {"error": "vLLM backend requires Windows + WSL2"})
            return True
        return False

    # -------------------------------------------------------------- dispatch
    def _run(self, handler, req):
        """Call a route handler and write whatever it returns."""
        try:
            result = handler(req)
        except ApiError as e:
            return self._send(e.status, {"error": e.message})
        except Exception as e:                   # a bug in one route must not
            return self._send(500, {"error": str(e)})   # take the dashboard down
        if len(result) == 3:
            status, payload, ctype = result
            return self._send(status, payload, ctype)
        status, payload = result
        return self._send(status, payload)

    def do_GET(self):
        if self._guard("GET"):
            return
        p, _, query = self.path.partition("?")
        qs = {k: v[0] for k, v in urllib.parse.parse_qs(query).items()} if query else {}
        if self._vllm_gate(p):
            return

        if p in ("/", "/index.html"):
            return self._file("index.html", "text/html; charset=utf-8")
        if p.startswith("/web/"):                # ES modules under web/
            return self._static_module(p[len("/web/"):])
        if p.startswith("/docs/img/"):
            return self._docs_image(p[len("/docs/img/"):])

        handler = routes.GET_ROUTES.get(p)
        if not handler:
            return self._send(404, {"error": "not found"})
        return self._run(handler, Req(qs=qs, headers=self._headers_lower(), path=p))

    def do_POST(self):
        if self._guard("POST"):
            return
        p = self.path.split("?")[0]
        body = self._read_json_body(p)
        if body is _BODY_ERROR:
            return
        if self._vllm_gate(p):
            return

        headers = self._headers_lower()
        # Streaming routes own their response, so they can't go in the table.
        if p == "/v1/messages":
            if not routes.cfg().get("anthropic_shim_enabled", True):
                return self._send(404, {"error": "not found"})
            if not routes._shim_auth_ok(headers):
                st, err = anthropic_shim.anthropic_error(
                    401, "authentication_error", "invalid x-api-key")
                return self._send(st, err)
            if body.get("stream"):
                return self._anthropic_stream(body)
            status, out = routes._anthropic_messages(body, headers)
            return self._send(status, out)
        if p == "/v1/chat/completions":
            if not routes._shim_auth_ok(headers):
                return self._send(401, {"error": {"message": "invalid key",
                                                  "type": "authentication_error"}})
            fwd = routes._inject_openai_system(
                body, wiki.compose(wiki.active_profile(body.get("model", ""))))
            if fwd.get("stream"):
                fwd.setdefault("stream_options", {"include_usage": True})
                return self._openai_proxy_stream(fwd)
            status, data = routes._router_openai(fwd, stream=False)
            return self._send(status, data)

        handler = routes.POST_ROUTES.get(p)
        if not handler:
            return self._send(404, {"error": "not found"})
        return self._run(handler, Req(body=body, headers=headers, path=p))

    # ------------------------------------------------------- static payloads
    _MODULE_TYPES = {".js": "application/javascript; charset=utf-8",
                     ".css": "text/css; charset=utf-8"}

    def _static_module(self, rel):
        """Serve web/**.js so the frontend can use ES modules. Confined to routes.WEB."""
        if not rel or ".." in rel or os.path.isabs(rel) or ":" in rel:
            return self._send(404, {"error": "not found"})
        base = os.path.realpath(routes.WEB)
        full = os.path.realpath(os.path.join(base, *rel.split("/")))
        if os.path.commonpath([base, full]) != base or not os.path.isfile(full):
            return self._send(404, {"error": "not found"})
        ctype = self._MODULE_TYPES.get(os.path.splitext(full)[1].lower())
        if not ctype:
            return self._send(404, {"error": "not found"})
        with open(full, "rb") as f:
            return self._send(200, f.read(), ctype)

    _IMG_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                  ".gif": "image/gif", ".svg": "image/svg+xml", ".webp": "image/webp"}

    def _docs_image(self, name):
        from routes import docs
        try:
            path = docs._safe_img(name)
        except ValueError:
            return self._send(404, {"error": "bad image"})
        if not os.path.exists(path):
            return self._send(404, {"error": "not found"})
        ctype = self._IMG_TYPES.get(os.path.splitext(path)[1].lower(),
                                    "application/octet-stream")
        with open(path, "rb") as f:
            return self._send(200, f.read(), ctype)

    # ----------------------------------------------------------- SSE proxies
    def _begin_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        # length is unknown up front, so the connection delimits the body
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()

    def _writer(self):
        def write(b):
            try:
                self.wfile.write(b)
                self.wfile.flush()
            except Exception:
                pass
        return write

    def _anthropic_stream(self, body):
        model = routes._resolve_anthropic_model(body.get("model", ""))
        body = routes._inject_anthropic_system(body, wiki.compose(wiki.active_profile(model)))
        oai = anthropic_shim.to_openai_request({**body, "model": model, "stream": True})
        oai["stream_options"] = {"include_usage": True}
        status, resp = routes._router_openai(oai, stream=True)
        self._begin_stream()
        routes._write_anthropic_stream(self._writer(), model, status, resp)
        if hasattr(resp, "close"):
            try:
                resp.close()
            except Exception:
                pass

    def _openai_proxy_stream(self, body):
        status, resp = routes._router_openai(body, stream=True)
        self._begin_stream()
        write = self._writer()
        if status >= 400:
            write(("data: " + json.dumps(resp) + "\n\n").encode())
            return
        for line in resp:
            write(line if line.endswith(b"\n") else line + b"\n")
        if hasattr(resp, "close"):
            try:
                resp.close()
            except Exception:
                pass


def _tray_counts():
    """(loaded, total) model counts for the optional tray tooltip."""
    st, data = routes.router("/models", timeout=3)
    rows = [m for m in data.get("data", []) if m.get("id") != "default"] if st == 200 else []
    loaded = sum(1 for m in rows if m.get("status", {}).get("value") == "loaded")
    return loaded, len(rows)


def _auto_load(model_id):
    """Load a favourite model once the router answers /models. Runs in the
    background so a slow/absent router never delays the dashboard."""
    import time
    for _ in range(60):                       # wait up to ~60s for the router
        st, data = routes.router("/models", timeout=3)
        if st == 200:
            known = {m.get("id") for m in data.get("data", [])}
            if model_id not in known and model_id not in config.read_sections():
                return                          # unknown model id - nothing to load
            routes.router("/models/load", "POST", {"model": model_id})
            return
        time.sleep(1)


def main():
    import stats
    config.migrate()
    c = routes.cfg()
    port = c["panel_port"]
    print(f"LlamaForge -> http://127.0.0.1:{port}")
    if config.LOAD_ERROR:
        print(f"  WARNING: {config.LOAD_ERROR}")
        print(f"  previous contents saved to {config.CONFIG}.corrupt")
    try:                    # the repo ships no models.ini; llama-server needs one
        if config.ensure_models_ini():
            print(f"  created {config.ini_path()}")
    except OSError as e:    # unwritable path: say so, the router will fail next
        print(f"  WARNING: could not create models.ini ({e})")
    try:                    # backfill ctx-size defaults, then nudge the router
        if config.apply_ctx_defaults().get("changed"):
            routes.router("/models?reload=1")
    except Exception:
        pass
    stats.TRACKER.start()   # background usage poller
    try:                    # optional tray icon (no-op unless pystray+pillow present)
        import tray
        if tray.available():
            tray.start(port, _tray_counts)
    except Exception:
        pass
    if c.get("auto_load_model"):
        import threading
        threading.Thread(target=_auto_load, args=(c["auto_load_model"],),
                         daemon=True, name="auto-load").start()
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()


if __name__ == "__main__":
    main()
