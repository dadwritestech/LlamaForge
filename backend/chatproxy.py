"""Serve llama.cpp's own chat UI from LlamaForge, on its own origin.

llama-server ships a full chat client (markdown, reasoning, attachments, model
switching in router mode); every llama.cpp update improves our Chat tab for
free. This module is a small reverse proxy in front of it, listening on
`chat_port` (default 8091), which the panel embeds in an iframe.

Two reasons it isn't just an iframe of the router port, and isn't a path on
the panel either:

* The router may require an API key, which is never sent to browser JS. The
  proxy adds it server-side, so the chat works with no key prompt.
* The chat renders untrusted model output. On the panel's origin, any XSS in
  it could drive every /api route (rebuild, install packages, rebind the
  router to the LAN). On its own port it is a different origin: the panel's
  port-exact Origin check refuses it, and this listener serves nothing but
  the chat.
"""
import json, urllib.error, urllib.parse, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ALLOWED_HOSTS = {"127.0.0.1", "localhost", "[::1]", "::1"}
MAX_BODY_BYTES = 64 * 1024 * 1024         # image attachments travel inline
# Request headers worth passing upstream. Everything else - Cookie, the
# browser's own Authorization, Origin, hop-by-hop headers - stays behind.
_FWD_REQ = ("content-type", "accept", "if-none-match", "if-modified-since",
            "last-event-id", "cache-control")
# Response headers worth passing back. Content-Length is handled separately.
_FWD_RESP = ("content-type", "cache-control", "etag", "last-modified",
             "content-encoding", "cross-origin-embedder-policy",
             "cross-origin-opener-policy")
# A slow prompt-processing phase can be silent for minutes before the first
# token, and /models/sse idles between events.
UPSTREAM_TIMEOUT = 3600

ROUTER_DOWN_HTML = b"""<!doctype html><meta charset="utf-8">
<body style="background:#111;color:#bbb;font:14px ui-monospace,monospace;padding:40px">
<h3 style="color:#ffa630">The llama.cpp router isn't running</h3>
<p>Start it from the LlamaForge panel (Models tab), then reload this page.</p></body>"""


# ------------------------------------------------------------------ pure parts

def upstream_path(path):
    """The router path for a request path, or None if it must not be relayed."""
    if not path.startswith("/") or path.startswith("//") or "\\" in path:
        return None
    return path


def upstream_headers(client_headers, api_key):
    """client_headers: lower-cased dict. Returns headers for the router."""
    out = {k: v for k, v in client_headers.items() if k in _FWD_REQ}
    if api_key:
        out["authorization"] = "Bearer " + api_key
    return out


def response_headers(items, panel_origin):
    """Headers to send back, from the upstream's (name, value) pairs."""
    out = [(k, v) for k, v in items if k.lower() in _FWD_RESP]
    out.append(("X-Content-Type-Options", "nosniff"))
    # Only the panel may frame the chat (clickjacking).
    out.append(("Content-Security-Policy", f"frame-ancestors {panel_origin}"))
    return out


def host_ok(host_header, port):
    host = (host_header or "").strip()
    if not host:
        return False
    if host.startswith("["):
        addr, _, tail = host.partition("]")
        addr, got = addr + "]", tail.lstrip(":")
    else:
        addr, _, got = host.partition(":")
    return (not got or got == str(port)) and addr in ALLOWED_HOSTS


def origin_ok(origin, port):
    """Absent (plain navigation / same-origin GET) or this chat origin only."""
    if not origin:
        return True
    try:
        u = urllib.parse.urlsplit(origin)
    except ValueError:
        return False
    return u.scheme in ("http", "https") and host_ok(u.netloc, port)


def open_upstream(port, path, method, headers, body=None):
    """Returns (status, header_items, response_or_bytes). HTTP errors are
    returned for relaying; URLError (router down) propagates."""
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=body,
                                 method=method, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT)
    except urllib.error.HTTPError as e:
        return e.code, list(e.headers.items()), e.read()
    return resp.status, list(resp.headers.items()), resp


def pump(resp, write, chunk=65536):
    """Copy an upstream body as bytes arrive. read1() returns whatever has
    arrived (chunked decoding included); read(n) would wait for n bytes and
    turn a token stream into stutters. Stops when the client goes away."""
    read = getattr(resp, "read1", None) or resp.read
    try:
        while True:
            b = read(chunk)
            if not b or not write(b):
                break
    finally:
        try:
            resp.close()
        except Exception:
            pass


# ------------------------------------------------------------------ listener

def make_handler(get_cfg):
    """get_cfg() -> config dict; read per request so port/key changes apply."""

    class ChatHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _plain(self, code, body, ctype="application/json"):
            if isinstance(body, dict):
                body = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _guard(self, c, method):
            port = c.get("chat_port", 8091)
            if not host_ok(self.headers.get("Host"), port):
                self.close_connection = True
                self._plain(403, {"error": "bad Host header"})
                return True
            if not origin_ok(self.headers.get("Origin"), port):
                self.close_connection = True
                self._plain(403, {"error": "cross-origin request refused"})
                return True
            if method == "POST":
                ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
                if ctype and ctype != "application/json":
                    self.close_connection = True
                    self._plain(415, {"error": "Content-Type must be application/json"})
                    return True
            return False

        def _body(self):
            if self.headers.get("Transfer-Encoding") is not None:
                return None
            raw = self.headers.get("Content-Length") or "0"
            if not raw.isascii() or not raw.isdecimal() or int(raw) > MAX_BODY_BYTES:
                return None
            data = self.rfile.read(int(raw))
            return data if len(data) == int(raw) else None

        def _relay(self, method):
            c = get_cfg()
            if self._guard(c, method):
                return
            path = upstream_path(self.path)
            if path is None:
                return self._plain(404, {"error": "not found"})
            body = None
            if method == "POST":
                body = self._body()
                if body is None:
                    self.close_connection = True
                    return self._plain(400, {"error": "bad request body"})
            headers = upstream_headers({k.lower(): v for k, v in self.headers.items()},
                                       c.get("router_api_key", ""))
            try:
                status, items, resp = open_upstream(c["router_port"], path, method,
                                                    headers, body)
            except Exception:
                return self._plain(502, ROUTER_DOWN_HTML, "text/html; charset=utf-8")
            self.send_response(status)
            panel = f"http://127.0.0.1:{c['panel_port']} http://localhost:{c['panel_port']}"
            for k, v in response_headers(items, panel):
                self.send_header(k, v)
            if isinstance(resp, bytes):
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
                return
            length = next((v for k, v in items if k.lower() == "content-length"), None)
            if length is not None:
                self.send_header("Content-Length", length)
            else:                  # streamed (SSE): the connection delimits it
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()

            def write(b):
                try:
                    self.wfile.write(b)
                    self.wfile.flush()
                    return True
                except OSError:
                    return False
            pump(resp, write)

        def do_GET(self):
            self._relay("GET")

        def do_POST(self):
            self._relay("POST")

    return ChatHandler


def serve(get_cfg):
    """Start the chat listener on a daemon thread. Returns the server, or None
    when the port is taken (the Chat tab then says so; the panel runs on)."""
    import threading
    port = get_cfg().get("chat_port", 8091)
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(get_cfg))
    except OSError as e:
        print(f"  WARNING: chat proxy could not bind 127.0.0.1:{port} ({e})")
        return None
    threading.Thread(target=httpd.serve_forever, daemon=True, name="chat-proxy").start()
    return httpd
