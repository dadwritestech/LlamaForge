"""Phone remote: chat with this box's models from a phone on the same network.

Off by default. When the user turns it on in the panel, this listener binds
0.0.0.0:`phone_port` (8092) and serves exactly three things:

* GET  /             a self-contained chat page (no data in it)
* GET  /api/status   models (id, loaded) and GPUs (name, memory, load)
* POST /api/chat     a plain chat, streamed from the router
* POST /api/talk     a recording in, what it heard and the answer out (spoken
                     on the PC by the Voice tab's engine when that is set up)

The panel itself stays loopback-only: it can rebuild llama.cpp and install
packages, which is not something a phone link should reach. The two endpoints
want `Authorization: Bearer <phone_token>`. The token travels in the QR code's
URL fragment (#k=...), which browsers never send to a server, and the page
moves it into localStorage. Rotating the token in the panel locks out every
phone that has the old one. The router key is added server-side, as the Chat
tab's proxy does, and the chat body is rebuilt from an allowlist: model,
plain-text messages, stream.

A phone browser hands a page the mic only over HTTPS. The listener stays plain
HTTP; `tailscale serve --bg <port>` puts the tailnet's own certificate in front
of it, and the dialog offers that https://<name>.ts.net link when this PC is on
a tailnet.
"""
import hmac, ipaddress, json, os, re, secrets, shutil, socket, subprocess, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import chatproxy

PAGE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "phone.html")
MAX_BODY_BYTES = 512 * 1024
TALK_MAX_BYTES = 12 * 1024 * 1024     # base64 WAV: about 6 minutes at 16 kHz, far past a turn
MAX_MESSAGES = 200
ROLES = ("system", "user", "assistant")
TAILNET = ipaddress.ip_network("100.64.0.0/10")   # Tailscale's CGNAT range
CSP = ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
       "connect-src 'self'; img-src data:; media-src blob:; base-uri 'none'; form-action 'none'; "
       "frame-ancestors 'none'")


# ------------------------------------------------------------------ pure parts

def new_token():
    return secrets.token_urlsafe(16)


def token_ok(auth_header, token):
    """Constant-time check of an `Authorization: Bearer ...` header."""
    if not token or not auth_header or not auth_header.startswith("Bearer "):
        return False
    return hmac.compare_digest(auth_header[7:].encode(), token.encode())


def sanitize_chat(body):
    """The router body for a phone chat, or None. Everything but the model and
    plain-text system/user/assistant messages is dropped."""
    if not isinstance(body, dict):
        return None
    model, msgs = body.get("model"), body.get("messages")
    if not isinstance(model, str) or not model or len(model) > 256 or not isinstance(msgs, list):
        return None
    clean = [{"role": m["role"], "content": m["content"]} for m in msgs[-MAX_MESSAGES:]
             if isinstance(m, dict) and m.get("role") in ROLES and isinstance(m.get("content"), str)]
    if not clean:
        return None
    return {"model": model, "messages": clean, "stream": True,
            "stream_options": {"include_usage": True}}


def sanitize_talk(body):
    """{audio, model, history} for routes.phone_talk, or None. The history keeps
    only plain user/assistant text; the audio itself is checked further on."""
    if not isinstance(body, dict):
        return None
    audio, model, hist = body.get("audio"), body.get("model", ""), body.get("history", [])
    if (not isinstance(audio, str) or not audio or not isinstance(model, str)
            or len(model) > 256 or not isinstance(hist, list)):
        return None
    keep = [{"role": m["role"], "content": m["content"]} for m in hist[-MAX_MESSAGES:]
            if isinstance(m, dict) and m.get("role") in ("user", "assistant")
            and isinstance(m.get("content"), str)]
    return {"audio": audio, "model": model, "history": keep}


def status(code, models, gpus):
    rows = models.get("data", []) if code == 200 and isinstance(models, dict) else []
    return {"models": [{"id": m.get("id"), "loaded": (m.get("status") or {}).get("value") == "loaded"}
                       for m in rows if isinstance(m, dict) and m.get("id")],
            "gpus": [{k: g.get(k) for k in ("name", "used", "total", "util")} for g in gpus or []]}


def is_tailnet(ip):
    try:
        return ipaddress.ip_address(ip) in TAILNET
    except ValueError:
        return False


def tailnet_ips():
    """This box's Tailscale addresses, if it is on a tailnet."""
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
    except OSError:
        return []
    return sorted({i[4][0] for i in infos if is_tailnet(i[4][0])})


_DNS = re.compile(r"^[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)+$")


def ts_host(status_json):
    """This PC's MagicDNS name from `tailscale status --json`, or ""."""
    try:
        st = json.loads(status_json)
        name = (st.get("Self") or {}).get("DNSName") or ""
    except (ValueError, AttributeError, TypeError):
        return ""
    name = name.rstrip(".") if isinstance(name, str) else ""
    return name if st.get("BackendState") == "Running" and _DNS.match(name) else ""


def _find_tailscale():
    for c in (shutil.which("tailscale"), r"C:\Program Files\Tailscale\tailscale.exe",
              "/Applications/Tailscale.app/Contents/MacOS/Tailscale"):
        if c and os.path.isfile(c):
            return c
    return ""


def tailscale_host(run=subprocess.run, find=_find_tailscale):
    """Ask the Tailscale CLI, when there is one, for this PC's tailnet name."""
    exe = find()
    if not exe:
        return ""
    try:
        r = run([exe, "status", "--json"], capture_output=True, text=True, timeout=4,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return ""
    return ts_host(r.stdout) if r.returncode == 0 else ""


def urls(port, token, lan, tailnet, https_host=""):
    out = [{"kind": "lan", "url": f"http://{lan}:{port}/#k={token}"}] if lan else []
    out += [{"kind": "tailnet", "url": f"http://{ip}:{port}/#k={token}"} for ip in tailnet]
    return out + ([{"kind": "https", "url": f"https://{https_host}/#k={token}"}] if https_host else [])


# ------------------------------------------------------------------ listener

class RouterDeps:
    """What the listener needs from the rest of the backend. Tests pass fakes."""

    def models(self):
        import routes
        return routes.router("/models")

    def gpus(self):
        import routes
        return routes._GPU_TELEMETRY.get()

    def chat(self, body):
        import routes
        return routes._router_openai(body, stream=True)

    def talk(self, body):
        import routes
        return routes.phone_talk(body)


def make_handler(get_cfg, deps):

    class PhoneHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            if isinstance(body, dict):
                body = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            if ctype.startswith("text/html"):
                self.send_header("Content-Security-Policy", CSP)
            self.end_headers()
            self.wfile.write(body)

        def _authed(self, c):
            if token_ok(self.headers.get("Authorization"), c.get("phone_token", "")):
                return True
            self._send(401, {"error": "this link is out of date: scan the QR code in the panel again"})
            return False

        def do_GET(self):
            c = get_cfg()
            if not c.get("phone_enabled"):
                return self._send(404, {"error": "not found"})
            path = self.path.split("?", 1)[0]
            if path == "/":
                try:
                    with open(PAGE, "rb") as f:
                        return self._send(200, f.read(), "text/html; charset=utf-8")
                except OSError:
                    return self._send(500, {"error": "phone page missing"})
            if path == "/api/status":
                if self._authed(c):
                    code, models = deps.models()
                    try:
                        gpus = deps.gpus()
                    except Exception:
                        gpus = []
                    self._send(200, status(code, models, gpus))
                return
            self._send(404, {"error": "not found"})

        def do_POST(self):
            c = get_cfg()
            path = self.path.split("?", 1)[0]
            if not c.get("phone_enabled") or path not in ("/api/chat", "/api/talk"):
                self.close_connection = True
                return self._send(404, {"error": "not found"})
            if not self._authed(c):
                self.close_connection = True
                return
            ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype != "application/json":
                self.close_connection = True
                return self._send(415, {"error": "Content-Type must be application/json"})
            raw = self.headers.get("Content-Length") or ""
            if (self.headers.get("Transfer-Encoding") is not None or not raw.isascii()
                    or not raw.isdecimal()
                    or int(raw) > (TALK_MAX_BYTES if path == "/api/talk" else MAX_BODY_BYTES)):
                self.close_connection = True
                return self._send(400, {"error": "bad request body"})
            try:
                got = json.loads(self.rfile.read(int(raw)) or b"null")
            except ValueError:
                got = None
            if path == "/api/talk":
                body = sanitize_talk(got)
                if body is None:
                    return self._send(400, {"error": "send a recording"})
                code, out = deps.talk(body)
                return self._send(code, out)
            body = sanitize_chat(got)
            if body is None:
                return self._send(400, {"error": "send a model and at least one message"})
            code, resp = deps.chat(body)
            if isinstance(resp, dict):
                return self._send(code if code >= 400 else 502, resp)
            self.send_response(code)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
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
            chatproxy.pump(resp, write)

    return PhoneHandler


_LOCK = threading.Lock()
_SERVER = None


def running():
    return _SERVER is not None


def start(get_cfg, deps=None, host="0.0.0.0"):
    """Start the listener (once). Returns the server, or None when the port is
    taken. Every request re-reads the config, so a rotated token or a
    disabled remote applies at once."""
    global _SERVER
    with _LOCK:
        if _SERVER is not None:
            return _SERVER
        port = get_cfg().get("phone_port", 8092)
        try:
            srv = ThreadingHTTPServer((host, port), make_handler(get_cfg, deps or RouterDeps()))
        except OSError as e:
            print(f"  WARNING: phone remote could not bind {host}:{port} ({e})")
            return None
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True, name="phone-remote").start()
        _SERVER = srv
        return srv


def stop():
    global _SERVER
    with _LOCK:
        srv, _SERVER = _SERVER, None
    if srv is not None:
        srv.shutdown()
        srv.server_close()
