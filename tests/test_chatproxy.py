"""The chat listener: llama.cpp's own web UI, proxied on its own origin."""
import conftest_paths  # noqa: F401
import http.client, json, threading, time, unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import chatproxy, server


class PathTest(unittest.TestCase):
    def test_paths_pass_through(self):
        for p in ("/", "/?model=qwen&load=1", "/v1/chat/completions",
                  "/_app/immutable/x.js"):
            self.assertEqual(chatproxy.upstream_path(p), p)

    def test_rejects_smuggling(self):
        for p in ("//evil.com/x", "/..\\x", "http://evil.com/"):
            self.assertIsNone(chatproxy.upstream_path(p), p)

    def test_headers_drop_browser_credentials_and_add_router_key(self):
        h = chatproxy.upstream_headers(
            {"content-type": "application/json", "cookie": "s=1",
             "authorization": "Bearer from-browser", "origin": "http://x",
             "accept": "text/event-stream"}, "sekrit")
        self.assertEqual(h, {"content-type": "application/json",
                             "accept": "text/event-stream",
                             "authorization": "Bearer sekrit"})
        self.assertNotIn("authorization", chatproxy.upstream_headers({}, ""))

    def test_chat_origin_cannot_drive_the_panel(self):
        """The whole point of the separate port: the panel refuses it."""
        self.assertFalse(server._origin_ok("http://127.0.0.1:8091", 8090))
        self.assertFalse(server._origin_ok("http://localhost:8091", 8090))

    def test_chat_listener_refuses_panel_and_foreign_origins(self):
        self.assertTrue(chatproxy.origin_ok("", 8091))
        self.assertTrue(chatproxy.origin_ok("http://127.0.0.1:8091", 8091))
        for o in ("http://127.0.0.1:8090", "https://evil.example", "null"):
            self.assertFalse(chatproxy.origin_ok(o, 8091), o)


class FakeRouter(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    seen = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        FakeRouter.seen.append(("GET", self.path, dict(self.headers)))
        body = b"<html>webui</html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Set-Cookie", "leak=1")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        FakeRouter.seen.append(("POST", self.path, dict(self.headers),
                                self.rfile.read(n)))
        if self.path == "/v1/chat/completions":
            # SSE with a gap between events: must arrive incrementally
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for ev in (b"data: one\n\n", b"data: two\n\n"):
                self.wfile.write(b"%x\r\n%s\r\n" % (len(ev), ev))
                self.wfile.flush()
                time.sleep(0.3)
            self.wfile.write(b"0\r\n\r\n")
            return
        body = b'{"error":"nope"}'
        self.send_response(400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class LiveProxyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.router = ThreadingHTTPServer(("127.0.0.1", 0), FakeRouter)
        cls.cfg = {}
        cls.panel = ThreadingHTTPServer(("127.0.0.1", 0),
                                        chatproxy.make_handler(lambda: cls.cfg))
        for s in (cls.router, cls.panel):
            threading.Thread(target=s.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        for s in (cls.router, cls.panel):
            s.shutdown(); s.server_close()

    def setUp(self):
        FakeRouter.seen.clear()
        self.pport = self.panel.server_address[1]
        self.cfg.clear()
        self.cfg.update(chat_port=self.pport, panel_port=8090, router_api_key="sekrit",
                        router_port=self.router.server_address[1])

    def _conn(self):
        c = http.client.HTTPConnection("127.0.0.1", self.pport, timeout=10)
        self.addCleanup(c.close)
        return c

    def test_page_is_proxied_with_key_and_without_cookies(self):
        c = self._conn()
        c.request("GET", "/?model=a", headers={
            "Host": f"127.0.0.1:{self.pport}", "Cookie": "panel=1"})
        r = c.getresponse()
        self.assertEqual(r.status, 200)
        self.assertEqual(r.read(), b"<html>webui</html>")
        self.assertIsNone(r.getheader("Set-Cookie"))
        self.assertIn("frame-ancestors http://127.0.0.1:8090",
                      r.getheader("Content-Security-Policy"))
        method, path, headers = FakeRouter.seen[0]
        self.assertEqual(path, "/?model=a")
        self.assertEqual(headers.get("Authorization"), "Bearer sekrit")
        self.assertNotIn("Cookie", headers)

    def test_stream_arrives_incrementally(self):
        c = self._conn()
        body = json.dumps({"stream": True}).encode()
        c.request("POST", "/v1/chat/completions", body=body, headers={
            "Host": f"127.0.0.1:{self.pport}", "Origin": f"http://127.0.0.1:{self.pport}",
            "Content-Type": "application/json"})
        r = c.getresponse()
        self.assertEqual(r.status, 200)
        t0 = time.time()
        first = r.read1(100)
        self.assertIn(b"data: one", first)
        self.assertLess(time.time() - t0, 0.25, "first event was buffered")
        rest = r.read()
        self.assertIn(b"data: two", first + rest)
        self.assertEqual(FakeRouter.seen[0][3], body)

    def test_upstream_errors_are_relayed(self):
        c = self._conn()
        c.request("POST", "/models/load", body=b"{}", headers={
            "Host": f"127.0.0.1:{self.pport}", "Content-Type": "application/json"})
        r = c.getresponse()
        self.assertEqual(r.status, 400)
        self.assertEqual(r.read(), b'{"error":"nope"}')

    def test_cross_site_post_is_refused_before_reaching_router(self):
        c = self._conn()
        c.request("POST", "/v1/chat/completions", body=b"{}", headers={
            "Host": f"127.0.0.1:{self.pport}", "Origin": "https://evil.example",
            "Content-Type": "application/json"})
        r = c.getresponse(); r.read()
        self.assertEqual(r.status, 403)
        self.assertEqual(FakeRouter.seen, [])

    def test_router_down_gives_friendly_page(self):
        self.cfg["router_port"] = 1
        c = self._conn()
        c.request("GET", "/", headers={"Host": f"127.0.0.1:{self.pport}"})
        r = c.getresponse()
        self.assertEqual(r.status, 502)
        self.assertIn(b"router isn't running", r.read())

    def test_panel_origin_post_is_refused(self):
        c = self._conn()
        c.request("POST", "/v1/chat/completions", body=b"{}", headers={
            "Host": f"127.0.0.1:{self.pport}", "Origin": "http://127.0.0.1:8090",
            "Content-Type": "application/json"})
        r = c.getresponse(); r.read()
        self.assertEqual(r.status, 403)
        self.assertEqual(FakeRouter.seen, [])


if __name__ == "__main__":
    unittest.main()
