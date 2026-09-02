"""The trust boundary around the dashboard.

The panel binds 127.0.0.1, which keeps it off the network but leaves it
reachable by every page the user browses - and its routes rebuild llama.cpp,
install packages and rewrite configuration. These tests drive a real
ThreadingHTTPServer the way a hostile page would.
"""
import conftest_paths  # noqa: F401
import json, socket, threading, unittest, urllib.error, urllib.request
from http.server import ThreadingHTTPServer
from unittest import mock

import routes, server


class GuardUnitTest(unittest.TestCase):
    def test_host_accepts_loopback_names(self):
        for h in ("127.0.0.1:8090", "localhost:8090", "127.0.0.1", "[::1]:8090"):
            self.assertTrue(server._host_ok(h, 8090), h)

    def test_host_rejects_foreign_names_and_ports(self):
        for h in ("evil.com:8090", "192.168.1.5:8090", "127.0.0.1:9999", ""):
            self.assertFalse(server._host_ok(h, 8090), h)

    def test_host_rejects_rebinding_hostname(self):
        """DNS rebinding: attacker's name resolved to 127.0.0.1."""
        self.assertFalse(server._host_ok("attacker.test:8090", 8090))

    def test_origin_absent_is_allowed(self):
        # curl, and agent clients hitting /v1/messages, send no Origin
        self.assertTrue(server._origin_ok("", 8090))

    def test_origin_same_service_allowed(self):
        for o in ("http://127.0.0.1:8090", "http://localhost:8090"):
            self.assertTrue(server._origin_ok(o, 8090), o)

    def test_origin_foreign_rejected(self):
        for o in ("http://evil.com", "https://evil.com", "http://127.0.0.1:9999",
                  "null", "file://"):
            self.assertFalse(server._origin_ok(o, 8090), o)


class LiveServerTest(unittest.TestCase):
    """Exercises dispatch end to end over a real socket."""

    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.H)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        # the guard reads panel_port from config; point it at the test socket
        self.cfg_patch = mock.patch.object(
            routes, "cfg", return_value={"panel_port": self.port,
                                         "anthropic_shim_enabled": False})
        self.cfg_patch.start()
        self.addCleanup(self.cfg_patch.stop)
        self.seen = []

        def probe(req):
            self.seen.append(req)
            return 200, {"ok": True, "body": req.body, "qs": req.qs}

        self.routes_patch = mock.patch.dict(
            routes.GET_ROUTES, {"/api/_probe": probe}, clear=False)
        self.post_patch = mock.patch.dict(
            routes.POST_ROUTES, {"/api/_probe": probe}, clear=False)
        self.routes_patch.start(); self.post_patch.start()
        self.addCleanup(self.routes_patch.stop)
        self.addCleanup(self.post_patch.stop)

    def _req(self, path, method="GET", headers=None, data=None):
        url = f"http://127.0.0.1:{self.port}{path}"
        h = {"Host": f"127.0.0.1:{self.port}"}
        h.update(headers or {})
        body = json.dumps(data).encode() if data is not None else None
        if body is not None:
            h.setdefault("Content-Type", "application/json")
        r = urllib.request.Request(url, data=body, method=method, headers=h)
        try:
            with urllib.request.urlopen(r, timeout=10) as resp:
                return resp.status, json.loads(resp.read().decode() or "{}")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode() or "{}")

    def _raw(self, head, body=b"", shutdown_write=True):
        import socket
        with socket.create_connection(("127.0.0.1", self.port), timeout=10) as sock:
            sock.sendall(head + body)
            if shutdown_write:
                sock.shutdown(socket.SHUT_WR)
            chunks = []
            while True:
                part = sock.recv(65536)
                if not part:
                    break
                chunks.append(part)
        raw = b"".join(chunks)
        header, _, payload = raw.partition(b"\r\n\r\n")
        lines = header.decode("iso-8859-1").split("\r\n")
        status = int(lines[0].split()[1])
        headers = {}
        for line in lines[1:]:
            if ":" in line:
                key, value = line.split(":", 1)
                headers[key.lower()] = value.strip()
        return status, headers, payload

    def _raw_post(self, path, content_length=None, extra_headers=(), body=b""):
        lines = [
            f"POST {path} HTTP/1.1",
            f"Host: 127.0.0.1:{self.port}",
            "Content-Type: application/json",
        ]
        if content_length is not None:
            lines.append(f"Content-Length: {content_length}")
        lines.extend(extra_headers)
        head = ("\r\n".join(lines) + "\r\n\r\n").encode("ascii")
        return self._raw(head, body)

    # ---------------------------------------------------------------- happy
    def test_same_origin_get_is_dispatched(self):
        status, body = self._req(
            "/api/_probe?x=1", headers={"Origin": f"http://127.0.0.1:{self.port}"})
        self.assertEqual(status, 200)
        self.assertEqual(body["qs"], {"x": "1"})

    def test_no_origin_get_is_dispatched(self):
        status, body = self._req("/api/_probe")
        self.assertEqual(status, 200)

    def test_json_post_is_dispatched(self):
        status, body = self._req("/api/_probe", "POST", data={"hello": "world"})
        self.assertEqual(status, 200)
        self.assertEqual(body["body"], {"hello": "world"})

    def test_json_responses_are_no_store(self):
        status, body = self._req("/api/_probe")
        self.assertEqual(status, 200)
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/_probe",
            headers={"Host": f"127.0.0.1:{self.port}"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            self.assertEqual(resp.headers["Cache-Control"], "no-store")

    def test_missing_content_length_is_411_and_closes(self):
        status, headers, _ = self._raw_post("/api/_probe")
        self.assertEqual(status, 411)
        self.assertEqual(headers.get("connection"), "close")
        self.assertEqual(self.seen, [])

    def test_management_limit_is_selected_before_read(self):
        limit = 128
        body = b'{"x":"' + b"a" * (limit - 8) + b'"}'
        self.assertEqual(len(body), limit)
        with mock.patch.object(server, "MAX_MANAGEMENT_JSON_BODY_BYTES", limit):
            status, _, _ = self._raw_post("/api/_probe", limit, body=body)
            self.assertEqual(status, 200)
            self.seen.clear()
            status, headers, _ = self._raw_post("/api/_probe", limit + 1)
        self.assertEqual(status, 413)
        self.assertEqual(headers.get("connection"), "close")
        self.assertEqual(self.seen, [])

    def test_proxy_payload_above_management_limit_uses_proxy_limit(self):
        prefix = b'{"model":"m","messages":[],"pad":"'
        suffix = b'"}'
        body = prefix + b"a" * (256 - len(prefix) - len(suffix)) + suffix
        self.assertEqual(len(body), 256)
        with mock.patch.object(server, "MAX_MANAGEMENT_JSON_BODY_BYTES", 96), \
             mock.patch.object(server, "MAX_PROXY_JSON_BODY_BYTES", 256), \
             mock.patch.object(routes, "_router_openai", return_value=(200, {"ok": True})):
            status, _, payload = self._raw_post(
                "/v1/chat/completions", len(body), body=body)
        self.assertEqual(status, 200)
        self.assertIn(b'"ok": true', payload)

    def test_proxy_over_its_limit_is_413_without_read(self):
        with mock.patch.object(server, "MAX_PROXY_JSON_BODY_BYTES", 256):
            status, headers, _ = self._raw_post("/v1/messages", 257)
        self.assertEqual(status, 413)
        self.assertEqual(headers.get("connection"), "close")

    def test_invalid_and_duplicate_lengths_never_dispatch(self):
        for value in ("-1", "+1", "nope", "1x", "2 ", "2\t"):
            with self.subTest(value=value):
                status, headers, _ = self._raw_post("/api/_probe", value)
                self.assertEqual(status, 400)
                self.assertEqual(headers.get("connection"), "close")
        duplicate = (
            f"POST /api/_probe HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{self.port}\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: 2\r\nContent-Length: 2\r\n\r\n{}"
        ).encode("ascii")
        status, headers, _ = self._raw(duplicate)
        self.assertEqual(status, 400)
        self.assertEqual(headers.get("connection"), "close")
        self.assertEqual(self.seen, [])

    def test_transfer_encoding_and_short_body_close_without_dispatch(self):
        status, headers, _ = self._raw_post(
            "/api/_probe", None, extra_headers=("Transfer-Encoding: chunked",),
            body=b"2\r\n{}\r\n0\r\n\r\n")
        self.assertEqual(status, 400)
        self.assertEqual(headers.get("connection"), "close")
        status, headers, _ = self._raw_post("/api/_probe", 20, body=b"{}")
        self.assertEqual(status, 400)
        self.assertEqual(headers.get("connection"), "close")
        self.assertEqual(self.seen, [])
        self.assertEqual(self._req("/api/_probe")[0], 200)

    def test_unknown_path_404s(self):
        status, _ = self._req("/api/nope")
        self.assertEqual(status, 404)

    # ------------------------------------------------------------- rejected
    def test_cross_origin_get_is_refused(self):
        status, body = self._req("/api/_probe",
                                 headers={"Origin": "http://evil.example"})
        self.assertEqual(status, 403)
        self.assertEqual(self.seen, [], "handler ran despite a foreign Origin")

    def test_cross_origin_post_is_refused(self):
        status, _ = self._req("/api/_probe", "POST", data={"a": 1},
                              headers={"Origin": "http://evil.example"})
        self.assertEqual(status, 403)
        self.assertEqual(self.seen, [])

    def test_rebound_host_is_refused(self):
        status, _ = self._req("/api/_probe", headers={"Host": "attacker.test"})
        self.assertEqual(status, 403)
        self.assertEqual(self.seen, [])

    def test_form_content_type_post_is_refused(self):
        """The CSRF vector: a cross-site <form> can only send these types, and
        the body was previously json.loads()ed regardless of Content-Type."""
        for ctype in ("text/plain", "application/x-www-form-urlencoded",
                      "multipart/form-data"):
            status, _ = self._req("/api/_probe", "POST", data={"a": 1},
                                  headers={"Content-Type": ctype})
            self.assertEqual(status, 415, ctype)
        self.assertEqual(self.seen, [])

    def test_malformed_json_is_a_400_not_a_crash(self):
        url = f"http://127.0.0.1:{self.port}/api/_probe"
        r = urllib.request.Request(
            url, data=b"{not json", method="POST",
            headers={"Host": f"127.0.0.1:{self.port}",
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(r, timeout=10) as resp:
                status = resp.status
        except urllib.error.HTTPError as e:
            status = e.code
        self.assertEqual(status, 400)

    def test_non_object_json_body_is_refused(self):
        status, _ = self._req("/api/_probe", "POST", data=[1, 2, 3])
        self.assertEqual(status, 400)

    def test_handler_exception_becomes_500_not_a_dead_server(self):
        def boom(req):
            raise RuntimeError("kaboom")

        with mock.patch.dict(routes.GET_ROUTES, {"/api/_boom": boom}):
            status, body = self._req("/api/_boom")
        self.assertEqual(status, 500)
        self.assertIn("kaboom", body["error"])
        self.assertEqual(self._req("/api/_probe")[0], 200)   # still serving

    def test_api_error_carries_its_status(self):
        def refuse(req):
            raise routes.ApiError(418, "nope")

        with mock.patch.dict(routes.GET_ROUTES, {"/api/_refuse": refuse}):
            status, body = self._req("/api/_refuse")
        self.assertEqual((status, body["error"]), (418, "nope"))


if __name__ == "__main__":
    unittest.main()
