"""backend/phone.py: the opt-in phone remote. A LAN listener that serves one
chat page and two token-guarded endpoints, and nothing else."""
import conftest_paths  # noqa: F401
import http.client, io, json, unittest

import phone

TOKEN = "t0ken-abcdefghijklmnop"


class FakeResp(io.BytesIO):
    status = 200


class Fakes:
    def __init__(self):
        self.chats = []

    def models(self):
        return 200, {"data": [{"id": "qwen", "status": {"value": "loaded"}},
                              {"id": "gemma", "status": {"value": "unloaded"}}]}

    def gpus(self):
        return [{"index": 0, "name": "RTX 5080", "used": 9000, "total": 16303,
                 "util": 71, "temp": 60}]

    def chat(self, body):
        self.chats.append(body)
        return 200, FakeResp(b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n')


class PureParts(unittest.TestCase):
    def test_token_must_match_exactly(self):
        self.assertTrue(phone.token_ok("Bearer " + TOKEN, TOKEN))
        self.assertFalse(phone.token_ok("Bearer " + TOKEN + "x", TOKEN))
        self.assertFalse(phone.token_ok(TOKEN, TOKEN))
        self.assertFalse(phone.token_ok(None, TOKEN))
        self.assertFalse(phone.token_ok("Bearer ", ""))      # no token = locked

    def test_new_token_is_long_and_url_safe(self):
        t = phone.new_token()
        self.assertGreaterEqual(len(t), 22)
        self.assertRegex(t, r"^[A-Za-z0-9_-]+$")
        self.assertNotEqual(t, phone.new_token())

    def test_chat_body_keeps_only_plain_messages(self):
        out = phone.sanitize_chat({
            "model": "qwen", "temperature": 9, "tools": [{"x": 1}],
            "messages": [{"role": "system", "content": "be brief"},
                         {"role": "user", "content": "hi", "name": "x"},
                         {"role": "tool", "content": "no"},
                         {"role": "assistant", "content": ["not", "a", "str"]}]})
        self.assertEqual(out["model"], "qwen")
        self.assertEqual(out["messages"], [{"role": "system", "content": "be brief"},
                                           {"role": "user", "content": "hi"}])
        self.assertTrue(out["stream"])
        self.assertNotIn("tools", out)
        self.assertNotIn("temperature", out)

    def test_chat_body_refuses_junk(self):
        self.assertIsNone(phone.sanitize_chat({"model": "", "messages": [{"role": "user", "content": "x"}]}))
        self.assertIsNone(phone.sanitize_chat({"model": "q", "messages": []}))
        self.assertIsNone(phone.sanitize_chat({"model": "q", "messages": "hi"}))
        self.assertIsNone(phone.sanitize_chat([]))

    def test_status_lists_models_and_gpus_only(self):
        s = phone.status(*Fakes().models(), Fakes().gpus())
        self.assertEqual(s["models"], [{"id": "qwen", "loaded": True}, {"id": "gemma", "loaded": False}])
        self.assertEqual(s["gpus"], [{"name": "RTX 5080", "used": 9000, "total": 16303, "util": 71}])
        self.assertEqual(phone.status(599, {"error": "down"}, [])["models"], [])

    def test_tailnet_addresses_are_100_64_slash_10(self):
        self.assertTrue(phone.is_tailnet("100.101.102.103"))
        self.assertFalse(phone.is_tailnet("100.128.0.1"))
        self.assertFalse(phone.is_tailnet("192.168.1.20"))
        self.assertFalse(phone.is_tailnet("nonsense"))

    def test_urls_carry_the_token_in_the_fragment(self):
        u = phone.urls(8092, TOKEN, lan="192.168.1.20", tailnet=["100.70.1.2"])
        self.assertEqual(u, [{"kind": "lan", "url": "http://192.168.1.20:8092/#k=" + TOKEN},
                             {"kind": "tailnet", "url": "http://100.70.1.2:8092/#k=" + TOKEN}])
        self.assertEqual(phone.urls(8092, TOKEN, lan=None, tailnet=[]), [])


class Listener(unittest.TestCase):
    def setUp(self):
        self.fakes = Fakes()
        self.cfg = {"phone_enabled": True, "phone_port": 0, "phone_token": TOKEN}
        self.srv = phone.start(lambda: self.cfg, self.fakes, host="127.0.0.1")
        self.port = self.srv.server_address[1]

    def tearDown(self):
        phone.stop()

    def req(self, method, path, body=None, token=TOKEN, ctype="application/json"):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        headers = {}
        if token is not None:
            headers["Authorization"] = "Bearer " + token
        if body is not None:
            body = json.dumps(body).encode()
            headers["Content-Type"] = ctype
        conn.request(method, path, body=body, headers=headers)
        r = conn.getresponse()
        data = r.read()
        conn.close()
        return r, data

    def test_page_is_served_without_a_token_and_holds_no_data(self):
        r, data = self.req("GET", "/", token=None)
        self.assertEqual(r.status, 200)
        self.assertIn("text/html", r.getheader("Content-Type"))
        self.assertIn("default-src 'none'", r.getheader("Content-Security-Policy"))
        self.assertNotIn(TOKEN.encode(), data)
        self.assertNotIn(b"qwen", data)

    def test_status_needs_the_token(self):
        r, _ = self.req("GET", "/api/status", token=None)
        self.assertEqual(r.status, 401)
        r, _ = self.req("GET", "/api/status", token="wrong")
        self.assertEqual(r.status, 401)
        r, data = self.req("GET", "/api/status")
        self.assertEqual(r.status, 200)
        self.assertEqual(json.loads(data)["models"][0]["id"], "qwen")
        self.assertIsNone(r.getheader("Access-Control-Allow-Origin"))

    def test_chat_streams_the_sanitized_body(self):
        r, _ = self.req("POST", "/api/chat", {"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
                        token=None)
        self.assertEqual(r.status, 401)
        self.assertEqual(self.fakes.chats, [])
        r, data = self.req("POST", "/api/chat", {"model": "qwen", "tools": [1],
                                                 "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(r.status, 200)
        self.assertIn(b"[DONE]", data)
        self.assertEqual(len(self.fakes.chats), 1)
        self.assertNotIn("tools", self.fakes.chats[0])

    def test_chat_refuses_other_content_types_and_bad_bodies(self):
        r, _ = self.req("POST", "/api/chat", {"model": "q", "messages": [{"role": "user", "content": "x"}]},
                        ctype="text/plain")
        self.assertEqual(r.status, 415)
        r, _ = self.req("POST", "/api/chat", {"model": "q"})
        self.assertEqual(r.status, 400)

    def test_nothing_else_is_reachable(self):
        for path in ("/api/state", "/api/config", "/v1/chat/completions", "/web/js/app.js"):
            r, _ = self.req("GET", path)
            self.assertEqual(r.status, 404, path)

    def test_disabling_or_rotating_locks_the_old_link(self):
        self.cfg["phone_token"] = "rotated-token-0123456789"
        r, _ = self.req("GET", "/api/status")
        self.assertEqual(r.status, 401)
        self.cfg["phone_enabled"] = False
        r, _ = self.req("GET", "/api/status", token="rotated-token-0123456789")
        self.assertEqual(r.status, 404)

    def test_start_twice_is_one_listener_and_stop_closes_it(self):
        self.assertIs(phone.start(lambda: self.cfg, self.fakes, host="127.0.0.1"), self.srv)
        self.assertTrue(phone.running())
        phone.stop()
        self.assertFalse(phone.running())


class PanelRoutes(unittest.TestCase):
    """GET/POST /api/phone on the loopback panel: turn it on, off, rotate."""
    def setUp(self):
        import os, tempfile
        from unittest import mock
        import config, routes
        self.config, self.routes = config, routes
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.remove(path)
        self.old, config.CONFIG = config.CONFIG, path
        self.addCleanup(lambda: (setattr(config, "CONFIG", self.old),
                                 os.path.exists(path) and os.remove(path)))
        self.started = []
        patches = [mock.patch.object(phone, "start", side_effect=lambda get_cfg: self.started.append(get_cfg) or object()),
                   mock.patch.object(phone, "stop"),
                   mock.patch.object(phone, "running", side_effect=lambda: bool(self.started)),
                   mock.patch("router_ctl.lan_ip", return_value="192.168.1.20"),
                   mock.patch.object(phone, "tailnet_ips", return_value=[])]
        for p in patches:
            self.mocks = getattr(self, "mocks", []) + [p.start()]
            self.addCleanup(p.stop)

    def test_off_by_default_with_no_link(self):
        st, out = self.routes.get_phone(self.routes.Req())
        self.assertEqual(st, 200)
        self.assertFalse(out["enabled"])
        self.assertEqual(out["urls"], [])

    def test_enabling_makes_a_token_starts_the_listener_and_links_it(self):
        st, out = self.routes.post_phone(self.routes.Req(body={"enabled": True}))
        self.assertTrue(out["enabled"])
        token = self.config.load()["phone_token"]
        self.assertGreaterEqual(len(token), 22)
        self.assertEqual(len(self.started), 1)
        self.assertEqual(out["urls"][0]["url"], "http://192.168.1.20:8092/#k=" + token)
        # turning it on again keeps the paired phones paired
        self.routes.post_phone(self.routes.Req(body={"enabled": True}))
        self.assertEqual(self.config.load()["phone_token"], token)

    def test_rotate_replaces_the_token(self):
        self.routes.post_phone(self.routes.Req(body={"enabled": True}))
        old = self.config.load()["phone_token"]
        _, out = self.routes.post_phone(self.routes.Req(body={"rotate": True}))
        self.assertNotEqual(self.config.load()["phone_token"], old)
        self.assertNotIn(old, out["urls"][0]["url"])

    def test_disabling_stops_the_listener(self):
        self.routes.post_phone(self.routes.Req(body={"enabled": True}))
        _, out = self.routes.post_phone(self.routes.Req(body={"enabled": False}))
        self.assertFalse(out["enabled"])
        self.assertEqual(out["urls"], [])
        phone.stop.assert_called()

    def test_a_taken_port_leaves_it_off_and_says_so(self):
        phone.start.side_effect = lambda get_cfg: None
        from routes import ApiError
        with self.assertRaises(ApiError) as e:
            self.routes.post_phone(self.routes.Req(body={"enabled": True}))
        self.assertIn("8092", e.exception.message)
        self.assertFalse(self.config.load()["phone_enabled"])

    def test_bad_bodies_are_refused(self):
        from routes import ApiError
        for body in ({}, {"enabled": "yes"}, {"rotate": 1}):
            with self.assertRaises(ApiError):
                self.routes.post_phone(self.routes.Req(body=body))

    def test_the_token_never_reaches_panel_state(self):
        self.routes.post_phone(self.routes.Req(body={"enabled": True}))
        import network_policy
        self.assertNotIn("phone_token", network_policy.public_config(self.config.load()))


if __name__ == "__main__":
    unittest.main()
