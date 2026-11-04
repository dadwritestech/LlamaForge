"""Speech to text over HTTP: /v1/audio/transcriptions the way the OpenAI SDK calls it,
and the panel's own /api/stt/* routes."""
import conftest_paths  # noqa: F401
import base64
import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest import mock

import routes
import server

STT_MODELS = routes._stt_models       # the real one; Live patches it out
WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 24


def multipart(fields, boundary="lfSttB0undary"):
    out = b""
    for name, (value, filename) in fields.items():
        disp = f'form-data; name="{name}"' + (f'; filename="{filename}"' if filename else "")
        out += f"--{boundary}\r\nContent-Disposition: {disp}\r\n\r\n".encode() + value + b"\r\n"
    return out + f"--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"


class Live(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.H)
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        self.conf = {"panel_port": self.port, "anthropic_shim_enabled": False}
        self.sent = []
        self.reply = (200, {"choices": [{"message": {"content": "Hello from the GPU."}}]})
        self.models = ["voxtral"]
        for name, fn in (("cfg", lambda: self.conf),
                         ("_stt_models", lambda: list(self.models)),
                         ("_router_openai", lambda body, **kw: (self.sent.append(body), self.reply)[1])):
            p = mock.patch.object(routes, name, side_effect=fn)
            p.start()
            self.addCleanup(p.stop)

    def req(self, path, method="GET", body=None, ctype="application/json", headers=None):
        h = {"Host": f"127.0.0.1:{self.port}"}
        if body is not None:
            h["Content-Type"] = ctype
            if isinstance(body, dict):
                body = json.dumps(body).encode()
        h.update(headers or {})
        r = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=body,
                                   method=method, headers=h)
        try:
            with urllib.request.urlopen(r, timeout=10) as resp:
                raw, ct, code = resp.read(), resp.headers.get("Content-Type", ""), resp.status
        except urllib.error.HTTPError as e:
            raw, ct, code = e.read(), e.headers.get("Content-Type", ""), e.code
        return code, (json.loads(raw or b"{}") if ct.split(";")[0].strip() == "application/json" else raw)

    def transcribe(self, extra=None, **kw):
        fields = {"file": (WAV, "clip.wav"), "model": (b"whisper-1", None)}
        fields.update(extra or {})
        body, ctype = multipart(fields)
        return self.req("/v1/audio/transcriptions", "POST", body, ctype, **kw)


class OpenAIRoute(Live):
    def test_sdk_upload_comes_back_as_text(self):
        code, got = self.transcribe()
        self.assertEqual((code, got), (200, {"text": "Hello from the GPU."}))
        sent = self.sent[0]
        self.assertEqual(sent["model"], "voxtral")
        audio = [p for p in sent["messages"][-1]["content"] if p["type"] == "input_audio"][0]["input_audio"]
        self.assertEqual((audio["format"], base64.b64decode(audio["data"])), ("wav", WAV))

    def test_text_format(self):
        code, got = self.transcribe({"response_format": (b"text", None)})
        self.assertEqual((code, got), (200, b"Hello from the GPU."))

    def test_no_model_can_hear(self):
        self.models = []
        code, got = self.transcribe()
        self.assertEqual(code, 400)
        self.assertIn("audio", got["error"]["message"])
        self.assertEqual(self.sent, [])

    def test_router_refusal_is_passed_on(self):
        self.reply = (500, {"error": {"message": "audio input is not supported by this model"}})
        code, got = self.transcribe()
        self.assertEqual(code, 502)
        self.assertIn("not supported", got["error"]["message"])
        self.assertEqual(got["error"]["type"], "server_error")

    def test_browser_pages_cannot_post_multipart(self):
        code, _ = self.transcribe(headers={"Origin": "http://evil.example"})
        self.assertEqual(code, 403)
        code, _ = self.transcribe(headers={"Origin": f"http://127.0.0.1:{self.port}"})
        self.assertEqual(code, 415)
        self.assertEqual(self.sent, [])

    def test_json_body_is_refused(self):
        code, got = self.req("/v1/audio/transcriptions", "POST", {"file": "x"})
        self.assertEqual(code, 400)
        self.assertIn("multipart", got["error"]["message"])

    def test_key_when_the_router_is_exposed(self):
        self.conf.update(router_host="0.0.0.0", router_api_key="k")
        self.assertEqual(self.transcribe()[0], 401)
        self.assertEqual(self.transcribe(headers={"Authorization": "Bearer k"})[0], 200)


class PanelRoutes(Live):
    def test_status_lists_the_models_that_can_hear(self):
        code, got = self.req("/api/stt/status")
        self.assertEqual((code, got["models"]), (200, ["voxtral"]))
        self.assertEqual(got["offer"]["repo"], "ggml-org/Voxtral-Mini-3B-2507-GGUF")
        self.assertIn("phase", got["download"])

    def test_mic_posts_base64_json(self):
        code, got = self.req("/api/stt/transcribe", "POST",
                             {"audio": base64.b64encode(WAV).decode(), "language": "en"})
        self.assertEqual((code, got), (200, {"text": "Hello from the GPU.", "model": "voxtral"}))

    def test_mic_errors_are_panel_shaped(self):
        code, got = self.req("/api/stt/transcribe", "POST", {"audio": "not base64!"})
        self.assertEqual(code, 400)
        self.assertIn("error", got)
        code, got = self.req("/api/stt/transcribe", "POST", {"audio": base64.b64encode(b"webm..").decode()})
        self.assertEqual(code, 400)
        self.assertIn("WAV", got["error"])

    def test_models_come_from_audio_projectors(self):
        sections = {"vox": {"mmproj": "rel/a.gguf"}, "txt": {"model": "m"}}
        with mock.patch("config.read_sections", return_value=sections), \
             mock.patch("config._abs", side_effect=lambda p: "/abs/" + p), \
             mock.patch("gguf.projector_modalities",
                        side_effect=lambda p: {"audio": p == "/abs/rel/a.gguf", "vision": False}):
            self.assertEqual(STT_MODELS(), ["vox"])


if __name__ == "__main__":
    unittest.main()
