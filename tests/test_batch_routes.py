"""Local Batch API over HTTP: /v1/files and /v1/batches the way an OpenAI SDK calls them."""
import conftest_paths  # noqa: F401
import json
import shutil
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest import mock

import batchrun
import routes
import server

LINE = json.dumps({"custom_id": "a", "method": "POST", "url": "/v1/chat/completions",
                   "body": {"model": "qwen", "messages": [{"role": "user", "content": "hi"}]}}) + "\n"


def multipart(data, purpose="batch", boundary="lfB0undary"):
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\n{purpose}\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"in.jsonl\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


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
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.conf = {"panel_port": self.port, "anthropic_shim_enabled": False}
        p = mock.patch.object(routes, "cfg", side_effect=lambda: self.conf)
        p.start()
        self.addCleanup(p.stop)
        self.runner = batchrun.Runner(batchrun.Store(self.dir), lambda b: (200, {"ok": 1}),
                                      ensure=lambda m: None, busy=lambda: False,
                                      last_activity=lambda: 0)
        p = mock.patch.object(routes, "BATCHES", self.runner)
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
                raw, ct = resp.read(), resp.headers.get("Content-Type", "")
                code = resp.status
        except urllib.error.HTTPError as e:
            raw, ct, code = e.read(), e.headers.get("Content-Type", ""), e.code
        return code, (json.loads(raw or b"{}") if ct.split(";")[0].strip() == "application/json" else raw)

    def upload(self, data=LINE.encode(), **kw):
        body, ctype = multipart(data)
        return self.req("/v1/files", "POST", body, ctype, **kw)


class Files(Live):
    def test_sdk_multipart_upload_then_read_back(self):
        code, f = self.upload()
        self.assertEqual(code, 200, f)
        self.assertEqual((f["object"], f["purpose"], f["filename"]), ("file", "batch", "in.jsonl"))
        self.assertEqual(self.req(f"/v1/files/{f['id']}")[1]["id"], f["id"])
        self.assertEqual(self.req(f"/v1/files/{f['id']}/content")[1], LINE.encode())
        listed = self.req("/v1/files")[1]
        self.assertEqual((listed["object"], [x["id"] for x in listed["data"]]), ("list", [f["id"]]))

    def test_json_upload_for_the_panel_page(self):
        code, f = self.req("/v1/files", "POST", {"filename": "x.jsonl", "purpose": "batch", "content": LINE})
        self.assertEqual(code, 200, f)
        self.assertEqual(self.req(f"/v1/files/{f['id']}/content")[1], LINE.encode())

    def test_multipart_from_a_browser_page_is_refused(self):
        code, _ = self.upload(headers={"Origin": f"http://127.0.0.1:{self.port}"})
        self.assertEqual(code, 415)
        body, ctype = multipart(LINE.encode())
        self.assertEqual(self.req("/api/_nope", "POST", body, ctype)[0], 415)   # only /v1/files

    def test_bad_uploads(self):
        self.assertEqual(self.req("/v1/files", "POST", {"filename": "x", "purpose": "fine-tune", "content": "{}"})[0], 400)
        self.assertEqual(self.req("/v1/files", "POST", {"purpose": "batch"})[0], 400)
        self.assertEqual(self.req("/v1/files/file-0123abcd")[0], 404)
        self.assertEqual(self.req("/v1/files/..%2F..%2Fx")[0], 404)

    def test_exposed_router_needs_the_key(self):
        self.conf.update(router_host="0.0.0.0", router_api_key="k")
        self.assertEqual(self.upload()[0], 401)
        self.assertEqual(self.req("/v1/batches")[0], 401)
        self.assertEqual(self.upload(headers={"Authorization": "Bearer k"})[0], 200)


class Batches(Live):
    def test_create_run_and_fetch_results(self):
        fid = self.upload()[1]["id"]
        code, b = self.req("/v1/batches", "POST", {"input_file_id": fid, "endpoint": "/v1/chat/completions",
                                                     "completion_window": "24h", "metadata": {"job": "x"}})
        self.assertEqual(code, 200, b)
        self.assertEqual((b["object"], b["status"], b["request_counts"]["total"]), ("batch", "validating", 1))
        self.runner.step()
        got = self.req(f"/v1/batches/{b['id']}")[1]
        self.assertEqual(got["status"], "completed")
        out = self.req(f"/v1/files/{got['output_file_id']}/content")[1].decode().splitlines()
        self.assertEqual(json.loads(out[0])["response"]["body"], {"ok": 1})
        self.assertEqual([x["id"] for x in self.req("/v1/batches")[1]["data"]], [b["id"]])

    def test_create_errors(self):
        fid = self.upload()[1]["id"]
        base = {"input_file_id": fid, "endpoint": "/v1/chat/completions", "completion_window": "24h"}
        self.assertEqual(self.req("/v1/batches", "POST", {**base, "input_file_id": "file-0123abcd"})[0], 404)
        self.assertEqual(self.req("/v1/batches", "POST", {**base, "completion_window": "1h"})[0], 400)
        self.assertEqual(self.req("/v1/batches", "POST", {**base, "endpoint": "/v1/embeddings"})[0], 400)
        bad = self.req("/v1/files", "POST", {"filename": "b.jsonl", "purpose": "batch", "content": "nope\n"})[1]["id"]
        code, err = self.req("/v1/batches", "POST", {**base, "input_file_id": bad})
        self.assertEqual(code, 400)
        self.assertIn("line 1", err["error"]["message"])

    def test_cancel(self):
        fid = self.upload()[1]["id"]
        b = self.req("/v1/batches", "POST", {"input_file_id": fid, "endpoint": "/v1/chat/completions",
                                              "completion_window": "24h"})[1]
        code, got = self.req(f"/v1/batches/{b['id']}/cancel", "POST", {})
        self.assertEqual((code, got["status"]), (200, "cancelled"))
        self.assertEqual(self.req(f"/v1/batches/{b['id']}/cancel", "POST", {})[0], 400)
        self.assertEqual(self.req("/v1/batches/batch_0123abcd/cancel", "POST", {})[0], 404)
        self.assertEqual(self.req("/v1/batches/batch_0123abcd")[0], 404)


class Wiring(unittest.TestCase):
    def test_busy_while_other_gpu_runs_hold_the_gpus(self):
        with mock.patch.object(routes.PROVING, "status", return_value={"running": True}):
            self.assertTrue(routes._batch_busy())
        with mock.patch.object(routes.PROVING, "status", return_value={"running": False}), \
             mock.patch.object(routes.PRBENCH, "status", return_value={"running": False}), \
             mock.patch.object(routes.PRTRY, "status", return_value={"running": False}), \
             mock.patch.object(routes.FORGE.runner, "running", return_value=False):
            self.assertFalse(routes._batch_busy())

    def test_send_goes_to_the_models_own_endpoint(self):
        with mock.patch.object(routes, "_model_base", return_value="http://127.0.0.1:9999") as base, \
             mock.patch.object(routes, "_router_openai", return_value=(200, {"x": 1})) as ro:
            self.assertEqual(routes._batch_send({"model": "qwen", "stream": False}), (200, {"x": 1}))
        base.assert_called_once_with("qwen")
        ro.assert_called_once_with({"model": "qwen", "stream": False}, base="http://127.0.0.1:9999")


if __name__ == "__main__":
    unittest.main()
