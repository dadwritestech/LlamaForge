"""Local Batch API, pure parts: ids, multipart, input checks, output lines, objects."""
import json
import unittest

import conftest_paths  # noqa: F401
import batchapi


def line(cid, model="qwen", url="/v1/chat/completions", **extra):
    d = {"custom_id": cid, "method": "POST", "url": url,
         "body": {"model": model, "messages": [{"role": "user", "content": "hi"}]}}
    d.update(extra)
    return json.dumps(d)


class Ids(unittest.TestCase):
    def test_ids_have_openai_prefixes_and_are_unique(self):
        a, b = batchapi.new_id("file"), batchapi.new_id("file")
        self.assertTrue(a.startswith("file-"))
        self.assertNotEqual(a, b)
        self.assertTrue(batchapi.new_id("batch").startswith("batch_"))
        self.assertTrue(batchapi.new_id("batch_req").startswith("batch_req_"))

    def test_id_check_refuses_paths(self):
        self.assertTrue(batchapi.valid_id("file-0123abcd", "file"))
        self.assertFalse(batchapi.valid_id("file-../x", "file"))
        self.assertFalse(batchapi.valid_id("batch_ab", "file"))


class Multipart(unittest.TestCase):
    def body(self, parts, boundary="XyZ"):
        out = b""
        for name, value, filename in parts:
            disp = f'form-data; name="{name}"' + (f'; filename="{filename}"' if filename else "")
            out += f"--{boundary}\r\nContent-Disposition: {disp}\r\n".encode()
            if filename:
                out += b"Content-Type: application/jsonl\r\n"
            out += b"\r\n" + value + b"\r\n"
        return out + f"--{boundary}--\r\n".encode()

    def test_sdk_upload_parses(self):
        data = (line("a") + "\n" + line("b") + "\n").encode()
        raw = self.body([("purpose", b"batch", None), ("file", data, "in.jsonl")])
        got = batchapi.parse_multipart('multipart/form-data; boundary="XyZ"', raw)
        self.assertEqual(got, {"purpose": "batch", "filename": "in.jsonl", "content": data})

    def test_missing_file_or_boundary(self):
        with self.assertRaisesRegex(ValueError, "boundary"):
            batchapi.parse_multipart("multipart/form-data", b"")
        raw = self.body([("purpose", b"batch", None)])
        with self.assertRaisesRegex(ValueError, "file"):
            batchapi.parse_multipart("multipart/form-data; boundary=XyZ", raw)

    def test_filename_is_reduced_to_a_basename(self):
        raw = self.body([("purpose", b"batch", None), ("file", b"{}", "..\\..\\evil.jsonl")])
        got = batchapi.parse_multipart("multipart/form-data; boundary=XyZ", raw)
        self.assertEqual(got["filename"], "evil.jsonl")


class Input(unittest.TestCase):
    def test_good_file_gives_requests_and_the_one_model(self):
        data = "\n".join([line("a"), "", line("b")]) + "\n"
        reqs, model = batchapi.parse_input(data.encode(), "/v1/chat/completions")
        self.assertEqual([r["custom_id"] for r in reqs], ["a", "b"])
        self.assertEqual(model, "qwen")
        self.assertFalse(reqs[0]["body"].get("stream"))

    def test_stream_is_forced_off(self):
        d = json.loads(line("a"))
        d["body"]["stream"] = True
        reqs, _ = batchapi.parse_input(json.dumps(d).encode(), "/v1/chat/completions")
        self.assertIs(reqs[0]["body"]["stream"], False)

    def test_errors_name_the_line(self):
        cases = [
            ("not json", "line 1: not JSON"),
            (line("a") + "\n" + line("a"), "line 2: duplicate custom_id 'a'"),
            (line("a", url="/v1/embeddings"), "line 1: url must be /v1/chat/completions"),
            (json.dumps({"custom_id": "a", "method": "POST", "url": "/v1/chat/completions"}),
             "line 1: body must be an object"),
            (json.dumps({"method": "POST", "url": "/v1/chat/completions", "body": {}}),
             "line 1: custom_id must be a non-empty string"),
            (line("a", method="GET"), "line 1: method must be POST"),
            (line("a") + "\n" + line("b", model="gemma"),
             "line 2: model 'gemma' differs from 'qwen'"),
            ("", "the file has no requests"),
        ]
        for text, msg in cases:
            with self.subTest(msg=msg):
                with self.assertRaises(ValueError) as cm:
                    batchapi.parse_input(text.encode(), "/v1/chat/completions")
                self.assertIn(msg, str(cm.exception))

    def test_only_chat_completions_for_now(self):
        with self.assertRaisesRegex(ValueError, "endpoint"):
            batchapi.parse_input(line("a").encode(), "/v1/embeddings")

    def test_size_limits(self):
        with self.assertRaisesRegex(ValueError, "lines"):
            batchapi.parse_input(("\n".join(line(str(i)) for i in range(4))).encode(),
                                 "/v1/chat/completions", max_lines=3)
        with self.assertRaisesRegex(ValueError, "UTF-8"):
            batchapi.parse_input(b"\xff\xfe", "/v1/chat/completions")


class Output(unittest.TestCase):
    def test_result_line(self):
        out = batchapi.result_line("a", 200, {"id": "chatcmpl-1"})
        self.assertTrue(out["id"].startswith("batch_req_"))
        self.assertEqual(out["custom_id"], "a")
        self.assertEqual(out["response"]["status_code"], 200)
        self.assertEqual(out["response"]["body"], {"id": "chatcmpl-1"})
        self.assertIsNone(out["error"])

    def test_error_line(self):
        out = batchapi.error_line("a", "router_unreachable", "connection refused")
        self.assertIsNone(out["response"])
        self.assertEqual(out["error"], {"code": "router_unreachable", "message": "connection refused"})

    def test_http_error_counts_as_failed(self):
        self.assertTrue(batchapi.ok_status(200))
        self.assertFalse(batchapi.ok_status(400))
        self.assertFalse(batchapi.ok_status(503))


class Objects(unittest.TestCase):
    def test_file_object(self):
        f = batchapi.file_object({"id": "file-1", "bytes": 10, "created_at": 5,
                                  "filename": "in.jsonl", "purpose": "batch"})
        self.assertEqual(f, {"id": "file-1", "object": "file", "bytes": 10, "created_at": 5,
                             "filename": "in.jsonl", "purpose": "batch", "status": "processed"})

    def test_new_batch_then_object(self):
        b = batchapi.new_batch("file-1", "/v1/chat/completions", "24h", 3, "qwen",
                               metadata={"job": "x"}, when="idle", now=100)
        o = batchapi.batch_object(b)
        self.assertEqual(o["object"], "batch")
        self.assertEqual(o["status"], "validating")
        self.assertEqual(o["input_file_id"], "file-1")
        self.assertEqual(o["request_counts"], {"total": 3, "completed": 0, "failed": 0})
        self.assertEqual(o["created_at"], 100)
        self.assertEqual(o["expires_at"], 100 + 24 * 3600)
        self.assertEqual(o["metadata"], {"job": "x"})
        self.assertIsNone(o["output_file_id"])
        self.assertEqual(o["model"], "qwen")
        self.assertNotIn("when", o)            # panel-only fields stay internal

    def test_completion_window_is_24h_only(self):
        with self.assertRaisesRegex(ValueError, "24h"):
            batchapi.new_batch("file-1", "/v1/chat/completions", "1h", 1, "q")

    def test_metadata_limits(self):
        with self.assertRaisesRegex(ValueError, "metadata"):
            batchapi.new_batch("file-1", "/v1/chat/completions", "24h", 1, "q",
                               metadata={str(i): "v" for i in range(17)})

    def test_transitions(self):
        b = batchapi.new_batch("file-1", "/v1/chat/completions", "24h", 1, "q", now=0)
        batchapi.move(b, "in_progress", now=1)
        self.assertEqual(b["in_progress_at"], 1)
        batchapi.move(b, "cancelling", now=2)
        self.assertEqual(b["cancelling_at"], 2)
        batchapi.move(b, "cancelled", now=3)
        self.assertEqual(b["cancelled_at"], 3)
        with self.assertRaisesRegex(ValueError, "cancelled -> in_progress"):
            batchapi.move(b, "in_progress")

    def test_terminal_states(self):
        for s in ("completed", "failed", "cancelled", "expired"):
            self.assertTrue(batchapi.is_done({"status": s}))
        self.assertFalse(batchapi.is_done({"status": "in_progress"}))


if __name__ == "__main__":
    unittest.main()
