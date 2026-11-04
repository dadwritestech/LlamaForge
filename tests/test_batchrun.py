"""Local Batch API: the on-disk store and the runner (driven step by step, no threads)."""
import json
import os
import shutil
import tempfile
import unittest

import conftest_paths  # noqa: F401
import batchapi
import batchrun


def jsonl(*cids, model="qwen"):
    return "".join(json.dumps({"custom_id": c, "method": "POST", "url": "/v1/chat/completions",
                               "body": {"model": model, "messages": [{"role": "user", "content": c}]}})
                   + "\n" for c in cids).encode()


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.store = batchrun.Store(self.dir)
        self.clock = Clock()
        self.sent, self.loaded = [], []
        self.busy = False
        self.last = 0
        self.reply = lambda body: (200, {"id": "c", "choices": [{"message": {"content": "ok " + body["messages"][0]["content"]}}]})
        self.run = self.runner()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def runner(self):
        def send(body):
            self.sent.append(body)
            return self.reply(body)
        return batchrun.Runner(self.store, send, ensure=self.loaded.append,
                               busy=lambda: self.busy, last_activity=lambda: self.last,
                               clock=self.clock)

    def upload(self, data, name="in.jsonl"):
        return self.store.add_file(name, "batch", data)

    def lines(self, file_id):
        return [json.loads(x) for x in self.store.content(file_id).decode().splitlines()]


class StoreTests(Base):
    def test_file_round_trip(self):
        f = self.upload(jsonl("a"))
        self.assertTrue(batchapi.valid_id(f["id"], "file"))
        self.assertEqual(self.store.file(f["id"])["filename"], "in.jsonl")
        self.assertEqual(self.store.content(f["id"]), jsonl("a"))
        self.assertEqual([x["id"] for x in self.store.files()], [f["id"]])
        self.assertIsNone(self.store.file("file-00000000"))
        self.assertIsNone(self.store.file("../../etc"))

    def test_only_batch_purpose_and_size_caps(self):
        with self.assertRaisesRegex(ValueError, "purpose"):
            self.store.add_file("x.jsonl", "fine-tune", b"{}")
        small = batchrun.Store(self.dir, max_bytes=10, max_files=1)
        with self.assertRaisesRegex(ValueError, "too big"):
            small.add_file("x.jsonl", "batch", b"x" * 11)
        small.add_file("x.jsonl", "batch", b"x")
        with self.assertRaisesRegex(ValueError, "files"):
            small.add_file("y.jsonl", "batch", b"y")

    def test_nothing_on_disk_until_the_first_write(self):
        root = os.path.join(self.dir, "later")
        st = batchrun.Store(root)
        self.assertEqual((st.files(), st.batches()), ([], []))
        self.assertFalse(os.path.exists(root))

    def test_remove_file(self):
        f = self.upload(jsonl("a"))
        self.assertTrue(self.store.remove_file(f["id"]))
        self.assertIsNone(self.store.file(f["id"]))
        self.assertFalse(self.store.remove_file(f["id"]))


class RemoveTests(Base):
    def test_remove_a_finished_batch_with_its_files(self):
        f = self.upload(jsonl("a"))
        b = self.run.create(f["id"], "/v1/chat/completions", "24h")
        with self.assertRaisesRegex(ValueError, "still"):
            self.run.remove(b["id"])
        self.run.step()
        out = self.store.batch(b["id"])["output_file_id"]
        self.run.remove(b["id"])
        self.assertIsNone(self.store.batch(b["id"]))
        self.assertEqual((self.store.file(f["id"]), self.store.file(out)), (None, None))
        with self.assertRaisesRegex(LookupError, "no such batch"):
            self.run.remove(b["id"])

    def test_an_input_another_batch_shares_stays(self):
        f = self.upload(jsonl("a"))
        b1 = self.run.create(f["id"], "/v1/chat/completions", "24h")
        self.run.step()
        self.run.create(f["id"], "/v1/chat/completions", "24h")
        self.run.remove(b1["id"])
        self.assertIsNotNone(self.store.file(f["id"]))


class CreateTests(Base):
    def test_create_validates_the_input_file(self):
        f = self.upload(jsonl("a", "b"))
        b = self.run.create(f["id"], "/v1/chat/completions", "24h")
        self.assertEqual(b["status"], "validating")
        self.assertEqual(b["request_counts"]["total"], 2)
        self.assertEqual(b["model"], "qwen")
        self.assertEqual(self.store.batch(b["id"])["id"], b["id"])

    def test_create_errors(self):
        with self.assertRaisesRegex(LookupError, "no such file"):
            self.run.create("file-0123abcd", "/v1/chat/completions", "24h")
        f = self.upload(b"not json\n")
        with self.assertRaisesRegex(ValueError, "line 1"):
            self.run.create(f["id"], "/v1/chat/completions", "24h")
        f = self.upload(jsonl("a"))
        with self.assertRaisesRegex(ValueError, "when"):
            self.run.create(f["id"], "/v1/chat/completions", "24h", when="sometime")


class RunTests(Base):
    def test_runs_to_completion_with_openai_output(self):
        f = self.upload(jsonl("a", "b"))
        b = self.run.create(f["id"], "/v1/chat/completions", "24h")
        self.assertEqual(self.run.step(), "done")
        got = self.store.batch(b["id"])
        self.assertEqual(got["status"], "completed")
        self.assertEqual(got["request_counts"], {"total": 2, "completed": 2, "failed": 0})
        self.assertEqual(self.loaded, ["qwen"])
        self.assertTrue(all(s["stream"] is False for s in self.sent))
        out = self.lines(got["output_file_id"])
        self.assertEqual([o["custom_id"] for o in out], ["a", "b"])
        self.assertEqual(out[1]["response"]["body"]["choices"][0]["message"]["content"], "ok b")
        self.assertIsNone(got["error_file_id"])
        self.assertEqual(self.store.file(got["output_file_id"])["purpose"], "batch_output")
        self.assertEqual(self.run.step(), "idle")

    def test_failures_go_to_the_error_file(self):
        def reply(body):
            c = body["messages"][0]["content"]
            if c == "b":
                return 400, {"error": {"message": "context too long"}}
            if c == "c":
                raise OSError("connection refused")
            return 200, {"ok": 1}
        self.reply = reply
        f = self.upload(jsonl("a", "b", "c"))
        b = self.run.create(f["id"], "/v1/chat/completions", "24h")
        self.run.step()
        got = self.store.batch(b["id"])
        self.assertEqual(got["status"], "completed")
        self.assertEqual(got["request_counts"], {"total": 3, "completed": 1, "failed": 2})
        errs = self.lines(got["error_file_id"])
        self.assertEqual([e["custom_id"] for e in errs], ["b", "c"])
        self.assertEqual(errs[0]["response"]["status_code"], 400)   # an HTTP error keeps its body
        self.assertEqual(errs[1]["error"]["code"], "router_unreachable")

    def test_a_model_that_wont_load_fails_the_batch(self):
        def ensure(mid):
            raise RuntimeError("out of VRAM")
        run = batchrun.Runner(self.store, lambda b: (200, {}), ensure=ensure,
                              busy=lambda: False, last_activity=lambda: 0, clock=self.clock)
        b = run.create(self.upload(jsonl("a"))["id"], "/v1/chat/completions", "24h")
        run.step()
        got = self.store.batch(b["id"])
        self.assertEqual(got["status"], "failed")
        self.assertIn("out of VRAM", got["errors"]["data"][0]["message"])

    def test_input_file_removed_under_it_fails_the_batch(self):
        f = self.upload(jsonl("a"))
        b = self.run.create(f["id"], "/v1/chat/completions", "24h")
        self.store.remove_file(f["id"])
        self.assertEqual(self.run.step(), "done")
        got = self.store.batch(b["id"])
        self.assertEqual(got["status"], "failed")
        self.assertIn("input file", got["errors"]["data"][0]["message"])

    def test_idle_batches_wait_for_a_quiet_router(self):
        f = self.upload(jsonl("a"))
        b = self.run.create(f["id"], "/v1/chat/completions", "24h", when="idle")
        self.last = self.clock.t - 60
        self.assertEqual(self.run.step(), "waiting")
        self.assertEqual(self.store.batch(b["id"])["status"], "validating")
        self.last = self.clock.t - batchrun.QUIET_SECS - 1
        self.assertEqual(self.run.step(), "done")

    def test_waits_while_the_gpus_are_taken(self):
        b = self.run.create(self.upload(jsonl("a"))["id"], "/v1/chat/completions", "24h")
        self.busy = True
        self.assertEqual(self.run.step(), "waiting")
        self.assertEqual(self.sent, [])
        self.busy = False
        self.run.step()
        self.assertEqual(self.store.batch(b["id"])["status"], "completed")

    def test_pauses_mid_batch_when_the_gpus_are_taken(self):
        def reply(body):
            self.busy = True            # proving starts after the first request
            return 200, {}
        self.reply = reply
        b = self.run.create(self.upload(jsonl("a", "b"))["id"], "/v1/chat/completions", "24h")
        self.assertEqual(self.run.step(), "waiting")
        self.assertEqual(self.store.batch(b["id"])["next_line"], 1)
        self.busy = False
        self.reply = lambda body: (200, {})
        self.assertEqual(self.run.step(), "done")
        self.assertEqual(len(self.lines(self.store.batch(b["id"])["output_file_id"])), 2)

    def test_one_batch_at_a_time_oldest_first(self):
        b1 = self.run.create(self.upload(jsonl("a"))["id"], "/v1/chat/completions", "24h")
        self.clock.t += 1
        b2 = self.run.create(self.upload(jsonl("z", model="gemma"))["id"], "/v1/chat/completions", "24h")
        self.run.step()
        self.assertEqual(self.store.batch(b1["id"])["status"], "completed")
        self.assertEqual(self.store.batch(b2["id"])["status"], "validating")
        self.run.step()
        self.assertEqual(self.loaded, ["qwen", "gemma"])

    def test_expiry_keeps_partial_output(self):
        def reply(body):
            self.clock.t += batchapi.WINDOW_SECS + 1
            return 200, {}
        self.reply = reply
        b = self.run.create(self.upload(jsonl("a", "b"))["id"], "/v1/chat/completions", "24h")
        self.run.step()
        got = self.store.batch(b["id"])
        self.assertEqual(got["status"], "expired")
        self.assertEqual(len(self.lines(got["output_file_id"])), 1)


class CancelTests(Base):
    def test_cancel_before_it_starts(self):
        b = self.run.create(self.upload(jsonl("a"))["id"], "/v1/chat/completions", "24h")
        got = self.run.cancel(b["id"])
        self.assertEqual(got["status"], "cancelled")
        self.assertEqual(self.run.step(), "idle")
        self.assertEqual(self.sent, [])

    def test_cancel_mid_run_keeps_what_finished(self):
        def reply(body):
            self.run.cancel(b["id"])
            return 200, {}
        self.reply = reply
        b = self.run.create(self.upload(jsonl("a", "b", "c"))["id"], "/v1/chat/completions", "24h")
        self.run.step()
        got = self.store.batch(b["id"])
        self.assertEqual(got["status"], "cancelled")
        self.assertEqual(got["request_counts"]["completed"], 1)
        self.assertEqual(len(self.lines(got["output_file_id"])), 1)

    def test_cancel_errors(self):
        with self.assertRaises(LookupError):
            self.run.cancel("batch_0123abcd")
        b = self.run.create(self.upload(jsonl("a"))["id"], "/v1/chat/completions", "24h")
        self.run.step()
        with self.assertRaisesRegex(ValueError, "completed"):
            self.run.cancel(b["id"])


class ResumeTests(Base):
    def test_a_restarted_panel_carries_on_from_the_next_line(self):
        calls = []

        def reply(body):
            calls.append(body["messages"][0]["content"])
            if len(calls) == 2:
                raise KeyboardInterrupt      # the panel dies mid-request
            return 200, {}
        self.reply = reply
        b = self.run.create(self.upload(jsonl("a", "b", "c"))["id"], "/v1/chat/completions", "24h")
        with self.assertRaises(KeyboardInterrupt):
            self.run.step()
        self.reply = lambda body: (calls.append(body["messages"][0]["content"]), (200, {}))[1]
        fresh = batchrun.Runner(batchrun.Store(self.dir), lambda body: self.reply(body),
                                ensure=lambda m: None, busy=lambda: False,
                                last_activity=lambda: 0, clock=self.clock)
        self.assertEqual(fresh.step(), "done")
        got = fresh.store.batch(b["id"])
        self.assertEqual(calls, ["a", "b", "b", "c"])
        self.assertEqual([o["custom_id"] for o in self.lines(got["output_file_id"])], ["a", "b", "c"])

    def test_batches_list_newest_first(self):
        b1 = self.run.create(self.upload(jsonl("a"))["id"], "/v1/chat/completions", "24h")
        self.clock.t += 5
        b2 = self.run.create(self.upload(jsonl("b"))["id"], "/v1/chat/completions", "24h")
        self.assertEqual([b["id"] for b in self.store.batches()], [b2["id"], b1["id"]])
        self.assertTrue(os.path.isdir(os.path.join(self.dir, "files")))


if __name__ == "__main__":
    unittest.main()
