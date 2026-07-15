import conftest_paths  # noqa: F401
import os, tempfile, threading, unittest
import urllib.request
import hub


class FakeResp:
    def __init__(self, status, body, clen=None):
        self.status = status
        self._body = body
        self._sent = False
        self.headers = {"Content-Length": str(clen if clen is not None else len(body))}
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def read(self, n):
        if self._sent:
            return b""
        self._sent = True
        return self._body


class TestResume(unittest.TestCase):
    def _patch(self, resp, capture):
        def fake(req, *a, **k):
            capture["range"] = req.get_header("Range")
            return resp
        self._orig = urllib.request.urlopen
        urllib.request.urlopen = fake

    def tearDown(self):
        if hasattr(self, "_orig"):
            urllib.request.urlopen = self._orig

    def test_resume_sends_range_and_appends(self):
        dm = hub.DownloadManager()
        cap = {}
        self._patch(FakeResp(206, b"BBBBB", clen=5), cap)
        with tempfile.TemporaryDirectory() as d:
            dest = os.path.join(d, "m.gguf")
            with open(dest + ".part", "wb") as f:
                f.write(b"AAAAA")           # 5 bytes already fetched
            dm._fetch("http://x", dest)
            self.assertEqual(cap["range"], "bytes=5-")
            with open(dest, "rb") as f:
                self.assertEqual(f.read(), b"AAAAABBBBB")
            self.assertFalse(os.path.exists(dest + ".part"))
            self.assertEqual(dm.state["total"], 10)

    def test_range_ignored_restarts_from_scratch(self):
        dm = hub.DownloadManager()
        cap = {}
        self._patch(FakeResp(200, b"CCCCCCC", clen=7), cap)
        with tempfile.TemporaryDirectory() as d:
            dest = os.path.join(d, "m.gguf")
            with open(dest + ".part", "wb") as f:
                f.write(b"AAAAA")           # stale partial; server ignored Range
            dm._fetch("http://x", dest)
            with open(dest, "rb") as f:
                self.assertEqual(f.read(), b"CCCCCCC")
            self.assertEqual(dm.state["total"], 7)
            self.assertEqual(dm.state["downloaded"], 7)

    def test_pause_keeps_part_and_can_resume(self):
        dm = hub.DownloadManager()
        started = threading.Event()

        def fake_fetch(url, dest):
            open(dest + ".part", "wb").close()   # a partial exists
            started.set()
            while True:
                dm._check_signals()
        dm._fetch = fake_fetch

        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(dm.start("r/x", ["a.gguf"], d))
            started.wait(5)
            self.assertTrue(dm.pause())
            for _ in range(50):
                if not dm.state["running"]:
                    break
                threading.Event().wait(0.05)
            self.assertFalse(dm.state["running"])
            self.assertEqual(dm.state["phase"], "paused")
            self.assertTrue(os.path.exists(os.path.join(d, "a.gguf.part")))
            # resume should relaunch the same job
            started.clear()
            self.assertTrue(dm.resume())
            started.wait(5)
            self.assertTrue(dm.state["running"])
            dm.cancel()

    def test_resume_refused_when_no_paused_job(self):
        self.assertFalse(hub.DownloadManager().resume())


if __name__ == "__main__":
    unittest.main()
