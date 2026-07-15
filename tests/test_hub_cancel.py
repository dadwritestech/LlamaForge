import conftest_paths  # noqa: F401
import os, tempfile, threading, unittest
import hub


class TestCancel(unittest.TestCase):
    def test_cancel_refused_when_idle(self):
        self.assertFalse(hub.DownloadManager().cancel())

    def test_cancelled_run_reports_phase_and_stops(self):
        dm = hub.DownloadManager()
        started = threading.Event()

        def fake_fetch(url, dest):
            started.set()
            while True:                      # simulate a long transfer
                dm._check_cancel()
        dm._fetch = fake_fetch

        with tempfile.TemporaryDirectory() as d:
            dm.state.update(running=True)    # as start() would
            t = threading.Thread(target=dm._run, args=("r/x", ["a.gguf"], d), daemon=True)
            t.start()
            started.wait(5)
            self.assertTrue(dm.cancel())
            t.join(5)
        self.assertFalse(t.is_alive())
        self.assertEqual(dm.state["phase"], "cancelled")
        self.assertFalse(dm.state["running"])
        self.assertEqual(dm.state["error"], "")

    def test_cancel_removes_part_file(self):
        dm = hub.DownloadManager()
        with tempfile.TemporaryDirectory() as d:
            dest = os.path.join(d, "m.gguf")

            class FakeResp:
                headers = {"Content-Length": "10"}
                def __enter__(self): return self
                def __exit__(self, *a): return False
                def read(self, n):
                    dm.state["cancel"] = True   # cancel arrives mid-transfer
                    return b"x"
            import urllib.request
            orig = urllib.request.urlopen
            urllib.request.urlopen = lambda *a, **k: FakeResp()
            try:
                with self.assertRaises(hub.Cancelled):
                    dm._fetch("http://x", dest)
            finally:
                urllib.request.urlopen = orig
            self.assertFalse(os.path.exists(dest + ".part"))
            self.assertFalse(os.path.exists(dest))


if __name__ == "__main__":
    unittest.main()
