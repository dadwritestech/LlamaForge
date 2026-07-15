import conftest_paths  # noqa: F401
import json, os, tempfile, unittest, urllib.error
import stats


class RouterCase(unittest.TestCase):
    def setUp(self):
        self._orig = stats.STATS_FILE
        fd, self.path = tempfile.mkstemp(suffix=".json"); os.close(fd); os.unlink(self.path)
        stats.STATS_FILE = self.path
        self.tr = stats.StatsTracker()
        self.tr._poll_vllm = lambda: None   # keep the test off the network

    def tearDown(self):
        stats.STATS_FILE = self._orig
        if os.path.exists(self.path):
            os.unlink(self.path)

    def _wire(self, prompt, gen, model="nomic", seen=None):
        """Emulate the llama.cpp router: bare /metrics 400s (needs a model
        name), /metrics?model= works, /models reports one loaded model."""
        def fake_get(path, timeout=4):
            if seen is not None:
                seen.append(path)
            if path == "/models":
                return json.dumps({"data": [
                    {"id": "default", "status": {"value": "unloaded"}},
                    {"id": model, "status": {"value": "loaded"}},
                ]})
            if path.startswith("/metrics?model="):
                return (f"llamacpp:prompt_tokens_total {prompt}\n"
                        f"llamacpp:tokens_predicted_total {gen}\n"
                        f"llamacpp:predicted_tokens_seconds 12.5\n")
            if path == "/metrics":
                raise urllib.error.HTTPError(path, 400, "model name missing", {}, None)
            raise AssertionError("unexpected path " + path)
        self.tr._get = fake_get


class TestRouterMetricsScrape(RouterCase):
    def test_router_up_and_tokens_attributed(self):
        self._wire(prompt=10, gen=20)
        self.tr.poll_once()                       # baseline
        self.assertTrue(self.tr.live["router_up"])
        self._wire(prompt=15, gen=60)             # counters advanced
        self.tr.poll_once()
        m = self.tr.data["models"]["nomic"]
        self.assertEqual(m["prompt"], 5)
        self.assertEqual(m["generated"], 40)
        self.assertGreater(self.tr.live["gen_per_sec"], 0)

    def test_scrape_includes_model_param_never_bare(self):
        seen = []
        self._wire(prompt=1, gen=1, seen=seen)
        self.tr.poll_once()
        self.assertIn("/models", seen)
        self.assertTrue(any(p.startswith("/metrics?model=") for p in seen),
                        f"never scraped with ?model=; saw {seen}")
        self.assertNotIn("/metrics", seen)        # bare form must not be used

    def test_router_up_with_no_model_loaded(self):
        def fake_get(path, timeout=4):
            if path == "/models":
                return json.dumps({"data": [{"id": "default", "status": {"value": "unloaded"}}]})
            raise AssertionError("should not scrape metrics with nothing loaded")
        self.tr._get = fake_get
        self.tr.poll_once()
        self.assertTrue(self.tr.live["router_up"])   # up, just idle
        self.assertIsNone(self.tr.live["loaded_model"])

    def test_router_down_reports_offline(self):
        def fake_get(path, timeout=4):
            raise urllib.error.URLError("connection refused")
        self.tr._get = fake_get
        self.tr.poll_once()
        self.assertFalse(self.tr.live["router_up"])


if __name__ == "__main__":
    unittest.main()
