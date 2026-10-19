import conftest_paths  # noqa: F401
import json, os, tempfile, unittest, urllib.error, urllib.parse
import stats
from unittest import mock


def _model_of(path):
    """The model a /metrics?model=...&... scrape names."""
    return urllib.parse.parse_qs(urllib.parse.urlsplit(path).query)["model"][0]


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
    def test_router_scrapes_include_configured_bearer_key(self):
        seen = {}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self):
                return b'{"data": []}'

        def open_request(request, timeout=4):
            seen["authorization"] = (request.get_header("Authorization")
                                      if hasattr(request, "get_header") else None)
            return Response()

        with mock.patch.object(stats.config, "load", return_value={
                "router_port": 8080, "router_api_key": "secret-key"}), \
             mock.patch.object(stats.urllib.request, "urlopen",
                               side_effect=open_request):
            self.tr._get("/models")

        self.assertEqual(seen["authorization"], "Bearer secret-key")

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

    def test_scrape_never_loads_a_model(self):
        # /models said loaded, then something unloaded it before the scrape:
        # without autoload=false the router reloads it to answer /metrics.
        seen = []
        self._wire(prompt=1, gen=1, seen=seen)
        self.tr.poll_once()
        scrapes = [p for p in seen if p.startswith("/metrics?model=")]
        self.assertTrue(scrapes)
        for p in scrapes:
            self.assertTrue(p.endswith("&autoload=false"), p)

    def test_router_up_with_no_model_loaded(self):
        def fake_get(path, timeout=4):
            if path == "/models":
                return json.dumps({"data": [{"id": "default", "status": {"value": "unloaded"}}]})
            raise AssertionError("should not scrape metrics with nothing loaded")
        self.tr._get = fake_get
        self.tr.poll_once()
        self.assertTrue(self.tr.live["router_up"])   # up, just idle
        self.assertIsNone(self.tr.live["loaded_model"])

    def test_reset_rebaselines(self):
        self._wire(prompt=10, gen=20)
        self.tr.poll_once()
        self.tr.reset()
        self._wire(prompt=15, gen=60)
        self.tr.poll_once()                       # a fresh baseline, not a delta
        self.assertEqual(self.tr.data["models"]["nomic"]["generated"], 0)

    def test_router_down_reports_offline(self):
        def fake_get(path, timeout=4):
            raise urllib.error.URLError("connection refused")
        self.tr._get = fake_get
        self.tr.poll_once()
        self.assertFalse(self.tr.live["router_up"])


class TestPool(RouterCase):
    """With a multi-model pool several models are loaded at once; each has its
    own /metrics, so each keeps its own baseline."""

    def _pool(self, counters, main=""):
        """counters: {model: (prompt, gen, gen_per_sec)} for the loaded models."""
        def fake_get(path, timeout=4):
            if path == "/models":
                return json.dumps({"data": [{"id": "default", "status": {"value": "unloaded"}}] + [
                    {"id": m, "status": {"value": "loaded"}} for m in counters]})
            mid = _model_of(path)
            p, g, tps = counters[mid]
            return (f"llamacpp:prompt_tokens_total {p}\n"
                    f"llamacpp:tokens_predicted_total {g}\n"
                    f"llamacpp:predicted_tokens_seconds {tps}\n"
                    f"llamacpp:requests_processing 1\n")
        self.tr._get = fake_get
        self.tr._main = lambda: main

    def test_each_model_gets_its_own_tokens(self):
        self._pool({"big": (10, 100, 5.0), "small": (0, 0, 0.0)})
        self.tr.poll_once()
        self._pool({"big": (10, 150, 5.0), "small": (40, 300, 60.0)})
        self.tr.poll_once()
        models = self.tr.data["models"]
        self.assertEqual((models["big"]["prompt"], models["big"]["generated"]), (0, 50))
        self.assertEqual((models["small"]["prompt"], models["small"]["generated"]), (40, 300))

    def test_a_model_that_unloads_starts_over_when_it_returns(self):
        self._pool({"big": (0, 100, 0.0), "small": (0, 500, 0.0)})
        self.tr.poll_once()
        self._pool({"big": (0, 120, 0.0)})          # small unloaded
        self.tr.poll_once()
        self._pool({"big": (0, 120, 0.0), "small": (0, 30, 0.0)})   # back, counters from 0
        self.tr.poll_once()
        self.assertEqual(self.tr.data["models"]["small"]["generated"], 0)
        self.assertEqual(self.tr.data["models"]["big"]["generated"], 20)

    def test_runs_are_counted_per_model(self):
        self._pool({"big": (0, 0, 0.0), "small": (0, 0, 0.0)})
        self.tr.poll_once()
        self._pool({"big": (0, 10, 0.0), "small": (0, 0, 0.0)})
        self.tr.poll_once()
        self._pool({"big": (0, 20, 0.0), "small": (0, 10, 0.0)})    # big still going, small starts
        self.tr.poll_once()
        self.assertEqual(self.tr.data["models"]["big"]["runs"], 1)
        self.assertEqual(self.tr.data["models"]["small"]["runs"], 1)

    def test_live_lists_them_all_with_the_main_first(self):
        self._pool({"small": (0, 0, 60.0), "big": (0, 0, 5.0)}, main="big")
        self.tr.poll_once()
        live = self.tr.live
        self.assertEqual(live["loaded_model"], "big")
        self.assertEqual(live["loaded_models"], ["big", "small"])
        self.assertEqual(live["gen_per_sec"], 65.0)
        self.assertEqual(live["requests_processing"], 2)

    def test_a_failed_scrape_keeps_the_baseline(self):
        self._pool({"big": (0, 1000, 0.0)})
        self.tr.poll_once()
        good = self.tr._get

        def flaky(path, timeout=4):
            if path.startswith("/metrics"):
                raise TimeoutError("busy")
            return good(path, timeout)
        self.tr._get = flaky
        self.tr.poll_once()
        self._pool({"big": (0, 1010, 0.0)})
        self.tr.poll_once()
        self.assertEqual(self.tr.data["models"]["big"]["generated"], 10)

    def test_without_a_main_the_first_loaded_is_named(self):
        self._pool({"small": (0, 0, 0.0), "big": (0, 0, 0.0)})
        self.tr.poll_once()
        self.assertEqual(self.tr.live["loaded_model"], "small")


class TestProcessSlots(RouterCase):
    """A model pinned to another build (ik_llama.cpp, an older llama.cpp) runs
    in its own llama-server process on its own port, outside the router. It has
    its own bare /metrics, and has to be counted like a router model."""

    def _wire_all(self, router, procs, router_up=True, seen=None):
        """router: {model: (prompt, gen, tps)} loaded in the router;
        procs: {model: (prompt, gen, tps)} running as process slots."""
        def metrics(c):
            p, g, tps = c
            return (f"llamacpp:prompt_tokens_total {p}\n"
                    f"llamacpp:tokens_predicted_total {g}\n"
                    f"llamacpp:prompt_tokens_seconds {tps * 10}\n"
                    f"llamacpp:predicted_tokens_seconds {tps}\n"
                    f"llamacpp:requests_processing 1\n")

        def fake_get(path, timeout=4):
            if not router_up:
                raise urllib.error.URLError("connection refused")
            if path == "/models":
                return json.dumps({"data": [{"id": m, "status": {"value": "loaded"}} for m in router]})
            return metrics(router[_model_of(path)])

        ports = {m: f"http://127.0.0.1:{8100 + i}" for i, m in enumerate(procs)}
        by_port = {v: k for k, v in ports.items()}

        def fake_proc(endpoint, path, timeout=4):
            if seen is not None:
                seen.append(endpoint + path)
            return metrics(procs[by_port[endpoint]])

        self.tr._get = fake_get
        self.tr._get_proc = fake_proc
        self.tr.proc_source = lambda: dict(ports)
        self.tr._main = lambda: ""

    def test_process_slot_tokens_are_counted(self):
        self._wire_all({"big": (0, 100, 5.0)}, {"ik-moe": (0, 10, 9.0)})
        self.tr.poll_once()
        self._wire_all({"big": (0, 150, 5.0)}, {"ik-moe": (20, 70, 9.0)})
        self.tr.poll_once()
        models = self.tr.data["models"]
        self.assertEqual(models["big"]["generated"], 50)
        self.assertEqual((models["ik-moe"]["prompt"], models["ik-moe"]["generated"]), (20, 60))

    def test_process_slots_scrape_their_own_bare_metrics(self):
        seen = []
        self._wire_all({}, {"ik-moe": (0, 0, 0.0)}, seen=seen)
        self.tr.poll_once()
        self.assertEqual(seen, ["http://127.0.0.1:8100/metrics"])

    def test_process_slots_count_while_the_router_is_down(self):
        self._wire_all({}, {"ik-moe": (0, 10, 4.0)}, router_up=False)
        self.tr.poll_once()
        self._wire_all({}, {"ik-moe": (0, 30, 4.0)}, router_up=False)
        self.tr.poll_once()
        self.assertFalse(self.tr.live["router_up"])
        self.assertEqual(self.tr.live["loaded_models"], ["ik-moe"])
        self.assertEqual(self.tr.data["models"]["ik-moe"]["generated"], 20)

    def test_live_reports_each_model_separately(self):
        self._wire_all({"big": (0, 0, 5.0)}, {"ik-moe": (0, 0, 9.0)})
        self.tr.poll_once()
        live = self.tr.live
        self.assertEqual(live["loaded_models"], ["big", "ik-moe"])
        self.assertEqual(live["gen_per_sec"], 14.0)
        self.assertEqual(live["models"], [
            {"id": "big", "where": "router", "gen_per_sec": 5.0,
             "prompt_per_sec": 50.0, "requests_processing": 1},
            {"id": "ik-moe", "where": "process", "gen_per_sec": 9.0,
             "prompt_per_sec": 90.0, "requests_processing": 1},
        ])

    def test_a_model_the_router_already_serves_is_scraped_once(self):
        seen = []
        self._wire_all({"big": (0, 0, 5.0)}, {"big": (0, 0, 5.0)}, seen=seen)
        self.tr.poll_once()
        self.assertEqual(seen, [])
        self.assertEqual([m["id"] for m in self.tr.live["models"]], ["big"])

    def test_a_broken_process_source_is_ignored(self):
        self._wire_all({"big": (0, 0, 5.0)}, {})
        def boom():
            raise RuntimeError("slot table unreadable")
        self.tr.proc_source = boom
        self.tr.poll_once()
        self.assertEqual(self.tr.live["loaded_models"], ["big"])

    def test_process_scrapes_carry_the_bearer_key(self):
        seen = {}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self):
                return b""

        def open_request(request, timeout=4):
            seen["url"] = request.full_url
            seen["authorization"] = request.get_header("Authorization")
            return Response()

        with mock.patch.object(stats.config, "load", return_value={
                "router_port": 8080, "router_api_key": "secret-key"}), \
             mock.patch.object(stats.urllib.request, "urlopen", side_effect=open_request):
            self.tr._get_proc("http://127.0.0.1:8100", "/metrics")
        self.assertEqual(seen, {"url": "http://127.0.0.1:8100/metrics",
                                "authorization": "Bearer secret-key"})


if __name__ == "__main__":
    unittest.main()
