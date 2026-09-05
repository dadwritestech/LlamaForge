import conftest_paths  # noqa: F401
import unittest
import routes, hub, vram_predict, config


class TestHubFilesPredict(unittest.TestCase):
    def setUp(self):
        self._files = hub.files
        self._pred = vram_predict.predict_remote
        self._load = config.load
        hub.files = lambda repo, vram: {
            "files": [{"path": "M-Q4_K_M.gguf", "size": int(20e9), "shards": 1, "fit": "offload"}],
            "mmproj": []}
        vram_predict.predict_remote = lambda **kw: {
            "regime": "hybrid", "tok_s": 21.0, "confidence": "high", "note": "ok",
            "usability": "interactive"}
        config.load = lambda: {"vram_predict_enabled": True}

    def tearDown(self):
        hub.files = self._files
        vram_predict.predict_remote = self._pred
        config.load = self._load

    def test_files_include_prediction(self):
        req = routes.Req(body={"repo": "acme/model"})
        status, payload = routes.post_hub_files(req)
        self.assertEqual(status, 200)
        f0 = payload["files"][0]
        self.assertIn("predict", f0)
        self.assertEqual(f0["predict"]["regime"], "hybrid")

    def test_usable_hybrid_prediction_replaces_naive_offload_label(self):
        status, payload = routes.post_hub_files(routes.Req(body={"repo": "acme/model"}))

        self.assertEqual(status, 200)
        self.assertEqual(payload["files"][0]["fit"], "tight")

    def test_disabled_toggle_omits_prediction(self):
        config.load = lambda: {"vram_predict_enabled": False}
        req = routes.Req(body={"repo": "acme/model"})
        status, payload = routes.post_hub_files(req)
        self.assertNotIn("predict", payload["files"][0])


if __name__ == "__main__":
    unittest.main()
