import conftest_paths  # noqa: F401
import os, unittest
from unittest import mock
import routes


class TestRecommendPayload(unittest.TestCase):
    def test_builds_payload_from_model_and_hardware(self):
        fake_models = {"models": [{"id": "qwen", "model": "/models/qwen.gguf"}]}
        with mock.patch.object(routes, "model_state", return_value=fake_models), \
             mock.patch.object(routes.gguf, "metadata",
                               return_value={"block_count": 32, "context_length": 128000}), \
             mock.patch.object(routes.hardware, "detect_gpus",
                               return_value=[{"vram_mib": 24000, "compute_cap": "8.6"}]), \
             mock.patch.object(routes.hardware, "detect_cpu",
                               return_value={"threads": 24, "cores": 12}), \
             mock.patch.object(routes.os.path, "getsize", return_value=5 * 1024 * 1024 * 1024):
            out = routes._autotune_recommend({"model": "qwen", "intent": "context"})
        self.assertEqual(out["model"], "qwen")
        self.assertEqual(out["knobs"]["cache-type-k"], "q8_0")
        self.assertEqual(out["knobs"]["fit-ctx"], "128000")

    def test_resolves_model_path_from_settings_nested(self):
        """Covers the settings.model path-resolution branch (real model_state shape)."""
        fake_models = {"models": [{"id": "qwen", "settings": {"model": "/models/qwen.gguf"}}]}
        with mock.patch.object(routes, "model_state", return_value=fake_models), \
             mock.patch.object(routes.gguf, "metadata",
                               return_value={"block_count": 32, "context_length": 128000}), \
             mock.patch.object(routes.hardware, "detect_gpus",
                               return_value=[{"vram_mib": 24000, "compute_cap": "8.6"}]), \
             mock.patch.object(routes.hardware, "detect_cpu",
                               return_value={"threads": 24, "cores": 12}), \
             mock.patch.object(routes.os.path, "getsize", return_value=5 * 1024 * 1024 * 1024) as mock_getsize:
            out = routes._autotune_recommend({"model": "qwen", "intent": "context"})
        # Verify getsize was called with the resolved path from settings.model
        mock_getsize.assert_called_with("/models/qwen.gguf")
        # Verify result is a proper recommendation
        self.assertEqual(out["model"], "qwen")
        self.assertEqual(out["knobs"]["fit"], "on")

    def test_unknown_model_returns_error(self):
        with mock.patch.object(routes, "model_state", return_value={"models": []}):
            out = routes._autotune_recommend({"model": "nope", "intent": "balanced"})
        self.assertIn("error", out)


if __name__ == "__main__":
    unittest.main()
