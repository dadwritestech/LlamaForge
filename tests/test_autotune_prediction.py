import conftest_paths  # noqa: F401
import unittest
import autotune


class TestAutotunePrediction(unittest.TestCase):
    def test_prediction_surfaced_in_rationale(self):
        meta = {"block_count": 32, "context_length": 8192}
        hw = {"gpus": [{"vram_mib": 24000}], "cpu": {"threads": 16}}
        pred = {"regime": "hybrid", "tok_s": 18.0, "confidence": "high",
                "note": "Offloads ~8 GB of experts to CPU+RAM."}
        rec = autotune.recommend(meta, hw, "balanced", size_bytes=int(20e9),
                                 prediction=pred)
        self.assertIn("prediction", rec)
        self.assertEqual(rec["prediction"]["regime"], "hybrid")
        self.assertIn("~18.0 tok/s", rec["rationale"]["fit"])

    def test_no_prediction_keeps_old_shape(self):
        meta = {"block_count": 32}
        hw = {"gpus": [], "cpu": {}}
        rec = autotune.recommend(meta, hw, "balanced")
        self.assertNotIn("prediction", rec)
        self.assertIn("knobs", rec)


if __name__ == "__main__":
    unittest.main()
