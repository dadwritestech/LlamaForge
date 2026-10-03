import conftest_paths  # noqa: F401
import unittest
import autotune

MIB = 1024 * 1024


def gpu(vram_mib, cc="8.6"):
    return {"vram_mib": vram_mib, "compute_cap": cc}


class TestRecommendCore(unittest.TestCase):
    """llama.cpp's --fit (default on) sizes ctx, GPU layers, the multi-GPU split
    and MoE expert placement at load - but gives up entirely if any of ctx-size,
    n-gpu-layers or tensor-split is pinned. So autotune never pins them; it
    returns them blank (= unset) so applying a recommendation clears old pins."""

    def test_no_gpu_lets_flash_attention_decide(self):
        hw = {"gpus": [], "cpu": {"threads": 16, "cores": 8}}
        r = autotune.recommend({"block_count": 32}, hw, "balanced")
        self.assertEqual(r["knobs"]["flash-attn"], "auto")   # CPU has FA kernels too
        self.assertEqual(r["knobs"]["threads"], "16")

    def test_never_pins_what_fit_sizes(self):
        for gpus in ([gpu(24000)], [gpu(8000)], [gpu(16000), gpu(16000)], []):
            for size in (5, 40):
                r = autotune.recommend({"block_count": 80, "context_length": 131072},
                                       {"gpus": gpus, "cpu": {}}, "balanced",
                                       size_bytes=size * 1024 * MIB)
                for k in ("n-gpu-layers", "tensor-split", "ctx-size"):
                    self.assertEqual(r["knobs"].get(k), "", (gpus, size, k))
                self.assertEqual(r["knobs"]["fit"], "on")
                self.assertEqual(r["knobs"]["flash-attn"], "auto")

    def test_rationale_present_for_each_knob(self):
        hw = {"gpus": [gpu(24000)], "cpu": {"threads": 24, "cores": 12}}
        r = autotune.recommend({"block_count": 32}, hw, "balanced",
                               size_bytes=5 * 1024 * MIB)
        for k in r["knobs"]:
            self.assertIn(k, r["rationale"])
            self.assertTrue(r["rationale"][k])

    def test_unknown_meta_degrades_gracefully(self):
        hw = {"gpus": [gpu(24000)], "cpu": {"threads": 24, "cores": 12}}
        r = autotune.recommend({}, hw, "balanced", size_bytes=None)
        self.assertEqual(r["knobs"]["fit"], "on")


if __name__ == "__main__":
    unittest.main()
