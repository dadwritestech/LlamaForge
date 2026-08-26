import conftest_paths  # noqa: F401
import unittest
import vram_predict as vp
import osplat
from vramwise import constants as C


class TestBuildHardware(unittest.TestCase):
    def test_preset_bandwidth_from_gpu_name(self):
        self.assertEqual(vp._preset_vram_bw("NVIDIA GeForce RTX 4090"), 1008)
        self.assertEqual(vp._preset_vram_bw("NVIDIA GeForce RTX 5080"), 960)

    def test_unknown_gpu_falls_back_to_default(self):
        self.assertEqual(vp._preset_vram_bw("Some Weird Card"), C.DEFAULT_VRAM_BW)

    def test_overrides_win(self):
        cfg = {"vram_bandwidths": {"ram_bw": 80, "disk_bw": 12}}
        hw = vp.build_hardware(cfg=cfg, gpus=[{"name": "RTX 4090", "vram_mib": 24576}],
                               ram_gb=64.0)
        self.assertEqual(hw.ram_bw, 80)
        self.assertEqual(hw.disk_bw, 12)
        self.assertEqual(hw.vram_bw, 1008)
        self.assertAlmostEqual(hw.vram_gb, 24.0, delta=0.1)
        self.assertEqual(hw.ram_gb, 64.0)

    def test_apple_silicon_uses_metal_budget_without_ram_spill(self):
        original_is_mac = osplat.IS_MAC
        original_mem = osplat.mac_mem_bytes
        osplat.IS_MAC = True
        osplat.mac_mem_bytes = lambda: 32 * 1024 ** 3
        try:
            hw = vp.build_hardware(cfg={})
        finally:
            osplat.IS_MAC = original_is_mac
            osplat.mac_mem_bytes = original_mem

        self.assertAlmostEqual(hw.vram_gb, 32 * osplat.METAL_BUDGET, places=1)
        self.assertEqual(hw.ram_gb, 0.0)


class TestPredictLocal(unittest.TestCase):
    def _hw(self):
        return vp.build_hardware(cfg={}, gpus=[{"name": "RTX 5090", "vram_mib": 32768}],
                                 ram_gb=64.0)

    def test_dense_model_from_meta(self):
        meta = {"name": "llama-8b", "quantization": "Q4_K_M", "block_count": 32}
        size = int(8e9 * 4.9 / 8)
        out = vp.predict_local("x.gguf", size_bytes=size, cfg={}, hw=self._hw(), meta=meta)
        self.assertEqual(out["regime"], "gpu-resident")
        self.assertEqual(out["confidence"], "high")
        self.assertGreater(out["tok_s"], 0)

    def test_moe_active_less_than_total(self):
        meta = {"name": "moe", "quantization": "Q4_K_M", "block_count": 48,
                "expert_count": 128, "expert_used_count": 8}
        m, conf = vp._model_from_gguf(meta, int(100e9 * 4.9 / 8))
        self.assertTrue(m.is_moe)
        self.assertLess(m.active_params, m.total_params)
        self.assertEqual(conf, "high")

    def test_missing_size_is_unknown(self):
        out = vp.predict_local("x.gguf", size_bytes=0, cfg={}, hw=self._hw(), meta={})
        self.assertEqual(out["confidence"], "unknown")
        self.assertIsNone(out["tok_s"])

    def test_unknown_quant_degrades_to_estimate(self):
        meta = {"name": "m", "quantization": "ZZ_WEIRD", "block_count": 32}
        m, conf = vp._model_from_gguf(meta, int(7e9 * 4.8 / 8))
        self.assertEqual(conf, "estimate")
        self.assertIsNotNone(m)


class TestPredictRemote(unittest.TestCase):
    def _hw(self):
        return vp.build_hardware(cfg={}, gpus=[{"name": "RTX 5090", "vram_mib": 32768}],
                                 ram_gb=64.0)

    def test_moe_config_maps_active(self):
        cfgj = {"num_parameters": 106e9, "num_hidden_layers": 47,
                "num_experts": 128, "num_experts_per_tok": 8}
        m, conf = vp._model_from_hf(cfgj, "q4_k_m", None)
        self.assertLess(m.active_params, m.total_params)
        self.assertEqual(m.n_layers, 47)
        self.assertEqual(conf, "high")

    def test_dense_config_estimates_params(self):
        cfgj = {"hidden_size": 4096, "num_hidden_layers": 32,
                "intermediate_size": 14336, "vocab_size": 128000}
        m, conf = vp._model_from_hf(cfgj, "q4_k_m", None)
        self.assertGreater(m.total_params, 6e9)
        self.assertEqual(m.active_params, m.total_params)

    def test_fetch_failure_uses_size_fallback(self):
        orig = vp._fetch_hf_config
        vp._fetch_hf_config = lambda repo: None
        try:
            out = vp.predict_remote("acme/model", quant="q4_k_m",
                                    size_bytes=int(20e9), cfg={}, hw=self._hw())
            self.assertEqual(out["source"], "size-fallback")
            self.assertEqual(out["confidence"], "low")
            self.assertIsNotNone(out["tok_s"])
        finally:
            vp._fetch_hf_config = orig

    def test_fetch_failure_no_size_is_unknown(self):
        orig = vp._fetch_hf_config
        vp._fetch_hf_config = lambda repo: None
        try:
            out = vp.predict_remote("acme/model", quant="q4_k_m", cfg={}, hw=self._hw())
            self.assertEqual(out["confidence"], "unknown")
        finally:
            vp._fetch_hf_config = orig


class TestCacheKeying(unittest.TestCase):
    def test_override_change_busts_cache(self):
        orig = vp._fetch_hf_config
        vp._CACHE.clear()
        vp._CFG_CACHE.clear()
        vp._fetch_hf_config = lambda repo: {"num_parameters": 8e9, "num_hidden_layers": 32}
        try:
            hw1 = vp.build_hardware(cfg={"vram_bandwidths": {"ram_bw": 40}},
                                    gpus=[{"name": "RTX 5090", "vram_mib": 32768}], ram_gb=64.0)
            hw2 = vp.build_hardware(cfg={"vram_bandwidths": {"ram_bw": 80}},
                                    gpus=[{"name": "RTX 5090", "vram_mib": 32768}], ram_gb=64.0)
            self.assertNotEqual(vp._hw_sig(hw1), vp._hw_sig(hw2))
            vp.predict_remote("r/m", hw=hw1)
            vp.predict_remote("r/m", hw=hw2)
            remote_keys = [k for k in vp._CACHE if k[0] == "remote"]
            self.assertEqual(len(remote_keys), 2)
        finally:
            vp._fetch_hf_config = orig
            vp._CACHE.clear()
            vp._CFG_CACHE.clear()


if __name__ == "__main__":
    unittest.main()
