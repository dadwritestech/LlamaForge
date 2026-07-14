import conftest_paths  # noqa: F401
import unittest
from unittest import mock
import vllm_hub


class TestFit(unittest.TestCase):
    def test_fits_tight_wont(self):
        vram = 64 * 1024   # MiB (2x ~32GB)
        gb = 1024 * 1024 * 1024
        self.assertEqual(vllm_hub.fit(20 * gb, vram), "fits")
        self.assertEqual(vllm_hub.fit(55 * gb, vram), "tight")
        self.assertEqual(vllm_hub.fit(80 * gb, vram), "wont")

    def test_unknown_when_no_vram(self):
        self.assertEqual(vllm_hub.fit(10, 0), "unknown")


class TestQuant(unittest.TestCase):
    def test_nvfp4_from_config(self):
        cfg = {"quantization_config": {"quant_method": "modelopt",
                                       "quant_algo": "NVFP4"}}
        self.assertEqual(vllm_hub.detect_quant(cfg), "nvfp4")

    def test_awq(self):
        self.assertEqual(vllm_hub.detect_quant(
            {"quantization_config": {"quant_method": "awq"}}), "awq")

    def test_fp8(self):
        self.assertEqual(vllm_hub.detect_quant(
            {"quantization_config": {"quant_method": "fp8"}}), "fp8")

    def test_unquantized_is_dtype(self):
        self.assertEqual(vllm_hub.detect_quant({"torch_dtype": "bfloat16"}), "bf16")

    def test_missing_defaults_unknown(self):
        self.assertEqual(vllm_hub.detect_quant({}), "unknown")


class TestFiles(unittest.TestCase):
    def test_sums_safetensor_shards_and_tags(self):
        tree = [
            {"path": "model-00001-of-00002.safetensors", "size": 5 * 10**9},
            {"path": "model-00002-of-00002.safetensors", "size": 4 * 10**9},
            {"path": "config.json", "size": 20000},
            {"path": "tokenizer.json", "size": 20 * 10**6},
        ]
        config_json = {"quantization_config": {"quant_algo": "NVFP4",
                                               "quant_method": "modelopt"}}
        def fake_get_json(url, timeout=25):
            if url.endswith("/tree/main?recursive=1") or "tree/main" in url:
                return tree
            if url.endswith("config.json"):
                return config_json
            return {}
        with mock.patch.object(vllm_hub, "_get_json", side_effect=fake_get_json):
            info = vllm_hub.repo_info("org/Model-NVFP4", vram_mib=64 * 1024)
        self.assertEqual(info["size_bytes"], 9 * 10**9)
        self.assertEqual(info["quant"], "nvfp4")
        self.assertEqual(info["fit"], "fits")
        self.assertTrue(info["is_safetensors"])


if __name__ == "__main__":
    unittest.main()
