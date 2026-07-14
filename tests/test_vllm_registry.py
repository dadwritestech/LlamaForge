import conftest_paths  # noqa: F401
import os, tempfile, unittest
import vllm_registry as reg


class TestRegistry(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "vllm_models.json")

    def test_load_missing_returns_scaffold(self):
        data = reg.load(self.path)
        self.assertEqual(data, {"*": {}})

    def test_upsert_then_load_roundtrip(self):
        reg.upsert("Qwen/Qwen3-8B",
                   {"repo": "Qwen/Qwen3-8B", "wsl_path": "", "size_bytes": 16000000000,
                    "quant": "bf16"}, path=self.path)
        data = reg.load(self.path)
        self.assertIn("Qwen/Qwen3-8B", data)
        self.assertEqual(data["Qwen/Qwen3-8B"]["quant"], "bf16")
        self.assertEqual(data["Qwen/Qwen3-8B"]["settings"], {})

    def test_set_settings_merges_and_drops_blanks(self):
        reg.upsert("m", {"repo": "m"}, path=self.path)
        reg.set_settings("m", {"tensor-parallel-size": "2", "max-model-len": ""},
                         path=self.path)
        s = reg.load(self.path)["m"]["settings"]
        self.assertEqual(s, {"tensor-parallel-size": "2"})   # blank removed

    def test_global_settings_via_star(self):
        reg.set_settings("*", {"gpu-memory-utilization": "0.9"}, path=self.path)
        self.assertEqual(reg.load(self.path)["*"], {"gpu-memory-utilization": "0.9"})

    def test_remove(self):
        reg.upsert("m", {"repo": "m"}, path=self.path)
        reg.remove("m", path=self.path)
        self.assertNotIn("m", reg.load(self.path))

    def test_effective_settings_merges_global_then_model(self):
        reg.set_settings("*", {"gpu-memory-utilization": "0.9",
                               "tensor-parallel-size": "1"}, path=self.path)
        reg.upsert("m", {"repo": "m"}, path=self.path)
        reg.set_settings("m", {"tensor-parallel-size": "2"}, path=self.path)
        eff = reg.effective_settings("m", path=self.path)
        self.assertEqual(eff, {"gpu-memory-utilization": "0.9",
                               "tensor-parallel-size": "2"})


if __name__ == "__main__":
    unittest.main()
