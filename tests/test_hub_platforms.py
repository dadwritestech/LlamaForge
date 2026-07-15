import conftest_paths  # noqa: F401
import unittest
import hub, vllm_hub, server


class TestPlatformTags(unittest.TestCase):
    def test_gguf_runs_everywhere(self):
        self.assertEqual(hub.PLATFORMS, ["windows", "linux", "macos"])

    def test_safetensors_has_no_macos(self):
        self.assertNotIn("macos", vllm_hub.PLATFORMS)
        self.assertIn("windows", vllm_hub.PLATFORMS)


class TestInstalledRepos(unittest.TestCase):
    def test_gguf_repo_matched_via_download_folder_in_ini_path(self):
        results = [{"repo": "unsloth/Qwen3-8B-GGUF"}, {"repo": "other/Model-GGUF"}]
        ini = {"qwen3-8b": {"model": "D:/models/LlamaForge-downloads/unsloth--Qwen3-8B-GGUF/q4.gguf"}}
        inst = server.installed_repos(results, ini, [])
        self.assertEqual(inst, ["unsloth/Qwen3-8B-GGUF"])

    def test_vllm_repo_matched_by_registry_id(self):
        results = [{"repo": "Qwen/Qwen3-8B-FP8"}]
        inst = server.installed_repos(results, {}, ["Qwen/Qwen3-8B-FP8"])
        self.assertEqual(inst, ["Qwen/Qwen3-8B-FP8"])

    def test_no_match(self):
        inst = server.installed_repos([{"repo": "a/b"}], {"m": {"model": "c:/x.gguf"}}, [])
        self.assertEqual(inst, [])


if __name__ == "__main__":
    unittest.main()
