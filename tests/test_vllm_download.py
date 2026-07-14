import conftest_paths  # noqa: F401
import unittest
from unittest import mock
import vllm_download


class TestScripts(unittest.TestCase):
    def test_download_command_uses_hf_download_in_venv(self):
        cmd = vllm_download.download_cmd("Qwen/Qwen3-8B")
        self.assertIn("hf download Qwen/Qwen3-8B", cmd)
        self.assertIn(".llamaforge/vllm-venv/bin", cmd)

    def test_cache_dir_name_matches_hf_convention(self):
        self.assertEqual(vllm_download.cache_dirname("Qwen/Qwen3-8B"),
                         "models--Qwen--Qwen3-8B")

    def test_delete_command_targets_cache_dir(self):
        cmd = vllm_download.delete_cmd("Qwen/Qwen3-8B")
        self.assertIn("models--Qwen--Qwen3-8B", cmd)
        self.assertIn("rm -rf", cmd)
        self.assertNotIn("sudo", cmd)


class TestManager(unittest.TestCase):
    def test_start_guards_single_job(self):
        mgr = vllm_download.Manager(distro="Ubuntu")
        with mock.patch("wsl.popen") as popen, mock.patch("wsl.run",
                        return_value=(0, "0\n", "")):
            popen.return_value = mock.Mock(wait=lambda: 0)
            self.assertTrue(mgr.start("Qwen/Qwen3-8B", expected_bytes=1000))
            mgr.state["running"] = True
            self.assertFalse(mgr.start("other/model", expected_bytes=1000))


if __name__ == "__main__":
    unittest.main()
