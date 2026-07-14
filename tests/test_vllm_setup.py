import conftest_paths  # noqa: F401
import unittest
from unittest import mock
import vllm_setup


class TestStatus(unittest.TestCase):
    def test_wsl_absent(self):
        with mock.patch("wsl.list_distros", return_value=[]):
            st = vllm_setup.status(distro="")
        self.assertFalse(st["wsl"]["present"])
        self.assertFalse(st["vllm"]["present"])

    def test_gpu_and_vllm_present(self):
        distros = [{"name": "Ubuntu", "state": "Running", "version": "2", "default": True}]
        def fake_run(cmd, distro=None, timeout=60):
            if "nvidia-smi" in cmd:
                return 0, "NVIDIA-SMI 560.00  GPU 0: NVIDIA RTX 5090\n", ""
            if "vllm" in cmd and "--version" in cmd:
                return 0, "0.6.3\n", ""
            return 0, "", ""
        with mock.patch("wsl.list_distros", return_value=distros), \
             mock.patch("wsl.run", side_effect=fake_run):
            st = vllm_setup.status(distro="Ubuntu")
        self.assertTrue(st["wsl"]["present"])
        self.assertTrue(st["gpu"]["present"])
        self.assertTrue(st["vllm"]["present"])
        self.assertEqual(st["vllm"]["version"], "0.6.3")

    def test_vllm_missing_when_venv_check_fails(self):
        distros = [{"name": "Ubuntu", "state": "Running", "version": "2", "default": True}]
        def fake_run(cmd, distro=None, timeout=60):
            if "nvidia-smi" in cmd:
                return 0, "GPU 0: X\n", ""
            return 1, "", "No such file"     # vllm --version fails
        with mock.patch("wsl.list_distros", return_value=distros), \
             mock.patch("wsl.run", side_effect=fake_run):
            st = vllm_setup.status(distro="Ubuntu")
        self.assertFalse(st["vllm"]["present"])


class TestInstallScript(unittest.TestCase):
    def test_install_script_is_sudo_free_and_uses_uv(self):
        script = vllm_setup.install_script()
        self.assertNotIn("sudo", script)
        self.assertIn("astral.sh/uv", script)
        self.assertIn("uv venv", script)
        self.assertIn("uv pip install vllm", script)
        self.assertIn(".llamaforge/vllm-venv", script)


if __name__ == "__main__":
    unittest.main()
