import conftest_paths  # noqa: F401
import os, tempfile, time, unittest
from unittest import mock
import vllm_job


class TestWslJob(unittest.TestCase):
    def setUp(self):
        self.logdir = tempfile.mkdtemp()
        self.job = vllm_job.WslJob(self.logdir, "vllm-setup.log")

    def test_starts_only_once(self):
        with mock.patch("wsl.popen") as popen:
            popen.return_value = mock.Mock(wait=lambda: 0, returncode=0)
            self.assertTrue(self.job.start("echo hi", distro="Ubuntu"))
            self.job.state["running"] = True     # simulate still running
            self.assertFalse(self.job.start("echo hi", distro="Ubuntu"))

    def test_state_shape(self):
        s = self.job.progress()
        self.assertIn("running", s)
        self.assertIn("phase", s)


if __name__ == "__main__":
    unittest.main()
