import conftest_paths  # noqa: F401
import unittest
from unittest import mock
import vllm_ctl


class TestFlagBuilding(unittest.TestCase):
    def test_settings_to_flags(self):
        flags = vllm_ctl.settings_to_flags({
            "tensor-parallel-size": "2",
            "gpu-memory-utilization": "0.9",
            "enforce-eager": "true",
            "disable-log-stats": "false",
        })
        self.assertIn("--tensor-parallel-size 2", flags)
        self.assertIn("--gpu-memory-utilization 0.9", flags)
        self.assertIn("--enforce-eager", flags)
        self.assertNotIn("disable-log-stats", flags)   # false -> omitted

    def test_serve_command_uses_wsl_path_when_present(self):
        cmd = vllm_ctl.build_serve_cmd(
            venv="/home/u/.llamaforge/vllm-venv",
            model_ref="/home/u/.cache/huggingface/models--x",
            port=8081, flag_str="--tensor-parallel-size 2")
        self.assertIn("/home/u/.llamaforge/vllm-venv/bin/vllm serve", cmd)
        self.assertIn("/home/u/.cache/huggingface/models--x", cmd)
        self.assertIn("--port 8081", cmd)
        self.assertIn("--host 0.0.0.0", cmd)


class TestManagerLifecycle(unittest.TestCase):
    def setUp(self):
        self.mgr = vllm_ctl.Manager(distro="Ubuntu", port=8081,
                                     venv="/home/u/.llamaforge/vllm-venv",
                                     logdir="/tmp/lf-logs")

    def test_single_instance_guard(self):
        self.mgr.instances = [{"model_id": "a", "port": 8081, "state": "ready",
                               "started_at": 0}]
        ok, err = self.mgr.start("b", model_ref="b", flag_str="")
        self.assertFalse(ok)
        self.assertIn("already", err.lower())

    def test_start_spawns_via_wsl_and_records_instance(self):
        with mock.patch("wsl.popen") as popen, \
             mock.patch("os.makedirs"), \
             mock.patch("builtins.open", mock.mock_open()):
            popen.return_value = mock.Mock()
            ok, err = self.mgr.start("Qwen/Qwen3-8B", model_ref="Qwen/Qwen3-8B",
                                     flag_str="--tensor-parallel-size 2")
        self.assertTrue(ok, err)
        self.assertEqual(len(self.mgr.instances), 1)
        self.assertEqual(self.mgr.instances[0]["model_id"], "Qwen/Qwen3-8B")
        self.assertEqual(self.mgr.instances[0]["state"], "starting")
        self.assertIn("--tensor-parallel-size 2", popen.call_args[0][0])

    def test_stop_pkills_and_clears_instance(self):
        self.mgr.instances = [{"model_id": "a", "port": 8081, "state": "ready",
                               "started_at": 0}]
        with mock.patch("wsl.run", return_value=(0, "", "")) as run:
            self.mgr.stop("a")
        self.assertIn("pkill", run.call_args[0][0])
        self.assertEqual(self.mgr.instances, [])

    def test_status_reports_instances(self):
        self.mgr.instances = [{"model_id": "a", "port": 8081, "state": "ready",
                               "started_at": 123}]
        st = self.mgr.status()
        self.assertEqual(st[0]["model_id"], "a")
        self.assertEqual(st[0]["endpoint"], "http://127.0.0.1:8081")

    def test_reconcile_marks_offline_when_no_vllm_process(self):
        self.mgr.instances = [{"model_id": "a", "port": 8081, "state": "ready",
                               "started_at": 0}]
        with mock.patch("wsl.run", return_value=(1, "", "")):   # pgrep: nothing
            self.mgr.reconcile()
        self.assertEqual(self.mgr.instances, [])


if __name__ == "__main__":
    unittest.main()
