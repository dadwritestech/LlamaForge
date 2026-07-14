import conftest_paths  # noqa: F401
import os, tempfile, unittest
from unittest import mock
import server, vllm_registry as reg


class TestSaveHelper(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "vllm_models.json")
        reg.upsert("m", {"repo": "m"}, path=self.path)

    def test_save_persists_and_reports_running(self):
        with mock.patch.object(reg, "DEFAULT_PATH", self.path):
            restarted = server.vllm_save("m", {"tensor-parallel-size": "2"},
                                         is_running=False, restart=lambda mid: None)
            self.assertFalse(restarted)
            self.assertEqual(reg.load(self.path)["m"]["settings"],
                             {"tensor-parallel-size": "2"})

    def test_save_restarts_when_running(self):
        calls = []
        with mock.patch.object(reg, "DEFAULT_PATH", self.path):
            restarted = server.vllm_save("m", {"max-model-len": "4096"},
                                         is_running=True,
                                         restart=lambda mid: calls.append(mid))
        self.assertTrue(restarted)
        self.assertEqual(calls, ["m"])


if __name__ == "__main__":
    unittest.main()
