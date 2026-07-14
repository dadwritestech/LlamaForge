import conftest_paths  # noqa: F401
import unittest
import server


class TestMergeVllm(unittest.TestCase):
    def test_llamacpp_rows_get_backend_and_endpoint(self):
        base = {"models": [{"id": "gguf-a", "status": "loaded"}], "global": {}}
        out = server.merge_vllm_models(base, vllm_status=[], vllm_ids=[],
                                       router_port=8080)
        self.assertEqual(out["models"][0]["backend"], "llamacpp")
        self.assertEqual(out["models"][0]["endpoint"], "http://127.0.0.1:8080")

    def test_running_vllm_instance_appended_as_row(self):
        base = {"models": [], "global": {}}
        status = [{"model_id": "Qwen/Qwen3-8B", "state": "ready", "port": 8081,
                   "endpoint": "http://127.0.0.1:8081"}]
        out = server.merge_vllm_models(base, vllm_status=status,
                                       vllm_ids=["Qwen/Qwen3-8B"], router_port=8080)
        row = next(m for m in out["models"] if m["id"] == "Qwen/Qwen3-8B")
        self.assertEqual(row["backend"], "vllm")
        self.assertEqual(row["status"], "loaded")     # 'ready' -> 'loaded' for UI parity
        self.assertEqual(row["endpoint"], "http://127.0.0.1:8081")

    def test_registered_but_stopped_vllm_model_shows_offline(self):
        base = {"models": [], "global": {}}
        out = server.merge_vllm_models(base, vllm_status=[],
                                       vllm_ids=["Qwen/Qwen3-8B"], router_port=8080)
        row = next(m for m in out["models"] if m["id"] == "Qwen/Qwen3-8B")
        self.assertEqual(row["backend"], "vllm")
        self.assertEqual(row["status"], "offline")


if __name__ == "__main__":
    unittest.main()
