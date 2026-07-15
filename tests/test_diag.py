import conftest_paths  # noqa: F401
import unittest
import diag


class TestDiagnose(unittest.TestCase):
    def test_oom_suggests_reducing_ngl_and_ctx_with_current_values(self):
        log = "llama_model_load: loading model\nggml_backend_cuda_buffer_type_alloc_buffer: failed to allocate\nCUDA error: out of memory"
        r = diag.diagnose(log, {"n-gpu-layers": "99", "ctx-size": "150000"})
        self.assertIsNotNone(r)
        self.assertIn("n-gpu-layers", r["suggestion"])
        self.assertIn("99", r["suggestion"])
        self.assertIn("150000", r["suggestion"])

    def test_unknown_argument(self):
        log = "error: unknown argument: --frobnicate"
        r = diag.diagnose(log)
        self.assertIn("knob", r["suggestion"].lower())
        self.assertIn("frobnicate", r["error"])

    def test_missing_file(self):
        log = "error loading model: failed to open GGUF file /models/x.gguf: No such file"
        r = diag.diagnose(log)
        self.assertIn("path", r["suggestion"].lower())

    def test_no_failure_returns_none(self):
        log = "srv  update_slots: all slots are idle\nmain: server is listening on 127.0.0.1:8080"
        self.assertIsNone(diag.diagnose(log))

    def test_empty_log_returns_none(self):
        self.assertIsNone(diag.diagnose(""))
        self.assertIsNone(diag.diagnose(None))

    def test_error_line_is_the_failure_not_noise(self):
        log = "\n".join([
            "srv  log_server_r: request",
            "llama_model_loader: loaded meta data",
            "CUDA error: out of memory",
            "srv  update_slots: all slots idle",
        ])
        r = diag.diagnose(log)
        self.assertIn("out of memory", r["error"].lower())


if __name__ == "__main__":
    unittest.main()
