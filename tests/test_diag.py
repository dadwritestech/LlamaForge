import conftest_paths  # noqa: F401
import unittest
import diag

# What a healthy CUDA load prints. Every line here used to trigger a rule:
# "ggml_cuda_init" read as a CUDA failure, "n_ctx" as a context overflow.
HEALTHY = """\
[63752] ggml_cuda_init: found 2 CUDA devices:
[63752] llama_context: n_ctx         = 65536
[63752] llama_kv_cache: size = 2048.00 MiB
[63752] main: server is listening on http://127.0.0.1:63752"""


def attempt(model, port, child, status=None, err_extra=""):
    """A router log as LlamaForge reads it: child output ([port]-prefixed,
    router.out.log) first, then the router's own lines (router.err.log)."""
    out = "\n".join(f"[{port:5d}] {ln}" for ln in child.splitlines())
    err = (f"I srv load: spawning server instance with name={model} on port {port}\n"
           f"I srv load: spawning server instance with args:\n"
           f"I srv load:   --model\n" + err_extra)
    if status is not None:
        err += f"I srv operator(): instance name={model} exited with status {status}\n"
    return out + "\n--- stderr ---\n" + err


class TestRules(unittest.TestCase):
    def test_healthy_cuda_load_is_not_a_failure(self):
        self.assertIsNone(diag.diagnose(HEALTHY))

    def test_unknown_architecture_is_not_blamed_on_cuda(self):
        log = ("ggml_cuda_init: found 1 CUDA devices:\n"
               "llama_model_load: error loading model: error loading model architecture: "
               "unknown model architecture: 'qwen9'")
        r = diag.diagnose(log)
        self.assertIn("architecture", r["error"])
        self.assertIn("Update llama.cpp", r["suggestion"])
        self.assertNotIn("CUDA", r["suggestion"])

    def test_rejected_argument_names_the_flag(self):
        log = ('ggml_cuda_init: found 1 CUDA devices:\n'
               'error while handling argument "--split-mode": invalid value')
        r = diag.diagnose(log)
        self.assertIn("--split-mode", r["suggestion"])
        self.assertNotIn("CUDA", r["suggestion"])

    def test_unknown_argument(self):
        r = diag.diagnose("error: invalid argument: --frobnicate")
        self.assertIn("--frobnicate", r["suggestion"])
        self.assertIn("frobnicate", r["error"])

    def test_mtp_error_is_not_told_to_reduce_ctx(self):
        log = ("llama_context: n_ctx = 150000\n"
               "common_init: context type MTP requested but model doesn't contain MTP layers")
        r = diag.diagnose(log)
        self.assertIn("spec-type", r["suggestion"])
        self.assertNotIn("ctx-size", r["suggestion"])

    def test_quantized_v_cache_needs_flash_attention(self):
        r = diag.diagnose("llama_init_from_model: quantized V cache requires flash_attn to be enabled")
        self.assertIn("flash-attn", r["suggestion"])

    def test_mmproj(self):
        r = diag.diagnose("srv load_model: failed to load multimodal model, 'x/mmproj.gguf'")
        self.assertIn("mmproj", r["suggestion"])

    def test_missing_file(self):
        r = diag.diagnose("gguf_init_from_file: failed to open GGUF file '/models/x.gguf' (No such file)")
        self.assertIn("path", r["suggestion"].lower())

    def test_oom_with_pins_says_they_switch_off_fit(self):
        log = "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 9000.00 MiB on device 0: cudaMalloc failed: out of memory"
        r = diag.diagnose(log, {"n-gpu-layers": "99", "ctx-size": "150000"})
        self.assertIn("n-gpu-layers = 99", r["suggestion"])
        self.assertIn("ctx-size = 150000", r["suggestion"])
        self.assertIn("fit", r["suggestion"])
        self.assertIn("cudaMalloc failed", r["error"])

    def test_oom_without_pins_suggests_a_smaller_quant(self):
        r = diag.diagnose("llama_kv_cache: failed to allocate buffer for kv cache")
        self.assertIn("smaller quant", r["suggestion"])
        self.assertNotIn("n-gpu-layers", r["suggestion"])

    def test_fit_warning_alone_is_not_a_failure(self):
        """common_fit_params prints W ... failed to fit ... and then loads fine."""
        self.assertIsNone(diag.diagnose(
            "W common_fit_params: failed to fit params to free device memory: not implemented, abort"))

    def test_no_failure_returns_none(self):
        log = "srv  update_slots: all slots are idle\nmain: server is listening on 127.0.0.1:8080"
        self.assertIsNone(diag.diagnose(log))

    def test_empty_log_returns_none(self):
        self.assertIsNone(diag.diagnose(""))
        self.assertIsNone(diag.diagnose(None))


class TestLastAttempt(unittest.TestCase):
    def test_only_the_last_attempt_of_this_model_counts(self):
        """An OOM from an earlier load must not be blamed on today's failure."""
        old = attempt("m", 50001, "cudaMalloc failed: out of memory", status=1)
        new = attempt("m", 50002, "error while handling argument \"--split-mode\": invalid value", status=1)
        r = diag.diagnose(old + "\n" + new, model="m")
        self.assertIn("--split-mode", r["suggestion"])

    def test_another_models_failure_is_ignored(self):
        log = attempt("other", 50001, "cudaMalloc failed: out of memory", status=1) + "\n" + \
              attempt("m", 50002, HEALTHY.replace("[63752] ", ""))
        self.assertIsNone(diag.diagnose(log, model="m"))

    def test_no_attempt_in_the_log_means_no_diagnosis(self):
        self.assertIsNone(diag.diagnose(attempt("other", 50001, "cudaMalloc failed", status=1), model="m"))

    def test_bare_nonzero_exit_is_reported(self):
        r = diag.diagnose(attempt("m", 50002, "llama_model_loader: loaded meta data", status=3), model="m")
        self.assertIn("status 3", r["error"])

    def test_error_line_has_no_port_prefix(self):
        r = diag.diagnose(attempt("m", 50002, "unknown model architecture: 'qwen9'", status=1), model="m")
        self.assertTrue(r["error"].startswith("unknown model architecture"), r["error"])


if __name__ == "__main__":
    unittest.main()
