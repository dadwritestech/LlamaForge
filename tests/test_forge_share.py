import conftest_paths  # noqa: F401
import unittest

from forge import report

WIN = {"id": "c7", "label": "RTX 5080 alone + drafter Qwen3.8-27B-DFlash2-Q8_0 (5 ahead) on RTX 5060 Ti, KV 16-bit, 32k",
       "code": 126.04, "prose": 71.2, "summary": 60.0, "prefill": 2400.0, "ctx": 32768, "spec": "dflash",
       "settings": {"model": "C:\\Users\\someone\\models\\Qwen3.8-27B-UD-IQ4_XS.gguf",
                    "mmproj": "C:\\Users\\someone\\models\\mmproj-F16.gguf",
                    "split-mode": "none", "device": "CUDA0", "ctx-size": "32768",
                    "spec-type": "draft-dflash",
                    "spec-draft-model": "C:\\Users\\someone\\models\\Qwen3.8-27B-DFlash2-Q8_0.gguf",
                    "spec-draft-device": "CUDA1", "spec-draft-n-max": "5",
                    "api-key": "secret", "host": "0.0.0.0", "temp": "0.6"}}
CUR = {"id": "current", "code": 26.8, "prose": 26.5, "summary": 20.0, "prefill": 900.0, "ctx": 70000}
STATE = {"req": {"intent": "coding", "ctx": 32768},
         "result": {"winner": WIN, "current": CUR, "vs_current": {"code": 370.3, "prose": 168.7},
                    "unverified_quality": False, "drops_vision": False, "current_faster": False,
                    "ctx_asked": 32768}}
GPUS = [{"index": 0, "name": "NVIDIA GeForce RTX 5080"}, {"index": 1, "name": "NVIDIA GeForce RTX 5060 Ti"}]


class Share(unittest.TestCase):
    def setUp(self):
        self.s = report.share(STATE, GPUS, "qwen3.8-27b")

    def test_nothing_to_share_without_a_winner(self):
        self.assertIsNone(report.share(None, GPUS, "m"))
        self.assertIsNone(report.share({"result": {"winner": None}}, GPUS, "m"))

    def test_numbers_and_speedup(self):
        self.assertEqual(self.s["before"], {"code": 26.8, "prose": 26.5})
        self.assertEqual(self.s["after"], {"code": 126.0, "prose": 71.2})
        self.assertEqual(self.s["x"], {"code": 4.7, "prose": 2.7})

    def test_gpu_names_are_short(self):
        self.assertEqual(self.s["gpus"], ["RTX 5080", "RTX 5060 Ti"])

    def test_no_path_secret_or_host_leaves_the_machine(self):
        blob = repr(self.s)
        for bad in ("someone", "Users", "secret", "0.0.0.0", "C:\\"):
            self.assertNotIn(bad, blob)

    def test_only_the_flags_forge_decides_with_files_as_names(self):
        self.assertEqual(self.s["settings"], {
            "ctx-size": "32768", "device": "CUDA0", "spec-draft-device": "CUDA1",
            "spec-draft-model": "Qwen3.8-27B-DFlash2-Q8_0.gguf", "spec-draft-n-max": "5",
            "spec-type": "draft-dflash", "split-mode": "none"})
        self.assertEqual(self.s["file"], "Qwen3.8-27B-UD-IQ4_XS.gguf")

    def test_markdown_has_the_table_flags_and_link(self):
        md = self.s["markdown"]
        self.assertIn("| code | 26.8 | 126.0 | 4.7x |", md)
        self.assertIn("RTX 5080 + RTX 5060 Ti", md)
        self.assertIn("spec-type = draft-dflash", md)
        self.assertIn("https://github.com/dadwritestech/LlamaForge", md)
        self.assertIn("token-for-token", md)

    def test_lossy_and_no_vision_are_stated(self):
        st = {"req": {}, "result": dict(STATE["result"], unverified_quality=True, drops_vision=True)}
        md = report.share(st, GPUS, "m")["markdown"]
        self.assertIn("lossy KV cache", md)
        self.assertIn("vision off", md)

    def test_without_a_baseline_there_is_no_before(self):
        st = {"req": {}, "result": dict(STATE["result"], current=None)}
        s = report.share(st, GPUS, "m")
        self.assertIsNone(s["before"])
        self.assertIsNone(s["x"])
        self.assertIn("| code | 126.0 |", s["markdown"])

    def test_model_ids_cannot_break_the_markdown(self):
        md = report.share(STATE, GPUS, "a|b`c\nd")["markdown"]
        self.assertNotIn("a|b", md)
        self.assertNotIn("\nd", md.split("\n", 1)[0])


if __name__ == "__main__":
    unittest.main()
