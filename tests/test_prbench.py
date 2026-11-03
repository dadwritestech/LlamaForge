import conftest_paths  # noqa: F401
import json
import os
import shutil
import tempfile
import unittest

import prbench


def row(n_prompt, n_gen, ts, sd, commit="ba40715"):
    return {"build_commit": commit, "n_prompt": n_prompt, "n_gen": n_gen, "n_depth": 0,
            "avg_ts": ts, "stddev_ts": sd, "model_type": "qwen3 8B Q4_K - Medium"}


BENCH_OUT = "ggml_cuda_init: found 2 CUDA devices\n" + json.dumps(
    [row(512, 0, 5012.3456, 40.1), row(0, 128, 112.349, 0.81)], indent=2)


class Argv(unittest.TestCase):
    def test_bench_bin_sits_next_to_the_server(self):
        files = {os.path.join("B", "llama-bench.exe"), os.path.join("C", "llama-bench")}
        isfile = files.__contains__
        self.assertEqual(prbench.bench_bin(os.path.join("B", "llama-server.exe"), isfile),
                         os.path.join("B", "llama-bench.exe"))
        self.assertEqual(prbench.bench_bin(os.path.join("C", "llama-server"), isfile),
                         os.path.join("C", "llama-bench"))
        self.assertEqual(prbench.bench_bin(os.path.join("D", "llama-server.exe"), isfile), "")
        self.assertEqual(prbench.bench_bin("", isfile), "")

    def test_the_models_gpu_settings_carry_over(self):
        s = {"model": "M.gguf", "n-gpu-layers": "99", "tensor-split": "3,1", "flash-attn": "on",
             "cache-type-k": "q8_0", "cache-type-v": "q8_0", "split-mode": "layer",
             "device": "CUDA0,CUDA1", "n-cpu-moe": "4", "main-gpu": "0",
             "ctx-size": "32768", "temp": "0.7", "jinja": "true"}       # not bench knobs
        a = prbench.argv("bench", s)
        self.assertEqual(a[:11], ["bench", "-m", "M.gguf", "-p", "512", "-n", "128",
                                  "-r", "3", "-o", "json"])
        rest = a[11:]
        pairs = dict(zip(rest[::2], rest[1::2]))
        self.assertEqual(pairs, {"-ngl": "99", "-ts": "3/1", "-fa": "on", "-ctk": "q8_0",
                                 "-ctv": "q8_0", "-sm": "layer", "-dev": "CUDA0/CUDA1",
                                 "-ncmoe": "4", "-mg": "0"})

    def test_values_bench_wont_take_are_left_out(self):
        a = prbench.argv("bench", {"model": "M", "n-gpu-layers": "auto", "gpu-layers": "all",
                                   "flash-attn": "true", "n-cpu-moe": "", "split-mode": "x;y"})
        self.assertEqual(a[11:], ["-fa", "on"])
        self.assertEqual(prbench.argv("bench", {"model": "M", "flash-attn": "0"})[11:], ["-fa", "off"])
        # gpu-layers is the newer name for n-gpu-layers
        self.assertEqual(prbench.argv("bench", {"model": "M", "gpu-layers": "40"})[11:], ["-ngl", "40"])

    def test_no_model_file_is_refused(self):
        with self.assertRaises(ValueError):
            prbench.argv("bench", {})


class Parse(unittest.TestCase):
    def test_pp_and_tg_rows(self):
        r = prbench.parse(BENCH_OUT)
        self.assertEqual(r, {"pp": {"ts": 5012.35, "sd": 40.1}, "tg": {"ts": 112.35, "sd": 0.81},
                             "commit": "ba40715", "model_type": "qwen3 8B Q4_K - Medium"})

    def test_garbage_is_a_value_error(self):
        for out in ["", "error: failed to load model", "[]", json.dumps([row(512, 0, 1, 0)])]:
            with self.assertRaises(ValueError, msg=out):
                prbench.parse(out)


class Compare(unittest.TestCase):
    def test_percent_and_noise(self):
        base = {"pp": {"ts": 5000.0, "sd": 40.0}, "tg": {"ts": 112.0, "sd": 0.8}}
        pr = {"pp": {"ts": 5030.0, "sd": 35.0}, "tg": {"ts": 126.0, "sd": 0.9}}
        d = prbench.compare(base, pr)
        self.assertEqual(d["tg"], {"pct": 12.5, "noise": False})
        self.assertEqual(d["pp"], {"pct": 0.6, "noise": True})    # 30 < 40 + 35


class Runner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.ran, self.leased = [], []
        self.outs = {"base": (0, BENCH_OUT.replace("112.349", "100.0"), ""), "pr": (0, BENCH_OUT, "")}

    def bench(self, argv):
        self.ran.append(argv[0])
        return self.outs[argv[0]]

    def lease(self):
        import contextlib

        @contextlib.contextmanager
        def cm():
            self.leased.append("in")
            try:
                yield
            finally:
                self.leased.append("out")
        return cm()

    def runner(self):
        return prbench.ABRunner(os.path.join(self.tmp, "ab.json"), run=self.bench,
                                lease=self.lease, clock=lambda: 7.0)

    def test_base_then_pr_inside_one_lease(self):
        r = self.runner()
        self.assertTrue(r.start({"pr": 31621, "install": "pr-31621-ba40715", "model": "qwen"},
                                ["base"], ["pr"]))
        r.join(5)
        st = r.status()
        self.assertEqual(self.ran, ["base", "pr"])
        self.assertEqual(self.leased, ["in", "out"])
        self.assertEqual(st["phase"], "done", st)
        self.assertFalse(st["running"])
        self.assertEqual(st["base"]["tg"]["ts"], 100.0)
        self.assertEqual(st["pr_result"]["tg"]["ts"], 112.35)
        self.assertEqual(st["delta"]["tg"]["pct"], 12.3)
        self.assertEqual((st["pr"], st["install"], st["model"], st["finished"]),
                         (31621, "pr-31621-ba40715", "qwen", 7.0))
        # the last result survives a restart
        self.assertEqual(self.runner().status()["delta"]["tg"]["pct"], 12.3)

    def test_a_failing_run_says_which_build(self):
        self.outs["pr"] = (1, "", "error: unknown argument: -ncmoe\n")
        r = self.runner()
        r.start({"pr": 1, "install": "pr-1-x", "model": "m"}, ["base"], ["pr"])
        r.join(5)
        st = r.status()
        self.assertEqual(st["phase"], "failed")
        self.assertIn("llama-bench on PR #1 failed", st["error"])
        self.assertIn("unknown argument", st["error"])
        self.assertEqual(self.leased, ["in", "out"])

    def test_one_at_a_time(self):
        import threading
        gate = threading.Event()
        self.bench = lambda argv: (gate.wait(5), self.outs[argv[0]])[1]
        r = self.runner()
        self.assertTrue(r.start({"pr": 1, "install": "i", "model": "m"}, ["base"], ["pr"]))
        self.assertFalse(r.start({"pr": 1, "install": "i", "model": "m"}, ["base"], ["pr"]))
        gate.set()
        r.join(5)


if __name__ == "__main__":
    unittest.main()
