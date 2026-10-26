import conftest_paths  # noqa: F401
import os
import unittest

from forge import probe

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "forge")


def fixture(name):
    with open(os.path.join(FIX, name), encoding="utf-8") as f:
        return f.read()


GPU16 = [{"index": 0, "name": "RTX 5080", "vram_mib": 16303},
         {"index": 1, "name": "RTX 5060 Ti", "vram_mib": 16311}]


class FailureReason(unittest.TestCase):
    def test_clean_load_is_not_a_failure(self):
        # carries "common_fit_params: failed to fit ..." - a warning when -ngl is set by hand
        self.assertIsNone(probe.failure_reason(fixture("loaded.txt")))

    def test_oom_names_device_size_and_what(self):
        r = probe.failure_reason(fixture("oom-kv-cache.txt"))
        self.assertIn("out of memory", r)
        self.assertIn("GPU0", r)
        self.assertIn("7616 MiB", r)
        self.assertIn("kv cache", r)
        self.assertIn("rs cache", probe.failure_reason(fixture("oom-rs-cache.txt")))

    def test_oom_that_hangs_without_exiting_is_still_caught(self):
        r = probe.failure_reason(fixture("oom-tensor-hang.txt"))
        self.assertIn("out of memory", r)
        self.assertIn("2720 MiB", r)

    def test_bad_flag(self):
        log = "0.00.01 E error: invalid argument: --spec-type\n"
        self.assertEqual(probe.failure_reason(log), "this build rejects --spec-type")

    def test_other_exit_uses_last_error_line(self):
        log = ("0.00.01 I srv  load_model: loading\n"
               "0.00.02 E llama_model_load: error loading model: tensor 'x' has wrong shape\n"
               "0.00.03 E srv  server_run_i: exiting due to model loading error\n")
        self.assertIn("wrong shape", probe.failure_reason(log))


class Spill(unittest.TestCase):
    def test_reports_devices_over_headroom(self):
        self.assertEqual(probe.spill([15000, 2000], GPU16), [])
        bad = probe.spill([16100, 2000], GPU16, headroom=0.98)
        self.assertEqual(len(bad), 1)
        self.assertIn("GPU0", bad[0])

    def test_unknown_reading_is_not_a_spill(self):
        self.assertEqual(probe.spill(None, GPU16), [])
        self.assertEqual(probe.spill([0, 0], GPU16), [])


class FakeProc:
    """A sandbox process that turns ready (or writes a log line) after n polls."""

    def __init__(self, ready_after=3, log="", fail_after=None, start_ok=True):
        self.polls, self.ready_after, self.fail_after = 0, ready_after, fail_after
        self._log, self.start_ok = log, start_ok
        self.started = self.stopped = 0
        self.t = 0.0

    def start(self, settings):
        self.started += 1
        self.settings = settings
        return (True, "") if self.start_ok else (False, "no free port")

    def state(self):
        self.polls += 1
        if self.fail_after is not None and self.polls >= self.fail_after:
            return "failed"
        return "ready" if self.ready_after is not None and self.polls >= self.ready_after else "loading"

    def stop(self):
        self.stopped += 1

    def log_tail(self):
        return self._log if self.polls >= 2 else ""

    def clock(self):
        return self.t

    def sleep(self, s):
        self.t += s


def prober(p, vram=lambda: [12000, 900]):
    return probe.Prober(start=p.start, state=p.state, stop=p.stop, vram=vram,
                        log_tail=p.log_tail, gpus=GPU16, clock=p.clock, sleep=p.sleep)


class ProberLoad(unittest.TestCase):
    def test_ready_returns_vram_and_keeps_process(self):
        p = FakeProc(ready_after=3)
        r = prober(p).load({"ctx-size": "32768"}, timeout=60)
        self.assertTrue(r["ok"])
        self.assertEqual(r["vram_mib"], [12000, 900])
        self.assertGreater(r["load_s"], 0)
        self.assertEqual(p.stopped, 0)           # caller benches it, then stops

    def test_log_failure_while_loading_stops_the_process(self):
        p = FakeProc(ready_after=None, log=fixture("oom-tensor-hang.txt"))
        r = prober(p).load({}, timeout=600)
        self.assertFalse(r["ok"])
        self.assertIn("out of memory", r["reason"])
        self.assertEqual(p.stopped, 1)
        self.assertLess(p.t, 30)                 # didn't wait for the timeout

    def test_the_previous_run_s_error_in_the_shared_log_is_ignored(self):
        old = ("\n=== 2026-10-26 01:20:24 llama-server -m a.gguf\n"
               "E srv    load_model: failed to load draft model, 'd.gguf'\n"
               "E srv  llama_server: exiting due to model loading error\n")
        now = "\n=== 2026-10-26 01:20:25 llama-server -m a.gguf\nI srv    load_model: loading model 'a.gguf'\n"
        p = FakeProc(ready_after=4, log=old + now)
        r = prober(p).load({}, timeout=60)
        self.assertTrue(r["ok"], r["reason"])

    def test_process_exit_reads_reason_from_log(self):
        p = FakeProc(ready_after=None, fail_after=3, log=fixture("oom-kv-cache.txt"))
        r = prober(p).load({}, timeout=60)
        self.assertIn("kv cache", r["reason"])
        self.assertEqual(p.stopped, 1)

    def test_timeout(self):
        p = FakeProc(ready_after=None)
        r = prober(p).load({}, timeout=20)
        self.assertFalse(r["ok"])
        self.assertIn("didn't load within 20 s", r["reason"])
        self.assertEqual(p.stopped, 1)

    def test_start_error(self):
        p = FakeProc(start_ok=False)
        r = prober(p).load({}, timeout=20)
        self.assertEqual(r["reason"], "no free port")

    def test_loaded_but_spilling_is_a_failure(self):
        p = FakeProc(ready_after=2)
        r = prober(p, vram=lambda: [16250, 900]).load({}, timeout=60)
        self.assertFalse(r["ok"])
        self.assertIn("GPU0", r["reason"])
        self.assertEqual(p.stopped, 1)


if __name__ == "__main__":
    unittest.main()
