import conftest_paths  # noqa: F401
import contextlib
import json
import os
import shutil
import tempfile
import threading
import unittest

from forge import job, space


def item(key, flags, typ="str", options=None):
    return {"key": key, "flags": flags, "type": typ, "options": options, "values": options}


ITEMS = [
    item("split-mode", ["-sm", "--split-mode"], "enum", ["none", "layer"]),
    item("device", ["-dev", "--device"]),
    item("tensor-split", ["-ts", "--tensor-split"]),
    item("cache-type-k", ["-ctk", "--cache-type-k"], "enum", ["f16", "q8_0", "q4_0"]),
    item("cache-type-v", ["-ctv", "--cache-type-v"], "enum", ["f16", "q8_0", "q4_0"]),
    item("spec-type", ["--spec-type"], "enum", ["none", "draft-dflash"]),
    item("spec-draft-model", ["--spec-draft-model"], "path"),
    item("spec-draft-device", ["--spec-draft-device"]),
    item("spec-draft-ngl", ["--spec-draft-ngl"]),
    item("spec-draft-n-max", ["--spec-draft-n-max"], "int"),
    item("fit", ["--fit"], "enum", ["on", "off"]),
]
GPU16 = [{"index": 0, "name": "RTX 5080", "vram_mib": 16303},
         {"index": 1, "name": "RTX 5060 Ti", "vram_mib": 16311}]
INV = {"gpus": GPU16, "caps": space.caps_from_items(ITEMS),
       "base": {"model": "/m/q.gguf", "cache-type-k": "q8_0", "cache-type-v": "q8_0",
                "ctx-size": "8192", "tensor-split": "1,1"},
       "drafters": [{"path": "/m/dflash.gguf", "kind": "dflash", "mib": 1800}], "mtp": False}
REQ = {"intent": "coding", "ctx": 32768, "allow_loss": False, "time": "quick"}


def speed(settings):
    """Fake machine: the drafter helps, solo on the 5080 beats splits, 5060 Ti is slow."""
    if settings.get("device") == "CUDA1":
        base = 60.0
    elif settings.get("split-mode") == "none":
        base = 100.0
    else:
        base = 85.0
    if "spec-draft-model" in settings:
        base *= 1.3 if settings.get("spec-draft-device") == "CUDA1" else 1.2
    return base


class Rig:
    """Stands in for the sandbox process, the bench and the GPU lease."""

    def __init__(self, fail=lambda s: None, diverge=False):
        self.loads, self.stops, self.leases, self.restores = [], 0, 0, 0
        self.benched = []                    # (candidate id, round)
        self.loaded = None
        self.fail, self.diverge = fail, diverge
        self.cancel = threading.Event()
        self.on_bench = None

    def load(self, settings, timeout=300):
        self.loads.append(settings)
        why = self.fail(settings)
        if why:
            return {"ok": False, "reason": why, "vram_mib": None, "load_s": 1.0}
        self.loaded = settings
        return {"ok": True, "reason": "", "vram_mib": [12000, 2000], "load_s": 1.0}

    def stop(self):
        self.stops += 1
        self.loaded = None

    def bench(self, counts, round_id, n_predict):
        self.benched.append((space._cid(self.loaded), round_id))
        if self.on_bench:
            self.on_bench(self)
        v = speed(self.loaded)
        return {"code": v, "prose": v / 2, "summary": v / 2, "prefill": 1500.0,
                "detail": {}, "error": None}

    def sample(self, prompts):
        spec = "spec-draft-model" in self.loaded and self.diverge
        return {p: ([9] * 64 if spec else [1] * 64) for p in prompts}

    @contextlib.contextmanager
    def lease(self):
        self.leases += 1
        try:
            yield
        finally:
            self.restores += 1

    def job(self, path, inv=INV, req=REQ):
        return job.Job(path, inv, req, load=self.load, stop=self.stop, bench=self.bench,
                       sample=self.sample, lease=self.lease, cancel=self.cancel,
                       clock=lambda: 1000.0)


class JobRun(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "forge-job.json")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def state(self):
        with open(self.path, encoding="utf-8") as f:
            return json.load(f)

    def test_full_run_picks_the_fastest_and_restores(self):
        rig = Rig()
        st = rig.job(self.path).run()
        self.assertEqual(st["stage"], "done")
        win = st["result"]["winner"]
        self.assertEqual(win["settings"]["device"], "CUDA0")
        self.assertEqual(win["settings"]["spec-draft-device"], "CUDA1")
        self.assertEqual(rig.leases, 1)
        self.assertEqual(rig.restores, 1)
        self.assertEqual(rig.stops, len(rig.loads))           # every loaded process stopped
        self.assertEqual(self.state()["stage"], "done")         # persisted

    def test_current_config_is_measured_and_compared(self):
        st = Rig().job(self.path).run()
        self.assertIn(job.CURRENT, st["rows"])
        self.assertGreater(st["result"]["vs_current"]["code"], 0)   # +% over what they run now

    def test_failures_are_kept_with_reasons_and_never_win(self):
        rig = Rig(fail=lambda s: "out of memory on GPU0" if s.get("spec-draft-device") == "CUDA1" else None)
        st = rig.job(self.path).run()
        self.assertTrue(any("out of memory" in r for r in st["failed"].values()))
        self.assertNotEqual(st["result"]["winner"]["settings"].get("spec-draft-device"), "CUDA1")

    def test_finalists_are_remeasured_in_round_b(self):
        rig = Rig()
        st = rig.job(self.path).run()
        self.assertTrue(st["finalists"])
        self.assertTrue(all(fid in st["rows_b"] for fid in st["finalists"]))

    def test_a_drafter_that_changes_answers_cannot_win(self):
        st = Rig(diverge=True).job(self.path).run()
        win = st["result"]["winner"]
        self.assertNotIn("spec-draft-model", win["settings"])
        self.assertTrue(any(not s["ok"] for s in st["sanity"].values()))

    def test_resume_skips_finished_candidates(self):
        rig = Rig()

        def stop_after_three(r):
            if len(r.loads) == 3:
                r.cancel.set()

        rig.on_bench = stop_after_three
        st = rig.job(self.path).run()
        self.assertEqual(st["stage"], "cancelled")
        self.assertEqual(rig.restores, 1)
        done = set(self.state()["rows"])
        self.assertEqual(len(done), 3)

        rig2 = Rig()
        st2 = rig2.job(self.path).run()
        self.assertEqual(st2["stage"], "done")
        rerun = {cid for cid, rnd in rig2.benched if rnd == "A"}
        self.assertEqual(len([1 for _, rnd in rig2.benched if rnd == "A"]) + 3,
                         len(st2["rows"]))                  # only the unfinished ones ran
        self.assertEqual(len(rerun), len(rerun - done))

    def test_a_crash_records_the_error_and_still_restores(self):
        rig = Rig()

        def boom(r):
            raise RuntimeError("driver reset")

        rig.on_bench = boom
        st = rig.job(self.path).run()
        self.assertEqual(st["stage"], "error")
        self.assertIn("driver reset", st["error"])
        self.assertEqual(rig.restores, 1)
        self.assertEqual(rig.stops, len(rig.loads))

    def test_server_dying_mid_bench_fails_only_that_candidate(self):
        rig = Rig()
        bench = rig.bench

        def flaky(counts, round_id, n_predict):
            if rig.loaded.get("spec-draft-device") == "CUDA0":
                return {"error": "benchmark request failed: connection reset", "detail": {}}
            return bench(counts, round_id, n_predict)

        rig.bench = flaky
        st = rig.job(self.path).run()
        self.assertEqual(st["stage"], "done")
        self.assertTrue(any("connection reset" in r for r in st["failed"].values()))
        self.assertEqual(rig.stops, len(rig.loads))

    def test_one_job_at_a_time(self):
        rig = Rig()
        inner = {}

        def nested(r):
            if "err" not in inner:
                try:
                    Rig().job(os.path.join(self.dir, "other.json")).run()
                except job.Busy as e:
                    inner["err"] = e

        rig.on_bench = nested
        rig.job(self.path).run()
        self.assertIsInstance(inner.get("err"), job.Busy)


if __name__ == "__main__":
    unittest.main()
