import conftest_paths  # noqa: F401
import threading
import time
import unittest

from forge import inventory, report, runner


class Family(unittest.TestCase):
    def test_strips_quant_and_shard(self):
        self.assertEqual(inventory.family("D:/m/Qwen3.8-27B-UD-IQ4_XS.gguf"), "qwen3.8-27b")
        self.assertEqual(inventory.family("Qwen3.8-27B-Q6_K-00001-of-00002.gguf"), "qwen3.8-27b")
        self.assertEqual(inventory.family("/x/Qwen3.8-27B-DFlash2-Q8_0.gguf"), "qwen3.8-27b-dflash2")


class Drafters(unittest.TestCase):
    PATHS = ["D:/m/a/Qwen3.8-27B-UD-IQ4_XS.gguf", "D:/m/b/Qwen3.8-27B-DFlash2-Q8_0.gguf",
             "D:/m/c/Qwen3.6-35B-A3B-DFlash-Q8_0.gguf", "D:/m/d/Llama-3-8B-Q4_K_M.gguf"]

    def test_finds_same_family_dflash(self):
        ds = inventory.drafters(self.PATHS[0], {}, self.PATHS, size_of=lambda p: 1800 * 2**20)
        self.assertEqual([d["path"] for d in ds], ["D:/m/b/Qwen3.8-27B-DFlash2-Q8_0.gguf"])
        self.assertEqual(ds[0]["kind"], "dflash")
        self.assertEqual(ds[0]["mib"], 1800)

    def test_the_users_own_drafter_comes_first_and_isnt_duplicated(self):
        s = {"spec-draft-model": "D:/m/b/Qwen3.8-27B-DFlash2-Q8_0.gguf", "spec-type": "draft-dflash"}
        ds = inventory.drafters(self.PATHS[0], s, self.PATHS, size_of=lambda p: 0)
        self.assertEqual(len(ds), 1)
        self.assertTrue(ds[0]["configured"])

    def test_configured_mtp(self):
        s = {"spec-draft-model": "D:/m/mtp-q.gguf", "spec-type": "draft-mtp"}
        self.assertEqual(inventory.drafters(self.PATHS[0], s, [], size_of=lambda p: 0)[0]["kind"], "mtp")

    def test_a_drafter_is_never_a_drafter_for_itself(self):
        self.assertEqual(inventory.drafters(self.PATHS[1], {}, self.PATHS, size_of=lambda p: 0), [])


class Apply(unittest.TestCase):
    def test_writes_differences_and_remembers_the_old_values(self):
        current = {"model": "/m/q.gguf", "temp": "0.7", "tensor-split": "1,1", "ctx-size": "8192"}
        winner = {"model": "/m/q.gguf", "temp": "0.7", "split-mode": "none", "device": "CUDA0",
                  "ctx-size": "32768", "fit": "off"}
        updates, undo = report.apply_plan(winner, current)
        self.assertEqual(updates, {"split-mode": "none", "device": "CUDA0", "ctx-size": "32768",
                                   "fit": "off", "tensor-split": None})
        self.assertEqual(undo, {"split-mode": None, "device": None, "ctx-size": "8192",
                                "fit": None, "tensor-split": "1,1"})

    def test_never_unsets_user_keys_forge_doesnt_own(self):
        updates, _ = report.apply_plan({"model": "/m/q.gguf"}, {"model": "/m/q.gguf", "mmproj": "/m/p"})
        self.assertNotIn("mmproj", updates)

    def test_with_a_baseline_only_forges_own_changes_are_written(self):
        # base = what Forge started from: the [*] keys merged in, and a key the
        # user has since renamed (mmap -> load-mode); neither may come back
        base = {"model": "/m/q.gguf", "load-on-startup": "false", "mmap": "false",
                "ctx-size": "70000", "main-gpu": "1", "temp": "1.0"}
        winner = {"model": "/m/q.gguf", "load-on-startup": "false", "mmap": "false",
                  "ctx-size": "32768", "device": "CUDA0", "temp": "1.0"}
        current = {"model": "/m/q.gguf", "load-mode": "none", "ctx-size": "70000",
                   "main-gpu": "1", "temp": "0.6"}
        updates, undo = report.apply_plan(winner, current, base)
        self.assertEqual(updates, {"ctx-size": "32768", "device": "CUDA0", "main-gpu": None})
        self.assertEqual(undo, {"ctx-size": "70000", "device": None, "main-gpu": "1"})


class Summary(unittest.TestCase):
    def state(self):
        return {"stage": "fit", "current": "b", "error": None,
                "cands": [{"id": "current", "label": "your current setup"},
                          {"id": "a", "label": "A"}, {"id": "b", "label": "B"}, {"id": "c", "label": "C"}],
                "rows": {"current": {"id": "current", "code": 50, "prose": 25},
                         "a": {"id": "a", "code": 100, "prose": 50}},
                "failed": {}, "rejected": [{"id": "z", "label": "Z", "reason": "needs 20 GB"}],
                "finalists": [], "rows_b": {}, "sanity": {}, "result": None,
                "req": {"intent": "coding"}}

    def test_progress_and_leaderboard(self):
        s = report.summary(self.state(), running=True)
        self.assertEqual(s["progress"], {"done": 2, "total": 4})
        self.assertEqual(s["loading"], "B")
        self.assertEqual([r["id"] for r in s["leaderboard"]], ["a", "current"])
        self.assertEqual(s["rejected"][0]["reason"], "needs 20 GB")
        self.assertTrue(s["running"])

    def test_no_state(self):
        self.assertEqual(report.summary(None, running=False)["stage"], "idle")


class Runner(unittest.TestCase):
    def test_runs_in_background_and_reports(self):
        gate = threading.Event()

        class FakeJob:
            state = {"stage": "fit"}
            cancel = threading.Event()

            def run(self):
                gate.wait(2)
                self.state = {"stage": "done"}
                return self.state

            def snapshot(self):
                return dict(self.state)

        r = runner.Runner()
        self.assertTrue(r.start(FakeJob()))
        self.assertTrue(r.running())
        self.assertFalse(r.start(FakeJob()))          # one at a time
        gate.set()
        for _ in range(100):
            if not r.running():
                break
            time.sleep(0.02)
        self.assertEqual(r.snapshot()["stage"], "done")

    def test_cancel_sets_the_jobs_event(self):
        class FakeJob:
            cancel = threading.Event()

            def run(self):
                self.cancel.wait(2)
                return {}

            def snapshot(self):
                return {"stage": "cancelled" if self.cancel.is_set() else "fit"}

        r = runner.Runner()
        j = FakeJob()
        r.start(j)
        self.assertTrue(r.cancel())
        r.join(2)
        self.assertEqual(r.snapshot()["stage"], "cancelled")


if __name__ == "__main__":
    unittest.main()
