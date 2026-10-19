"""slotkeep: a model's conversations survive a reload."""
import conftest_paths  # noqa: F401
import os
import tempfile
import unittest

import slotkeep


class FakeRouter:
    def __init__(self, slots):
        self.slots = slots
        self.calls = []
        self.save_code = 200

    def __call__(self, path, method="GET", body=None, timeout=30):
        self.calls.append((path, method, body))
        if path.startswith("/slots?model="):
            return 200, self.slots
        if "action=save" in path:
            if self.save_code != 200:
                return self.save_code, {"error": {"message": "This server does not support slots action."}}
            return 200, {"n_saved": 100}
        if "action=restore" in path:
            return 200, {"n_restored": 100}
        return 404, {}


def actions(r):
    return [(p, b) for p, m, b in r.calls if m == "POST"]


class SaveTest(unittest.TestCase):
    def test_reading_slots_never_loads_the_model(self):
        # /slots?model=X loads X unless autoload=false; a gone model must 400, not reload.
        r = FakeRouter([])
        slotkeep.slots(r, "qwen/27b")
        self.assertEqual(r.calls[0][0], "/slots?model=qwen%2F27b&autoload=false")

    def test_only_slots_with_a_conversation_are_saved(self):
        r = FakeRouter([{"id": 0, "n_prompt_tokens": 5000}, {"id": 1, "n_prompt_tokens": 0}])
        files, why = slotkeep.save(r, "qwen/27b", "D:/x", disk_free=lambda p: 10 * 2**30)
        name = slotkeep.filename("qwen/27b", 0)
        self.assertEqual(files, [(0, name)])
        self.assertEqual(why, "")
        self.assertEqual(actions(r), [("/slots/0?action=save", {"model": "qwen/27b", "filename": name})])

    def test_filenames_are_ours_and_safe(self):
        n = slotkeep.filename("org/model:Q4 <x>", 3)
        self.assertRegex(n, r"^lf-[0-9a-f]{12}-3\.bin$")

    def test_too_little_disk_skips_the_save(self):
        r = FakeRouter([{"id": 0, "n_prompt_tokens": 5000}])
        files, why = slotkeep.save(r, "m", "D:/x", need_mib=600, disk_free=lambda p: 1000 * 2**20)
        self.assertEqual(files, [])
        self.assertIn("disk", why)
        self.assertEqual(actions(r), [])

    def test_a_router_without_a_save_path_says_how_to_fix_it(self):
        r = FakeRouter([{"id": 0, "n_prompt_tokens": 5000}]); r.save_code = 501
        files, why = slotkeep.save(r, "m", "D:/x", disk_free=lambda p: 10 * 2**30)
        self.assertEqual(files, [])
        self.assertIn("restart the router", why)                # one restart turns it on
        self.assertIn("Apply & Restart Router", why)


    def test_a_200_that_saved_nothing_says_so(self):
        r = FakeRouter([{"id": 0, "n_prompt_tokens": 5000}])
        r2 = lambda path, method="GET", body=None, timeout=30: (
            (200, {}) if "action=save" in path else r(path, method, body, timeout))
        files, why = slotkeep.save(r2, "m", "D:/x", disk_free=lambda p: 10 * 2**30)
        self.assertEqual(files, [])
        self.assertEqual(why, "conversation not kept: the router saved nothing")


class RestoreTest(unittest.TestCase):
    def test_only_empty_idle_slots_are_restored(self):
        r = FakeRouter([{"id": 0, "n_prompt_tokens": 0, "is_processing": False},
                        {"id": 1, "n_prompt_tokens": 40, "is_processing": True}])
        n = slotkeep.restore(r, "m", [(0, "a.bin"), (1, "b.bin")])
        self.assertEqual(n, 1)
        self.assertEqual(actions(r), [("/slots/0?action=restore", {"model": "m", "filename": "a.bin"})])


class WaitIdleTest(unittest.TestCase):
    def test_waits_while_a_reply_is_running(self):
        states = [[{"id": 0, "is_processing": True}], [{"id": 0, "is_processing": False}]]
        r = lambda path, method="GET", body=None, timeout=30: (200, states.pop(0) if len(states) > 1 else states[0])
        slept = []
        self.assertTrue(slotkeep.wait_idle(r, "m", timeout=30, sleep=slept.append))
        self.assertEqual(slept, [1])

    def test_gives_up_after_the_timeout(self):
        r = lambda path, method="GET", body=None, timeout=30: (200, [{"id": 0, "is_processing": True}])
        self.assertFalse(slotkeep.wait_idle(r, "m", timeout=3, sleep=lambda s: None))


class FilesTest(unittest.TestCase):
    def test_drop_and_sweep_delete_only_ours(self):
        d = tempfile.mkdtemp()
        for f in ("lf-abc-0.bin", "lf-abc-1.bin", "keep.txt"):
            open(os.path.join(d, f), "w").close()
        slotkeep.drop(d, [(0, "lf-abc-0.bin")])
        self.assertEqual(sorted(os.listdir(d)), ["keep.txt", "lf-abc-1.bin"])
        self.assertEqual(slotkeep.sweep(d), 1)
        self.assertEqual(os.listdir(d), ["keep.txt"])
        self.assertEqual(slotkeep.sweep(os.path.join(d, "missing")), 0)


if __name__ == "__main__":
    unittest.main()
