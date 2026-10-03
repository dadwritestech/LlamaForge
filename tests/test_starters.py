"""Zero-model first run: three starter downloads sized to this GPU (review 01 #2),
and a finished download that registers itself (01 #4)."""
import conftest_paths  # noqa: F401
import os, shutil, tempfile, unittest
from unittest import mock

import hub
import starters

GIB = 1024  # MiB


class PickTest(unittest.TestCase):
    def ids(self, vram_mib):
        return [s["repo"] for s in starters.pick(vram_mib)]

    def test_three_largest_that_fit_biggest_first(self):
        picks = starters.pick(16 * GIB)
        self.assertEqual(len(picks), 3)
        sizes = [p["size"] for p in picks]
        self.assertEqual(sizes, sorted(sizes, reverse=True))
        for p in picks:
            self.assertEqual(hub._fit(p["size"], 16 * GIB), "fits")
        self.assertTrue(picks[0]["recommended"])
        self.assertFalse(any(p["recommended"] for p in picks[1:]))

    def test_a_bigger_card_gets_bigger_models(self):
        small = max(p["size"] for p in starters.pick(8 * GIB))
        big = max(p["size"] for p in starters.pick(32 * GIB))
        self.assertGreater(big, small)

    def test_a_general_model_beats_a_same_size_specialist(self):
        """32 GB: the 30B Instruct, not the (2.8 KB bigger) Coder, is the default."""
        first = starters.pick(32 * GIB)[0]
        self.assertNotIn("Coder", first["title"])

    def test_no_gpu_gets_the_smallest_flagged_cpu(self):
        picks = starters.pick(0)
        self.assertEqual(len(picks), 3)
        smallest = sorted(s["size"] for s in starters.pick_all())[:3]
        self.assertEqual(sorted(p["size"] for p in picks), smallest)
        self.assertTrue(all(p["fit"] == "cpu" for p in picks))

    def test_vision_size_counts_the_projector(self):
        for s in starters.pick_all():
            if s.get("mmproj"):
                self.assertEqual(s["size"], s["weights"] + s["mmproj_size"])

    def test_catalog_is_well_formed(self):
        for s in starters.pick_all():
            self.assertRegex(s["repo"], r"^[\w.-]+/[\w.-]+$")
            self.assertTrue(s["path"].endswith(".gguf"))
            self.assertTrue(s["title"] and s["blurb"])


class AutoRegisterTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        open(os.path.join(self.dir, "m.gguf"), "wb").close()

    def run_job(self, on_done):
        dm = hub.DownloadManager()
        dm.on_done = on_done
        dm.state.update(running=True)
        with mock.patch.object(dm, "_fetch"):
            dm._run("org/repo", ["m.gguf"], self.dir)
        return dm.progress()

    def test_done_download_is_registered_before_phase_done(self):
        seen = []
        s = self.run_job(lambda p: seen.append(p) or ["m"])
        self.assertEqual(seen, [os.path.join(self.dir, "m.gguf")])
        self.assertEqual((s["phase"], s["added"], s["register_error"]), ("done", ["m"], ""))

    def test_a_failed_registration_still_finishes_the_download(self):
        def boom(_p):
            raise RuntimeError("models.ini is read-only")
        s = self.run_job(boom)
        self.assertEqual(s["phase"], "done")
        self.assertEqual(s["added"], [])
        self.assertIn("read-only", s["register_error"])

    def test_no_hook_is_fine(self):
        self.assertEqual(self.run_job(None)["phase"], "done")


if __name__ == "__main__":
    unittest.main()
