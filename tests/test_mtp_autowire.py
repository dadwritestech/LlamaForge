"""Auto-wire MTP draft models (issue #3).

scanner attaches an mtp-* sibling as a speculative draft model, enabling
spec-type=draft-mtp only when the sidecar declares NextN layers (the signal
llama.cpp gates on). Wiring is additive: spec-type is also the ngram-* selector,
so a re-scan must never wipe a hand-set speculative mode.
"""
import conftest_paths  # noqa: F401
import json, os, tempfile, unittest
from unittest import mock

import config, gguf, routes, scanner


class HasNextnTest(unittest.TestCase):
    def _write_gguf(self, kvs):
        """Minimal GGUF with the given (key, int_value) pairs. type 4 = uint32."""
        import struct
        path = os.path.join(self.tmp, "m.gguf")
        with open(path, "wb") as f:
            f.write(b"GGUF")
            f.write(struct.pack("<I", 3))          # version
            f.write(struct.pack("<Q", 0))          # tensor count
            f.write(struct.pack("<Q", len(kvs)))   # kv count
            for k, v in kvs:
                kb = k.encode()
                f.write(struct.pack("<Q", len(kb))); f.write(kb)
                f.write(struct.pack("<I", 4))      # value type uint32
                f.write(struct.pack("<I", v))
        return path

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_true_when_nextn_layers_present(self):
        p = self._write_gguf([("gemma4-assistant.nextn_predict_layers", 2)])
        self.assertTrue(gguf.has_nextn(p))

    def test_false_when_zero_layers(self):
        p = self._write_gguf([("arch.nextn_predict_layers", 0)])
        self.assertFalse(gguf.has_nextn(p))

    def test_false_when_absent(self):
        p = self._write_gguf([("arch.block_count", 40)])
        self.assertFalse(gguf.has_nextn(p))

    def test_false_on_unreadable(self):
        self.assertFalse(gguf.has_nextn(os.path.join(self.tmp, "nope.gguf")))


class BuildEntriesMtpTest(unittest.TestCase):
    """Pure over fake paths; has_nextn is patched so no files are read."""

    def _entries(self, paths, nextn=False):
        with mock.patch.object(scanner, "_slug", side_effect=lambda s: s.lower()), \
             mock.patch("gguf.has_nextn", return_value=nextn), \
             mock.patch("gguf.metadata", return_value={}):
            return {e["id"]: e for e in scanner.build_entries(paths)}

    def test_sidecar_attaches_as_draft_model(self):
        paths = ["/m/model.gguf", "/m/mtp-model.gguf"]
        e = self._entries(paths, nextn=True)
        self.assertEqual(len(e), 1)                 # sidecar isn't a standalone model
        (entry,) = e.values()
        self.assertEqual(entry["draft_model"], "/m/mtp-model.gguf")
        self.assertTrue(entry.get("draft_mtp"))

    def test_attach_only_when_no_nextn(self):
        paths = ["/m/model.gguf", "/m/mtp-model.gguf"]
        (entry,) = self._entries(paths, nextn=False).values()
        self.assertEqual(entry["draft_model"], "/m/mtp-model.gguf")
        self.assertNotIn("draft_mtp", entry)        # inert until the user opts in

    def test_no_sidecar_no_draft_keys(self):
        (entry,) = self._entries(["/m/model.gguf"]).values()
        self.assertNotIn("draft_model", entry)
        self.assertNotIn("draft_mtp", entry)

    def test_sidecar_in_other_dir_not_attached(self):
        paths = ["/a/model.gguf", "/b/mtp-model.gguf"]
        (entry,) = self._entries(paths, nextn=True).values()
        self.assertNotIn("draft_model", entry)

    def test_matching_sidecar_attaches_only_to_its_main_in_a_multi_model_dir(self):
        entries = self._entries([
            "/m/alpha-q4.gguf", "/m/beta-q4.gguf", "/m/mtp-beta-q4.gguf",
        ], nextn=True)
        self.assertNotIn("draft_model", entries["alpha-q4.gguf"])
        self.assertEqual(entries["beta-q4.gguf"]["draft_model"],
                         "/m/mtp-beta-q4.gguf")

    def test_generic_sidecar_attaches_to_none_in_a_multi_model_dir(self):
        entries = self._entries([
            "/m/alpha-q4.gguf", "/m/beta-q4.gguf", "/m/mtp-draft.gguf",
        ], nextn=True)
        self.assertNotIn("draft_model", entries["alpha-q4.gguf"])
        self.assertNotIn("draft_model", entries["beta-q4.gguf"])

    def test_generic_sidecar_attaches_when_directory_has_one_main(self):
        (entry,) = self._entries([
            "/m/alpha-q4.gguf", "/m/mtp-draft.gguf",
        ], nextn=True).values()
        self.assertEqual(entry["draft_model"], "/m/mtp-draft.gguf")


class ScanApplyMtpTest(unittest.TestCase):
    """Route-level: additive wiring that never clobbers a hand-set spec-type."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._saved = config.CONFIG
        config.CONFIG = os.path.join(self.tmp, "config.json")
        self.ini = os.path.join(self.tmp, "models.ini")
        with open(config.CONFIG, "w") as f:
            json.dump({"models_ini": self.ini}, f)
        # keep the router + ctx pass out of the way
        self._r, self._c = routes.router, config.apply_ctx_defaults
        routes.router = lambda *a, **k: (200, {})
        config.apply_ctx_defaults = lambda *a, **k: {"changed": []}

    def tearDown(self):
        config.CONFIG = self._saved
        routes.router, config.apply_ctx_defaults = self._r, self._c

    def _apply(self, entries):
        req = mock.Mock()
        req.body = {"entries": entries}
        routes.post_scan_apply(req)

    def test_enables_draft_mtp_on_a_fresh_model(self):
        self._apply([{"id": "qwopus", "model": "/m/q.gguf",
                      "draft_model": "/m/mtp-q.gguf", "draft_mtp": True}])
        sect = config.read_sections()["qwopus"]
        self.assertEqual(sect["spec-draft-model"], "/m/mtp-q.gguf")
        self.assertEqual(sect["spec-type"], "draft-mtp")

    def test_does_not_overwrite_a_hand_set_spec_type(self):
        config.set_keys("ornith", {"model": "/m/o.gguf", "spec-type": "ngram-mod"})
        self._apply([{"id": "ornith", "model": "/m/o.gguf",
                      "draft_model": "/m/mtp-o.gguf", "draft_mtp": True}])
        sect = config.read_sections()["ornith"]
        self.assertEqual(sect["spec-type"], "ngram-mod", "clobbered a hand-set mode")
        # the draft model still attaches (it was absent), harmless while inert
        self.assertEqual(sect["spec-draft-model"], "/m/mtp-o.gguf")

    def test_attach_only_leaves_spec_type_unset(self):
        self._apply([{"id": "m", "model": "/m/m.gguf",
                      "draft_model": "/m/mtp-m.gguf"}])   # no draft_mtp
        sect = config.read_sections()["m"]
        self.assertEqual(sect["spec-draft-model"], "/m/mtp-m.gguf")
        self.assertNotIn("spec-type", sect)

    def test_rescan_replaces_then_removes_unchanged_auto_owned_draft_keys(self):
        self._apply([{"id": "m", "model": "/m/m.gguf",
                      "draft_model": "/m/mtp-old.gguf", "draft_mtp": True}])
        self._apply([{"id": "m", "model": "/m/m.gguf",
                      "draft_model": "/m/mtp-new.gguf", "draft_mtp": True}])
        replaced = config.read_sections()["m"]
        self.assertEqual(replaced["spec-draft-model"], "/m/mtp-new.gguf")
        self.assertEqual(replaced["spec-type"], "draft-mtp")

        self._apply([{"id": "m", "model": "/m/m.gguf"}])
        removed = config.read_sections()["m"]
        self.assertNotIn("spec-draft-model", removed)
        self.assertNotIn("spec-type", removed)

    def test_rescan_never_replaces_or_removes_a_manually_edited_draft_key(self):
        self._apply([{"id": "m", "model": "/m/m.gguf",
                      "draft_model": "/m/mtp-auto.gguf", "draft_mtp": True}])
        config.set_keys("m", {"spec-draft-model": "/m/manual-draft.gguf"})

        self._apply([{"id": "m", "model": "/m/m.gguf",
                      "draft_model": "/m/mtp-new.gguf", "draft_mtp": True}])
        self.assertEqual(config.read_sections()["m"]["spec-draft-model"],
                         "/m/manual-draft.gguf")

        self._apply([{"id": "m", "model": "/m/m.gguf"}])
        self.assertEqual(config.read_sections()["m"]["spec-draft-model"],
                         "/m/manual-draft.gguf")


class HubRegistrationMtpTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._saved = config.CONFIG
        config.CONFIG = os.path.join(self.tmp, "config.json")
        self.ini = os.path.join(self.tmp, "models.ini")
        with open(config.CONFIG, "w") as f:
            json.dump({"models_ini": self.ini}, f)

    def tearDown(self):
        config.CONFIG = self._saved

    def test_post_download_registration_persists_mtp_draft_keys(self):
        folder = os.path.join(self.tmp, "repo")
        os.makedirs(folder)
        main = os.path.join(folder, "qwen-q4.gguf")
        sidecar = os.path.join(folder, "mtp-qwen-q4.gguf")
        open(main, "wb").close()
        open(sidecar, "wb").close()

        req = mock.Mock()
        req.body = {"path": main}
        with mock.patch("gguf.has_nextn", return_value=True), \
             mock.patch.object(config, "apply_ctx_defaults",
                               return_value={"changed": []}), \
             mock.patch.object(routes, "router", return_value=(200, {})):
            status, result = routes.post_hub_add(req)

        self.assertEqual(status, 200)
        self.assertEqual(result["added"], ["qwen-q4"])
        section = config.read_sections()["qwen-q4"]
        self.assertEqual(section["spec-draft-model"], sidecar.replace("\\", "/"))
        self.assertEqual(section["spec-type"], "draft-mtp")


if __name__ == "__main__":
    unittest.main()
