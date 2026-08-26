"""Bind a preset as non-destructive model defaults (issue #2)."""
import conftest_paths  # noqa: F401
import json, os, tempfile, unittest
from unittest import mock

import config, routes


class _ConfigTempCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._saved = config.CONFIG
        config.CONFIG = os.path.join(self.tmp, "config.json")

    def tearDown(self):
        config.CONFIG = self._saved


class BindingStorageTest(_ConfigTempCase):
    def setUp(self):
        super().setUp()
        config.save_preset("coding", {"temp": "0.2"})

    def test_bind_records_the_pair(self):
        config.bind_preset("qwopus", "coding")
        self.assertEqual(config.get_bindings(), {"qwopus": "coding"})

    def test_bind_unknown_preset_raises(self):
        with self.assertRaises(ValueError):
            config.bind_preset("qwopus", "no-such-preset")

    def test_unbind_removes_only_that_model(self):
        config.bind_preset("qwopus", "coding")
        config.bind_preset("ornith", "coding")
        self.assertTrue(config.unbind_preset("qwopus"))
        self.assertEqual(config.get_bindings(), {"ornith": "coding"})

    def test_unbind_absent_is_false(self):
        self.assertFalse(config.unbind_preset("ghost"))

    def test_bindings_for_preset(self):
        config.bind_preset("qwopus", "coding")
        config.bind_preset("ornith", "coding")
        config.save_preset("chat", {"temp": "0.8"})
        config.bind_preset("gemma", "chat")
        self.assertEqual(sorted(config.bindings_for_preset("coding")), ["ornith", "qwopus"])

    def test_deleting_a_preset_drops_its_bindings(self):
        config.bind_preset("qwopus", "coding")
        config.delete_preset("coding")
        self.assertEqual(config.get_bindings(), {})

    def test_prune_binding_on_model_delete(self):
        config.bind_preset("qwopus", "coding")
        self.assertTrue(config.prune_binding("qwopus"))
        self.assertEqual(config.get_bindings(), {})

    def test_same_model_id_has_separate_bindings_per_engine(self):
        config.save_preset("chat", {"temp": "0.8"})
        config.bind_preset("shared", "coding", engine="llamacpp")
        config.bind_preset("shared", "chat", engine="ikllama")

        self.assertEqual(config.get_bindings("llamacpp"), {"shared": "coding"})
        self.assertEqual(config.get_bindings("ikllama"), {"shared": "chat"})

    def test_legacy_flat_bindings_migrate_into_the_active_engine(self):
        cfg = config.load()
        cfg["active_engine"] = "ikllama"
        cfg["preset_bindings"] = {"legacy": "coding"}
        config.save(cfg)

        config.migrate()

        self.assertEqual(config.get_bindings("llamacpp"), {})
        self.assertEqual(config.get_bindings("ikllama"), {"legacy": "coding"})


class BindMaterializeRouteTest(_ConfigTempCase):
    """Route-level reconciliation owns only values that it materialized."""

    def setUp(self):
        super().setUp()
        self.ini = os.path.join(self.tmp, "models.ini")
        cfg = config.load(); cfg["models_ini"] = self.ini; config.save(cfg)
        config.set_keys("qwopus", {"model": "/m/q.gguf"})
        config.save_preset("coding", {"temp": "0.2", "top-k": "20"})
        self.router = mock.patch.object(routes, "router", return_value=(200, {"data": []}))
        self.router.start()

    def tearDown(self):
        self.router.stop()
        super().tearDown()

    def _post(self, fn, **body):
        req = mock.Mock(); req.body = body
        return fn(req)

    def test_bind_materializes_the_preset(self):
        self._post(routes.post_presets_bind, model="qwopus", name="coding")
        self.assertEqual(config.get_bindings(), {"qwopus": "coding"})
        self.assertEqual(config.read_sections()["qwopus"]["temp"], "0.2")

    def test_bind_does_not_overwrite_an_existing_model_value(self):
        config.set_keys("qwopus", {"temp": "0.7"})

        self._post(routes.post_presets_bind, model="qwopus", name="coding")

        section = config.read_sections()["qwopus"]
        self.assertEqual(section["temp"], "0.7")
        self.assertEqual(section["top-k"], "20")

    def test_bind_rejects_an_empty_model_before_writing_the_registry(self):
        with open(self.ini, encoding="utf-8") as f:
            before = f.read()

        with self.assertRaises(routes.ApiError):
            self._post(routes.post_presets_bind, model="", name="coding")

        with open(self.ini, encoding="utf-8") as f:
            self.assertEqual(f.read(), before)

    def test_bind_normalizes_model_and_preset_names_before_reconciling(self):
        self._post(routes.post_presets_bind, model="  qwopus  ", name="  coding  ")

        self.assertEqual(config.get_bindings(), {"qwopus": "coding"})
        self.assertEqual(config.read_sections()["qwopus"]["temp"], "0.2")

    def test_editing_a_bound_preset_resyncs_every_model(self):
        self._post(routes.post_presets_bind, model="qwopus", name="coding")
        config.set_keys("ornith", {"model": "/m/o.gguf"})
        self._post(routes.post_presets_bind, model="ornith", name="coding")
        self._post(routes.post_presets_save, name="coding", settings={"temp": "0.9"})

        sections = config.read_sections()
        self.assertEqual(sections["qwopus"]["temp"], "0.9")
        self.assertEqual(sections["ornith"]["temp"], "0.9")

    def test_preset_edit_normalizes_name_before_resyncing_bindings(self):
        self._post(routes.post_presets_bind, model="qwopus", name="coding")

        self._post(routes.post_presets_save, name="  coding  ",
                   settings={"temp": "0.9"})

        self.assertEqual(config.read_sections()["qwopus"]["temp"], "0.9")

    def test_preset_edit_preserves_a_manually_changed_bound_key(self):
        self._post(routes.post_presets_bind, model="qwopus", name="coding")
        config.set_keys("qwopus", {"temp": "0.55"})

        self._post(routes.post_presets_save, name="coding",
                   settings={"temp": "0.9", "top-k": "40"})

        section = config.read_sections()["qwopus"]
        self.assertEqual(section["temp"], "0.55")
        self.assertEqual(section["top-k"], "40")

    def test_removed_preset_key_is_deleted_only_while_still_owned(self):
        self._post(routes.post_presets_bind, model="qwopus", name="coding")
        config.set_keys("qwopus", {"temp": "0.55"})

        self._post(routes.post_presets_save, name="coding", settings={"min-p": "0.1"})

        section = config.read_sections()["qwopus"]
        self.assertEqual(section["temp"], "0.55")
        self.assertNotIn("top-k", section)
        self.assertEqual(section["min-p"], "0.1")

    def test_unbind_removes_untouched_owned_values_but_keeps_manual_values(self):
        self._post(routes.post_presets_bind, model="qwopus", name="coding")
        config.set_keys("qwopus", {"temp": "0.55"})
        self._post(routes.post_presets_bind, model="qwopus", name="")   # unbind

        self.assertEqual(config.get_bindings(), {})
        section = config.read_sections()["qwopus"]
        self.assertEqual(section["temp"], "0.55")
        self.assertNotIn("top-k", section)

    def test_saving_an_unbound_preset_materializes_nothing(self):
        self._post(routes.post_presets_save, name="coding", settings={"temp": "0.5"})
        self.assertNotIn("temp", config.read_sections()["qwopus"])

    def test_deleting_preset_removes_only_untouched_owned_values(self):
        self._post(routes.post_presets_bind, model="qwopus", name="coding")
        config.set_keys("qwopus", {"temp": "0.55"})

        self._post(routes.post_presets_delete, name="coding")

        self.assertEqual(config.get_bindings(), {})
        section = config.read_sections()["qwopus"]
        self.assertEqual(section["temp"], "0.55")
        self.assertNotIn("top-k", section)


if __name__ == "__main__":
    unittest.main()
