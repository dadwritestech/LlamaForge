import conftest_paths  # noqa: F401
import os, tempfile, unittest
import config


class TestPresets(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.cfg_path = os.path.join(self.dir, "config.json")
        self._orig = config.CONFIG
        config.CONFIG = self.cfg_path

    def tearDown(self):
        config.CONFIG = self._orig

    def test_empty_by_default(self):
        self.assertEqual(config.get_presets(), {})

    def test_save_and_read_back(self):
        config.save_preset("coding", {"temp": "0.2", "top-p": "0.9"})
        presets = config.get_presets()
        self.assertEqual(presets["coding"], {"temp": "0.2", "top-p": "0.9"})

    def test_blank_values_dropped(self):
        config.save_preset("fast", {"temp": "0.7", "top-k": "  ", "n-gpu-layers": ""})
        self.assertEqual(config.get_presets()["fast"], {"temp": "0.7"})

    def test_overwrite_existing(self):
        config.save_preset("x", {"temp": "0.1"})
        config.save_preset("x", {"temp": "0.9"})
        self.assertEqual(config.get_presets()["x"], {"temp": "0.9"})

    def test_blank_name_rejected(self):
        with self.assertRaises(ValueError):
            config.save_preset("  ", {"temp": "0.1"})

    def test_delete(self):
        config.save_preset("gone", {"temp": "0.1"})
        self.assertTrue(config.delete_preset("gone"))
        self.assertNotIn("gone", config.get_presets())
        self.assertFalse(config.delete_preset("gone"))

    def test_survives_non_dict_presets_field(self):
        config.save({**config.load(), "presets": "corrupt"})
        self.assertEqual(config.get_presets(), {})
        config.save_preset("ok", {"temp": "0.5"})
        self.assertEqual(config.get_presets()["ok"], {"temp": "0.5"})


if __name__ == "__main__":
    unittest.main()
