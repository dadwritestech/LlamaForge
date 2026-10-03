import conftest_paths  # noqa: F401
import json, os, tempfile, unittest
import config


class TestMigrate(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.cfg_path = os.path.join(self.dir, "config.json")
        self._orig = config.CONFIG
        config.CONFIG = self.cfg_path

    def tearDown(self):
        config.CONFIG = self._orig

    def _write(self, obj):
        with open(self.cfg_path, "w", encoding="utf-8") as f:
            json.dump(obj, f)

    def test_no_file_is_left_untouched(self):
        # no config.json on disk yet -> nothing to migrate, no file created
        config.migrate()
        self.assertFalse(os.path.exists(self.cfg_path))

    def test_existing_install_gets_advanced_and_onboarded(self):
        self._write({"server_bin": "/x/llama-server"})
        cfg = config.migrate()
        self.assertEqual(cfg["ui_mode"], "advanced")
        self.assertTrue(cfg["onboarded"])

    def test_fresh_install_gets_lite_and_not_onboarded(self):
        self._write({"server_bin": ""})
        cfg = config.migrate()
        self.assertEqual(cfg["ui_mode"], "lite")
        self.assertFalse(cfg["onboarded"])

    def test_idempotent_when_key_present(self):
        self._write({"server_bin": "/x", "ui_mode": "lite", "onboarded": False})
        cfg = config.migrate()
        self.assertEqual(cfg["ui_mode"], "lite")
        self.assertFalse(cfg["onboarded"])

    def test_mints_a_persistent_router_local_key_once(self):
        self._write({"server_bin": "/x", "ui_mode": "lite"})
        key = config.migrate()["router_local_key"]
        with open(self.cfg_path, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["router_local_key"], key)
        self.assertEqual(config.migrate()["router_local_key"], key)


if __name__ == "__main__":
    unittest.main()
