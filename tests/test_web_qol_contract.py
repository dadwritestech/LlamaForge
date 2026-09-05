import os, unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


class WebQolContractTest(unittest.TestCase):
    def test_scan_ui_exposes_persistent_folder_list(self):
        src = read("web/js/setup.js")
        self.assertIn('id="scan-roots"', src)
        self.assertIn('model_dirs', src)
        self.assertIn('/api/config', src)

    def test_llama_editor_has_registry_only_unregister_action(self):
        src = read("web/js/models.js")
        self.assertIn('data-act="unregister"', src)
        self.assertIn('/api/models/unregister', src)
        self.assertIn('does not delete', src)

    def test_closed_logs_are_not_polled(self):
        src = read("web/js/main.js")
        self.assertIn('router-log-details', src)
        self.assertIn('.open', src)
        self.assertNotIn('models.refreshRouterLog();\nmodels.refreshVllmLog();', src)

    def test_scanline_overlay_avoids_blend_mode(self):
        src = read("web/index.html")
        rule = src.split("body::after", 1)[1].split("}", 1)[0]
        self.assertNotIn("mix-blend-mode", rule)

    def test_bootstrap_scripts_prompt_for_existing_checkout(self):
        for rel in ("bootstrap.ps1", "bootstrap.sh"):
            src = read(rel)
            self.assertIn("LLAMAFORGE_LLAMA_SRC", src, rel)
            self.assertIn("bootstrap_config.py", src, rel)


if __name__ == "__main__":
    unittest.main()
