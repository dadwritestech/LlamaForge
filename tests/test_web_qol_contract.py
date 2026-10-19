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

    def test_no_blend_mode_overlays(self):
        # A fixed full-viewport layer with mix-blend-mode repaints the whole
        # page on every scroll. The CRT scanline overlay is gone; keep any
        # future overlay (and everything else) off blend modes.
        src = read("web/index.html")
        self.assertNotIn("mix-blend-mode", src)
        if "body::after" in src:
            rule = src.split("body::after", 1)[1].split("}", 1)[0]
            self.assertNotIn("mix-blend-mode", rule)

    def test_bootstrap_scripts_prompt_for_existing_checkout(self):
        for rel in ("bootstrap.ps1", "bootstrap.sh"):
            src = read(rel)
            self.assertIn("LLAMAFORGE_LLAMA_SRC", src, rel)
            self.assertIn("bootstrap_config.py", src, rel)


if __name__ == "__main__":
    unittest.main()


class WebA11yContractTest(unittest.TestCase):
    """GitHub #30: interactive <span> controls must be <button type="button">
    so they are keyboard-focusable and activatable without a mouse."""

    def read(self, rel):
        return read(os.path.join("web", rel))

    def test_only_set_filter_is_a_button(self):
        src = self.read("js/models.js")
        self.assertIn('data-onlyset', src)
        # Must be a <button> element, not a <span>
        self.assertIn('<button type="button" class="chip', src)

    def test_favorite_star_is_a_button_with_aria(self):
        src = self.read("js/models.js")
        self.assertIn('data-fav=', src)
        # Must be a <button> element
        self.assertIn('<button type="button" class="fav', src)
        # Must expose ARIA state for screen readers
        self.assertIn('aria-pressed=', src)
        self.assertIn('aria-label=', src)

    def test_preset_chips_are_buttons(self):
        src = self.read("js/models.js")
        self.assertIn('data-preset-apply=', src)
        # preset chips must be buttons
        self.assertIn('<button type="button" class="pchip', src)

    def test_profile_chips_are_buttons(self):
        src = self.read("js/profiles.js")
        self.assertIn('data-prof-launch=', src)
        self.assertIn('data-prof-gallery', src)
        self.assertIn('data-prof-import', src)
        # profile chips must be buttons
        self.assertIn('<button type="button" class="pchip', src)

    def test_css_resets_button_appearance_for_chips(self):
        """When <span> is replaced by <button>, default browser button
        styling (outline, background, padding) must be suppressed so the
        visual appearance is unchanged."""
        # Check classic (index.html base styles)
        index = self.read("index.html")
        # Should have a button reset targeting .chip
        self.assertIn("button.chip", index)
        # Check stowage skin
        stowage = self.read("css/stowage.css")
        self.assertIn("button.chip", stowage)
        self.assertIn("button.pchip", stowage)
        # Check hearth skin
        hearth = self.read("css/hearth.css")
        self.assertIn("button.chip", hearth)
        self.assertIn("button.pchip", hearth)


class IndexHeadTest(unittest.TestCase):
    """Text inside <head> makes the parser open <body> early, so every later
    <link>/<style> lands in body as a grid item and the layout falls apart."""

    def test_head_holds_no_stray_text(self):
        from html.parser import HTMLParser

        class P(HTMLParser):
            def __init__(self):
                super().__init__(); self.where = []; self.stray = []
            def handle_starttag(self, tag, attrs):
                if tag in ("head", "style", "script", "title"): self.where.append(tag)
            def handle_endtag(self, tag):
                if self.where and self.where[-1] == tag: self.where.pop()
            def handle_data(self, data):
                if self.where == ["head"] and data.strip(): self.stray.append(data.strip()[:40])

        p = P(); p.feed(read("web/index.html"))
        self.assertEqual(p.stray, [])

    def test_no_font_cdn(self):
        # a local panel must render offline and tell no third party it was opened
        src = read("web/index.html")
        for host in ("fonts.googleapis.com", "fonts.gstatic.com"):
            self.assertNotIn(host, src)
        self.assertIn('/web/css/fonts.css', src)
        css = read("web/css/fonts.css")
        for name in __import__("re").findall(r"/web/fonts/([\w.-]+\.woff2)", css):
            self.assertTrue(os.path.isfile(os.path.join(ROOT, "web", "fonts", name)), name)
        self.assertTrue(os.path.isfile(os.path.join(ROOT, "web", "fonts", "OFL.txt")))
