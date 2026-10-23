# tests/test_build_docs.py
import conftest_paths  # noqa: F401
import os, re, sys, tempfile, unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import docs
import build_docs


class TestBuildDocs(unittest.TestCase):
    def setUp(self):
        self.src = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.src, "img"))
        for name, sec in [("intro", "getting-started"), ("models", "guides")]:
            with open(os.path.join(self.src, name + ".md"), "w", encoding="utf-8") as f:
                f.write("---\ntitle: %s\nsection: %s\norder: 1\n---\n# %s\ntext" % (name, sec, name))
        with open(os.path.join(self.src, "_style.md"), "w", encoding="utf-8") as f:
            f.write("---\ntitle: S\nsection: guides\norder: 0\n---\nx")
        open(os.path.join(self.src, "img", "a.png"), "wb").write(b"\x89PNG")
        self._orig = docs.content_dir
        docs.content_dir = lambda: self.src
        self.out = tempfile.mkdtemp()

    def tearDown(self):
        docs.content_dir = self._orig

    def test_build_emits_pages_and_copies_images_skips_underscore(self):
        written = build_docs.build(self.out)
        self.assertTrue(os.path.exists(os.path.join(self.out, "intro.html")))
        self.assertTrue(os.path.exists(os.path.join(self.out, "models.html")))
        self.assertTrue(os.path.exists(os.path.join(self.out, "index.html")))
        self.assertTrue(os.path.exists(os.path.join(self.out, "docs", "img", "a.png")))
        self.assertFalse(os.path.exists(os.path.join(self.out, "_style.html")))
        body = open(os.path.join(self.out, "intro.html"), encoding="utf-8").read()
        self.assertIn("<h1", body)
        self.assertIn("LlamaForge", body)  # template chrome present

    def test_build_publishes_markdown_beside_html_and_llms_txt(self):
        build_docs.build(self.out)
        md = open(os.path.join(self.out, "intro.md"), encoding="utf-8").read()
        self.assertTrue(md.startswith("# intro"))       # frontmatter stripped
        self.assertFalse(os.path.exists(os.path.join(self.out, "_style.md")))
        with open(os.path.join(build_docs._ROOT, "llms.txt"), encoding="utf-8") as f:
            want = f.read()
        with open(os.path.join(self.out, "llms.txt"), encoding="utf-8") as f:
            self.assertEqual(f.read(), want)


class TestLlmsTxt(unittest.TestCase):
    """llms.txt lists every docs page, and every link in it resolves."""
    SITE = "https://dadwritestech.github.io/LlamaForge/"
    RAW = "https://raw.githubusercontent.com/dadwritestech/LlamaForge/master/"

    def setUp(self):
        with open(os.path.join(build_docs._ROOT, "llms.txt"), encoding="utf-8") as f:
            self.text = f.read()
        self.links = re.findall(r"\]\((https?://[^)\s]+)\)", self.text)

    def test_starts_with_the_required_h1_and_summary(self):
        lines = self.text.splitlines()
        self.assertEqual(lines[0], "# LlamaForge")
        self.assertTrue(any(l.startswith("> ") for l in lines[1:4]))

    def test_every_docs_page_is_listed_once(self):
        listed = [u[len(self.SITE):-3] for u in self.links
                  if u.startswith(self.SITE) and u.endswith(".md")]
        slugs = [p["slug"] for p in docs.list_pages()]
        self.assertEqual(sorted(listed), sorted(slugs))

    def test_every_link_resolves(self):
        for u in self.links:
            with self.subTest(url=u):
                if u.startswith(self.SITE):
                    self.assertTrue(u.endswith(".md"))
                    path = os.path.join(docs.content_dir(), u[len(self.SITE):])
                elif u.startswith(self.RAW):
                    path = os.path.join(build_docs._ROOT, *u[len(self.RAW):].split("/"))
                else:
                    self.fail("unexpected link host")
                self.assertTrue(os.path.isfile(path), path)


if __name__ == "__main__":
    unittest.main()
