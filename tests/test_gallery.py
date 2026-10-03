import conftest_paths  # noqa: F401
import json, os, shutil, tempfile, unittest

import gallery

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def recipe(**kw):
    r = {"llamaforge_recipe": 1, "name": "gemma-moe",
         "model": {"id": "gemma", "file": "gemma-4-26B-A4B-it-MXFP4_MOE.gguf",
                   "hf_repo": "unsloth/gemma-4-26B-A4B-it-GGUF"},
         "settings": {"ctx-size": "90000", "n-cpu-moe": "10"}, "engine": None,
         "about": {"title": "Gemma 4 26B MoE on one 16 GB card", "hardware": "RTX 5060 Ti 16 GB",
                   "author": "dadwritestech", "notes": "Experts\non CPU."}}
    r.update(kw)
    return json.dumps(r)


class Entries(unittest.TestCase):
    def test_valid_recipe_becomes_an_entry(self):
        [e] = gallery.entries([("gemma-moe.json", recipe())])
        self.assertEqual((e["id"], e["title"], e["hardware"]),
                         ("gemma-moe", "Gemma 4 26B MoE on one 16 GB card", "RTX 5060 Ti 16 GB"))
        self.assertEqual(e["notes"], "Experts on CPU.")           # newline flattened
        self.assertEqual(e["recipe"]["settings"], {"ctx-size": "90000", "n-cpu-moe": "10"})
        self.assertNotIn("about", e["recipe"])
        self.assertFalse(e["have"])

    def test_invalid_files_are_skipped(self):
        files = [("a.json", "{nope"), ("b.json", json.dumps({"x": 1})),
                 ("c.json", recipe(settings={"temp": "1\n[x]"})), ("d.json", recipe())]
        self.assertEqual([e["id"] for e in gallery.entries(files)], ["d"])

    def test_unsafe_knobs_are_dropped_like_a_paste(self):
        [e] = gallery.entries([("x.json", recipe(settings={"temp": "1", "rpc": "1.2.3.4:1"}))])
        self.assertEqual((e["settings"], e["dropped"]), ({"temp": "1"}, ["rpc"]))

    def test_about_is_length_capped(self):
        [e] = gallery.entries([("x.json", recipe(about={"title": "t" * 500, "notes": "n" * 5000}))])
        self.assertEqual((len(e["title"]), len(e["notes"])), (80, 400))

    def test_marks_models_already_installed(self):
        sections = {"g": {"model": "D:/m/gemma-4-26B-A4B-it-MXFP4_MOE.gguf"}}
        [e] = gallery.entries([("x.json", recipe())], sections)
        self.assertTrue(e["have"])


class Remote(unittest.TestCase):
    def test_listing_keeps_small_json_files_only(self):
        listing = [{"name": "a.json", "type": "file", "size": 900},
                   {"name": "README.md", "type": "file", "size": 10},
                   {"name": "big.json", "type": "file", "size": 10 ** 7},
                   {"name": "sub", "type": "dir"}]
        self.assertEqual(gallery.remote_names(listing), ["a.json"])
        with self.assertRaises(ValueError):
            gallery.remote_names({"message": "API rate limit exceeded"})

    def test_fetches_raw_files_from_the_repo_only(self):
        seen = []

        def get(url):
            seen.append(url)
            return json.dumps([{"name": "a.json", "type": "file", "size": 9,
                                "download_url": "https://evil.example/a.json"}]) if url == gallery.LISTING else recipe()
        out = gallery.remote(get)
        self.assertEqual(seen, [gallery.LISTING, gallery.RAW + "a.json"])
        self.assertEqual(out[0][0], "a.json")

    def test_falls_back_to_bundled_when_offline(self):
        d = tempfile.mkdtemp(); self.addCleanup(shutil.rmtree, d, True)
        with open(os.path.join(d, "x.json"), "w", encoding="utf-8") as f:
            f.write(recipe())

        def down(url):
            raise OSError("offline")
        got, source = gallery.files(d, force=True, get=down)
        self.assertEqual((source, [n for n, _ in got]), ("bundled", ["x.json"]))


class RepoRecipes(unittest.TestCase):
    """Every recipe in the repo must be clean: valid, described, nothing dropped."""

    def test_repo_recipes_are_clean(self):
        folder = os.path.join(ROOT, "recipes")
        files = gallery.bundled(folder)
        self.assertTrue(files, "recipes/ should ship at least one recipe")
        names = [n for n, _ in files]
        self.assertEqual(len(gallery.entries(files)), len(files), "a recipe in recipes/ failed to parse")
        for n, text in files:
            src = json.loads(text)
            about = src.get("about") or {}
            for k in ("title", "hardware", "author"):
                self.assertTrue(about.get(k), f"{n}: about.{k} is required")
            self.assertTrue(src["model"].get("hf_repo"), f"{n}: model.hf_repo is required")
        for e in gallery.entries(files):
            self.assertEqual(e["dropped"], [], f"{e['id']}: remove knobs that recipes can't carry")
        self.assertEqual(len(set(names)), len(names))


if __name__ == "__main__":
    unittest.main()
