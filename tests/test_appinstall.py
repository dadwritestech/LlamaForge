"""appinstall: lay an extracted release over an install dir without ever
touching the user's data (config, registry, engines, logs, private Python)."""
import conftest_paths  # noqa: F401
import json, os, shutil, tempfile, unittest

import appinstall


def write(root, rel, text="x"):
    p = os.path.join(root, *rel.split("/"))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)


def read(root, rel):
    with open(os.path.join(root, *rel.split("/")), encoding="utf-8") as f:
        return f.read()


class AppInstallTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.src = os.path.join(self.tmp, "src")
        self.dest = os.path.join(self.tmp, "home")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def release(self, files):
        shutil.rmtree(self.src, True)
        for rel, text in files.items():
            write(self.src, rel, text)

    def test_fresh_install_copies_everything_and_writes_manifest(self):
        self.release({"backend/server.py": "v1", "web/index.html": "<html>", "run.sh": "sh"})
        r = appinstall.install(self.src, self.dest, version="v1.0.0")
        self.assertEqual(read(self.dest, "backend/server.py"), "v1")
        self.assertEqual(sorted(r["added"]), ["backend/server.py", "run.sh", "web/index.html"])
        m = json.load(open(os.path.join(self.dest, appinstall.MANIFEST)))
        self.assertEqual(m["version"], "v1.0.0")
        self.assertEqual(sorted(m["files"]), sorted(r["added"]))

    def test_update_replaces_code_and_removes_files_dropped_by_release(self):
        self.release({"backend/server.py": "v1", "backend/old.py": "gone soon"})
        appinstall.install(self.src, self.dest, version="v1")
        self.release({"backend/server.py": "v2", "backend/new.py": "hi"})
        r = appinstall.install(self.src, self.dest, version="v2")
        self.assertEqual(read(self.dest, "backend/server.py"), "v2")
        self.assertFalse(os.path.exists(os.path.join(self.dest, "backend", "old.py")))
        self.assertEqual(r["removed"], ["backend/old.py"])
        self.assertEqual(r["updated"], ["backend/server.py"])
        self.assertEqual(r["added"], ["backend/new.py"])

    def test_user_data_is_never_touched(self):
        self.release({"backend/server.py": "v1"})
        appinstall.install(self.src, self.dest)
        for rel in ("config.json", "models.ini", "stats.json", "engines/llama.cpp/b1/llama-server",
                    "logs/router.out.log", "python/python.exe", "backend/__pycache__/x.pyc",
                    "models/org--repo/m.gguf", "wiki/notes.md",
                    "backend/my_notes.txt"):
            write(self.dest, rel, "mine")
        # a hostile/broken release that ships user-data names must not clobber them
        self.release({"backend/server.py": "v2", "config.json": "{}",
                      "engines/evil": "x", "python/python.exe": "evil"})
        r = appinstall.install(self.src, self.dest)
        for rel in ("config.json", "models.ini", "stats.json", "engines/llama.cpp/b1/llama-server",
                    "logs/router.out.log", "python/python.exe", "backend/my_notes.txt",
                    "models/org--repo/m.gguf", "wiki/notes.md"):
            self.assertEqual(read(self.dest, rel), "mine", rel)
        self.assertFalse(os.path.exists(os.path.join(self.dest, "engines", "evil")))
        self.assertNotIn("config.json", r["added"] + r["updated"])

    def test_manifest_cannot_be_used_to_delete_outside_the_install(self):
        self.release({"backend/server.py": "v1"})
        appinstall.install(self.src, self.dest)
        victim = os.path.join(self.tmp, "victim.txt")
        write(self.tmp, "victim.txt", "keep")
        mpath = os.path.join(self.dest, appinstall.MANIFEST)
        m = json.load(open(mpath))
        m["files"] += ["../victim.txt", os.path.abspath(victim), "config.json"]
        json.dump(m, open(mpath, "w"))
        write(self.dest, "config.json", "mine")
        appinstall.install(self.src, self.dest)
        self.assertTrue(os.path.exists(victim))
        self.assertEqual(read(self.dest, "config.json"), "mine")

    def test_empty_dirs_left_by_removed_files_are_pruned(self):
        self.release({"backend/server.py": "v1", "web/old/a.js": "a"})
        appinstall.install(self.src, self.dest)
        self.release({"backend/server.py": "v2"})
        appinstall.install(self.src, self.dest)
        self.assertFalse(os.path.exists(os.path.join(self.dest, "web")))

    def test_archive_root_unwraps_github_top_folder(self):
        write(self.src, "LlamaForge-0.10.0/backend/server.py")
        self.assertEqual(appinstall.archive_root(self.src),
                         os.path.join(self.src, "LlamaForge-0.10.0"))
        write(self.src, "stray.txt")
        self.assertEqual(appinstall.archive_root(self.src), self.src)

    def test_refuses_a_source_that_is_not_llamaforge(self):
        self.release({"README.md": "something else"})
        with self.assertRaises(ValueError):
            appinstall.install(self.src, self.dest)

    def test_first_install_config_drops_placeholder_paths(self):
        self.release({"backend/server.py": "v1", "config.example.json": json.dumps(
            {"server_bin": "C:/path/to/llama-server.exe", "llama_src": "C:/path",
             "build_dir": "C:/path/build", "router_port": 8080})})
        appinstall.install(self.src, self.dest)
        self.assertTrue(appinstall.ensure_config(self.dest))
        cfg = json.load(open(os.path.join(self.dest, "config.json")))
        self.assertEqual((cfg["server_bin"], cfg["llama_src"], cfg["build_dir"]), ("", "", ""))
        self.assertEqual(cfg["router_port"], 8080)
        write(self.dest, "config.json", "mine")
        self.assertFalse(appinstall.ensure_config(self.dest))
        self.assertEqual(read(self.dest, "config.json"), "mine")

    def test_cli(self):
        self.release({"LlamaForge-x/backend/server.py": "v1",
                      "LlamaForge-x/config.example.json": "{}"})
        rc = appinstall.main(["--from", self.src, "--to", self.dest, "--version", "vX"])
        self.assertEqual(rc, 0)
        self.assertEqual(appinstall.installed_version(self.dest), "vX")
        self.assertIsNone(appinstall.installed_version(self.tmp))
        self.assertTrue(os.path.exists(os.path.join(self.dest, "config.json")))


class UninstallTest(unittest.TestCase):
    """uninstall.ps1/.sh used to `rm -rf` the folder they sat in. Run from a
    git checkout or a hand-made folder, that took everything with it (05 #1)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.src = os.path.join(self.tmp, "src")
        self.dest = os.path.join(self.tmp, "home")
        for rel in ("backend/server.py", "web/index.html", "run.sh"):
            write(self.src, rel)
        appinstall.install(self.src, self.dest, "v1")
        for rel in ("config.json", "models.ini", "models/a.gguf", "engines/b1/llama-server",
                    "logs/router.err.log", "stats.json", "backend/__pycache__/x.pyc",
                    "my-notes.txt"):
            write(self.dest, rel)

    def exists(self, rel):
        return os.path.exists(os.path.join(self.dest, *rel.split("/")))

    def test_refuses_without_a_manifest(self):
        os.remove(os.path.join(self.dest, appinstall.MANIFEST))
        with self.assertRaises(ValueError):
            appinstall.uninstall(self.dest, everything=True)
        self.assertTrue(self.exists("backend/server.py"))
        self.assertTrue(self.exists("models/a.gguf"))

    def test_refuses_a_git_checkout(self):
        os.makedirs(os.path.join(self.dest, ".git"))
        with self.assertRaises(ValueError):
            appinstall.uninstall(self.dest, everything=True)
        self.assertTrue(self.exists("backend/server.py"))

    def test_keep_mode_removes_the_app_and_keeps_settings_and_models(self):
        r = appinstall.uninstall(self.dest)
        for rel in ("backend", "web", "run.sh", "engines", "logs", "stats.json",
                    appinstall.MANIFEST):
            self.assertFalse(self.exists(rel), rel)
        for rel in ("config.json", "models.ini", "models/a.gguf", "my-notes.txt"):
            self.assertTrue(self.exists(rel), rel)
        self.assertIn("my-notes.txt", r["kept"])

    def test_everything_still_spares_files_it_does_not_know(self):
        r = appinstall.uninstall(self.dest, everything=True)
        for rel in ("config.json", "models.ini", "models", "backend", "engines"):
            self.assertFalse(self.exists(rel), rel)
        self.assertTrue(self.exists("my-notes.txt"))
        self.assertEqual(r["kept"], ["my-notes.txt"])

    def test_everything_removes_the_folder_once_empty(self):
        os.remove(os.path.join(self.dest, "my-notes.txt"))
        appinstall.uninstall(self.dest, everything=True)
        self.assertFalse(os.path.exists(self.dest))

    def test_leaves_the_private_python_for_the_script(self):
        """Windows locks python.exe while it runs this; uninstall.ps1 removes it after."""
        write(self.dest, "python/python.exe")
        appinstall.uninstall(self.dest, everything=True)
        self.assertTrue(self.exists("python/python.exe"))

    def test_cli(self):
        self.assertEqual(appinstall.main(["--uninstall", self.dest, "--all"]), 0)
        self.assertFalse(self.exists("models"))
        os.makedirs(self.dest, exist_ok=True)
        self.assertEqual(appinstall.main(["--uninstall", self.dest]), 1)


if __name__ == "__main__":
    unittest.main()
