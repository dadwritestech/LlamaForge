import conftest_paths  # noqa: F401
import json
import os
import shutil
import tempfile
import time
import unittest
from unittest import mock

import config
import hub
import reupload
import routes

SHA = "40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199"


def _w(path, data):
    with open(path, "wb") as f:
        f.write(data)


INFO = [{"type": "file", "path": "Q.gguf", "size": 10,
         "lfs": {"oid": SHA, "size": 10},
         "lastCommit": {"id": "c0ffee", "title": "Fix chat template", "date": "2026-10-30T12:00:00.000Z"}}]


class Local(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_hf_cache_blob_names_its_sha(self):
        blobs = os.path.join(self.d, "models--o--r", "blobs")
        os.makedirs(blobs)
        p = os.path.join(blobs, SHA)
        _w(p, b"x" * 10)
        self.assertEqual(reupload.local(p)["sha256"], SHA)

    def test_plain_file_has_size_and_mtime_only(self):
        p = os.path.join(self.d, "Q.gguf")
        _w(p, b"x" * 10)
        out = reupload.local(p)
        self.assertEqual(out["size"], 10)
        self.assertIsNone(out["sha256"])
        self.assertAlmostEqual(out["mtime"], os.path.getmtime(p), places=0)


class Remote(unittest.TestCase):
    def test_paths_info(self):
        with mock.patch.object(reupload, "_paths_info", return_value=INFO):
            r = reupload.remote("o/r", "Q.gguf")
        self.assertEqual(r, {"path": "Q.gguf", "size": 10, "sha256": SHA, "commit": "c0ffee",
                             "title": "Fix chat template", "date": "2026-10-30T12:00:00.000Z"})

    def test_flattened_download_is_found_in_the_tree(self):
        # LlamaForge saves repo/sub/Q.gguf as <dest>/Q.gguf: paths-info misses, the tree finds it
        hit = dict(INFO[0], path="UD/Q.gguf")
        with mock.patch.object(reupload, "_paths_info", side_effect=lambda repo, paths: [hit] if paths == ["UD/Q.gguf"] else []), \
                mock.patch.object(reupload, "_tree", return_value=["README.md", "UD/Q.gguf", "UD/Q2.gguf"]):
            r = reupload.remote("o/r", "Q.gguf")
        self.assertEqual(r["path"], "UD/Q.gguf")

    def test_gone(self):
        with mock.patch.object(reupload, "_paths_info", return_value=[]), \
                mock.patch.object(reupload, "_tree", return_value=["other.gguf"]):
            self.assertIsNone(reupload.remote("o/r", "Q.gguf"))


class Verdict(unittest.TestCase):
    R = {"size": 10, "sha256": SHA, "date": "2026-10-30T12:00:00.000Z"}
    after = time.mktime((2026, 10, 31, 0, 0, 0, 0, 0, -1))
    before = time.mktime((2026, 10, 1, 0, 0, 0, 0, 0, -1))

    def test_sha_decides_when_known(self):
        self.assertEqual(reupload.verdict({"sha256": SHA, "size": 10, "mtime": self.before}, self.R), "current")
        self.assertEqual(reupload.verdict({"sha256": "0" * 64, "size": 10, "mtime": self.after}, self.R), "reuploaded")

    def test_without_sha_size_then_date(self):
        self.assertEqual(reupload.verdict({"sha256": None, "size": 9, "mtime": self.after}, self.R), "reuploaded")
        self.assertEqual(reupload.verdict({"sha256": None, "size": 10, "mtime": self.after}, self.R), "current")
        self.assertEqual(reupload.verdict({"sha256": None, "size": 10, "mtime": self.before}, self.R), "changed")
        self.assertEqual(reupload.verdict({"sha256": None, "size": 10, "mtime": self.before}, None), "gone")


class Scan(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.cache = os.path.join(self.d, "reupload.json")
        self._p = mock.patch.object(reupload, "CACHE", self.cache)
        self._p.start()
        self.dl = os.path.join(self.d, "o--r")
        os.makedirs(self.dl)
        self.f = os.path.join(self.dl, "Q.gguf")
        _w(self.f, b"x" * 9)
        os.utime(self.f, (Verdict.before, Verdict.before))

    def tearDown(self):
        self._p.stop()
        shutil.rmtree(self.d, ignore_errors=True)

    def test_scan_checks_named_models_and_caches(self):
        models = {"m": self.f, "loose": os.path.join(self.d, "x.gguf")}
        with mock.patch.object(reupload, "_paths_info", return_value=INFO) as pi:
            out = reupload.scan(models)
            self.assertEqual(out["models"]["m"]["verdict"], "reuploaded")
            self.assertEqual(out["models"]["m"]["repo"], "o/r")
            self.assertEqual(out["models"]["m"]["title"], "Fix chat template")
            self.assertNotIn("loose", out["models"])
            self.assertTrue(out["checked_at"])
            again = reupload.cached(models)
        self.assertEqual(pi.call_count, 1)
        self.assertEqual(again["models"]["m"]["verdict"], "reuploaded")

    def test_cached_drops_models_that_left_and_offline_is_an_error_not_a_verdict(self):
        with mock.patch.object(reupload, "_paths_info", side_effect=OSError("offline")):
            out = reupload.scan({"m": self.f})
        self.assertEqual(out["models"]["m"]["verdict"], "unknown")
        self.assertIn("offline", out["models"]["m"]["error"])
        self.assertEqual(reupload.cached({})["models"], {})

    def test_stale(self):
        self.assertTrue(reupload.stale({"checked_at": 0}))
        self.assertFalse(reupload.stale({"checked_at": time.time()}))


class Overwrite(unittest.TestCase):
    def test_overwrite_refetches_existing_files(self):
        dm = hub.DownloadManager()
        got = []
        with tempfile.TemporaryDirectory() as d:
            _w(os.path.join(d, "Q.gguf"), b"old")
            dm._fetch = lambda url, dest: got.append(dest)
            dm._run("o/r", ["UD/Q.gguf"], d)
            self.assertEqual(got, [])                 # default: already there, skipped
            dm._overwrite = True
            dm._run("o/r", ["UD/Q.gguf"], d)
        self.assertEqual(len(got), 1)


class Routes(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.ini = os.path.join(self.d, "models.ini")
        self.cfg = os.path.join(self.d, "config.json")
        self._orig = config.CONFIG
        config.CONFIG = self.cfg
        self.dl = os.path.join(self.d, "o--r")
        os.makedirs(self.dl)
        self.f = os.path.join(self.dl, "Q.gguf")
        _w(self.f, b"x" * 9)
        with open(self.cfg, "w", encoding="utf-8") as f:
            json.dump({"models_ini": self.ini, "active_engine": "llamacpp", "download_dir": self.d}, f)
        with open(self.ini, "w", encoding="utf-8") as f:
            f.write(f"[*]\n\n[m]\nmodel = {self.f}\n")
        self._p = mock.patch.object(reupload, "CACHE", os.path.join(self.d, "reupload.json"))
        self._p.start()

    def tearDown(self):
        self._p.stop()
        config.CONFIG = self._orig
        shutil.rmtree(self.d, ignore_errors=True)

    def test_refresh_then_fetch_starts_an_overwriting_download(self):
        with mock.patch.object(reupload, "_paths_info", return_value=INFO):
            st, out = routes.GET_ROUTES["/api/reupload"](routes.Req(qs={"refresh": "1"}))
        self.assertEqual(out["models"]["m"]["verdict"], "reuploaded")
        with mock.patch.object(routes.DOWNLOADS, "start", return_value=True) as start, \
                mock.patch.object(routes, "_reload_preset", return_value=False):
            st, out = routes.POST_ROUTES["/api/reupload/fetch"](routes.Req(body={"model": "m"}))
        self.assertTrue(out["ok"])
        start.assert_called_once_with("o/r", ["Q.gguf"], self.dl, overwrite=True)

    def test_fetch_refuses_unchecked_or_current(self):
        st, out = routes.POST_ROUTES["/api/reupload/fetch"](routes.Req(body={"model": "m"}))
        self.assertFalse(out["ok"])


if __name__ == "__main__":
    unittest.main()
