import conftest_paths  # noqa: F401
import json
import os
import shutil
import tempfile
import time
import unittest
from unittest import mock

import config
import reclaim
import routes

DAY = 86400


def touch(path, data=b"GGUF", size=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
        if size:
            f.write(b"\0" * (size - len(data)))
    return path


class Files(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_a_split_model_is_all_of_its_shards(self):
        a = touch(os.path.join(self.d, "M-00001-of-00003.gguf"))
        b = touch(os.path.join(self.d, "M-00002-of-00003.gguf"))
        c = touch(os.path.join(self.d, "M-00003-of-00003.gguf"))
        touch(os.path.join(self.d, "Other-00001-of-00002.gguf"))
        self.assertEqual(reclaim.files_of(a), [a, b, c])

    def test_a_single_file_is_itself_and_missing_is_empty(self):
        a = touch(os.path.join(self.d, "M.gguf"))
        self.assertEqual(reclaim.files_of(a), [a])
        self.assertEqual(reclaim.files_of(os.path.join(self.d, "gone.gguf")), [])


class Report(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.now = 1_800_000_000

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_stale_models_are_the_ones_unused_for_days_largest_first(self):
        old = touch(os.path.join(self.d, "old.gguf"), size=3000)
        big = touch(os.path.join(self.d, "never.gguf"), size=5000)
        new = touch(os.path.join(self.d, "new.gguf"), size=4000)
        secs = {"*": {"ctx-size": "4096"}, "old": {"model": old}, "never": {"model": big},
                "new": {"model": new}, "gone": {"model": os.path.join(self.d, "x.gguf")}}
        used = {"old": self.now - 120 * DAY, "new": self.now - 2 * DAY}
        r = reclaim.report(secs, used, since=self.now - 365 * DAY, now=self.now, days=90)
        self.assertEqual([m["id"] for m in r["stale"]], ["never", "old"])
        self.assertEqual(r["stale_bytes"], 8000)
        self.assertEqual(r["stale"][0]["last_used"], 0)
        self.assertTrue(r["tracked_long_enough"])

    def test_young_stats_say_so(self):
        a = touch(os.path.join(self.d, "a.gguf"))
        r = reclaim.report({"a": {"model": a}}, {}, since=self.now - 5 * DAY, now=self.now, days=90)
        self.assertFalse(r["tracked_long_enough"])
        self.assertEqual([m["id"] for m in r["stale"]], ["a"])

    def test_duplicates_are_same_size_and_same_bytes_on_different_paths(self):
        body = b"GGUF" + os.urandom(64)
        a = touch(os.path.join(self.d, "c1", "m.gguf"), data=body, size=2048)
        b = touch(os.path.join(self.d, "c2", "m-copy.gguf"), data=body, size=2048)
        c = touch(os.path.join(self.d, "c3", "other.gguf"), data=b"GGUF" + os.urandom(64), size=2048)
        secs = {"m": {"model": a}, "m2": {"model": b}, "o": {"model": c}, "alias": {"model": a}}
        r = reclaim.report(secs, {}, since=self.now, now=self.now, days=90)
        self.assertEqual(len(r["dups"]), 1)
        self.assertEqual(sorted(r["dups"][0]["ids"]), ["alias", "m", "m2"])
        self.assertEqual(r["dups"][0]["copies"], 2)
        self.assertEqual([f["ids"] for f in r["dups"][0]["files"]], [["alias", "m"], ["m2"]])
        self.assertEqual(r["dup_bytes"], 2048)     # one copy can go

    def test_same_header_different_weights_is_not_a_duplicate(self):
        head = b"GGUF" + b"h" * (reclaim.HEAD - 4)
        size = reclaim.HEAD + 3 * reclaim.SAMPLE
        a = touch(os.path.join(self.d, "base.gguf"), data=head + b"" * (size - len(head)))
        b = touch(os.path.join(self.d, "tune.gguf"), data=head + b"" * (size - len(head)))
        r = reclaim.report({"a": {"model": a}, "b": {"model": b}}, {}, since=self.now, now=self.now, days=90)
        self.assertEqual(r["dups"], [])


class Delete(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_deletes_every_shard_and_reports_bytes(self):
        a = touch(os.path.join(self.d, "M-00001-of-00002.gguf"), size=100)
        b = touch(os.path.join(self.d, "M-00002-of-00002.gguf"), size=50)
        freed = reclaim.delete_model("m", {"m": {"model": a}})
        self.assertEqual(freed, 150)
        self.assertFalse(os.path.exists(a) or os.path.exists(b))

    def test_refuses_what_is_not_a_registered_gguf(self):
        txt = touch(os.path.join(self.d, "notes.txt"))
        g = touch(os.path.join(self.d, "m.gguf"))
        with self.assertRaises(ValueError):
            reclaim.delete_model("t", {"t": {"model": txt}})
        with self.assertRaises(ValueError):
            reclaim.delete_model("nope", {"m": {"model": g}})
        with self.assertRaises(ValueError):
            reclaim.delete_model("*", {"*": {"model": g}})
        self.assertTrue(os.path.exists(txt) and os.path.exists(g))

    def test_refuses_a_file_another_model_still_uses(self):
        g = touch(os.path.join(self.d, "m.gguf"))
        with self.assertRaises(ValueError) as cm:
            reclaim.delete_model("a", {"a": {"model": g}, "b": {"model": g}})
        self.assertIn("b", str(cm.exception))
        self.assertTrue(os.path.exists(g))


class Routes(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.ini = os.path.join(self.d, "models.ini")
        self.cfg = os.path.join(self.d, "config.json")
        self._orig = config.CONFIG
        config.CONFIG = self.cfg
        with open(self.cfg, "w", encoding="utf-8") as f:
            json.dump({"models_ini": self.ini, "active_engine": "llamacpp"}, f)
        self.g = touch(os.path.join(self.d, "m.gguf"), size=1000)
        with open(self.ini, "w", encoding="utf-8") as f:
            f.write("[*]\nctx-size = 4096\n\n[m]\nmodel = %s\n" % self.g)

    def tearDown(self):
        config.CONFIG = self._orig
        shutil.rmtree(self.d, ignore_errors=True)

    def test_registered(self):
        self.assertIn("/api/reclaim", routes.GET_ROUTES)
        self.assertIn("/api/reclaim/delete", routes.POST_ROUTES)

    def test_get_lists_the_unused_model(self):
        req = routes.Req(qs={"days": "90"})
        st, out = routes.GET_ROUTES["/api/reclaim"](req)
        self.assertEqual(st, 200)
        self.assertEqual([m["id"] for m in out["stale"]], ["m"])

    def test_delete_unloads_removes_file_and_section(self):
        calls = []

        def fake_router(path, method="GET", body=None):
            calls.append((path, method, body))
            if path == "/models":
                return 200, {"data": [{"id": "m", "status": {"value": "loaded"}}]}
            return 200, {}
        with mock.patch.object(routes, "router", fake_router), \
                mock.patch.object(routes, "reload_presets", return_value=(200, {})):
            st, out = routes.POST_ROUTES["/api/reclaim/delete"](routes.Req(body={"id": "m"}))
        self.assertEqual(st, 200)
        self.assertTrue(out["ok"])
        self.assertEqual(out["freed"], 1000)
        self.assertIn(("/models/unload", "POST", {"model": "m"}), calls)
        self.assertFalse(os.path.exists(self.g))
        self.assertNotIn("m", config.read_sections())

    def test_delete_refusal_keeps_everything(self):
        with mock.patch.object(routes, "router", return_value=(200, {"data": []})):
            st, out = routes.POST_ROUTES["/api/reclaim/delete"](routes.Req(body={"id": "zz"}))
        self.assertFalse(out["ok"])
        self.assertTrue(os.path.exists(self.g))


if __name__ == "__main__":
    unittest.main()
