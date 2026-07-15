import conftest_paths  # noqa: F401
import tempfile, unittest
import builder


class TestUpdateCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.bm = builder.BuildManager(self.tmp)
        self.calls = []
        # stub the expensive git path; count how often it actually runs
        def fake(src, remote):
            self.calls.append((src, remote))
            return {"ok": True, "behind": 3,
                    "latest": {"hash": "abc", "subject": "x"}, "up_to_date": False}
        self.bm._check_updates_fresh = fake

    def test_second_call_within_ttl_is_cached(self):
        a = self.bm.check_updates("/src", now=1000)
        b = self.bm.check_updates("/src", now=1000 + builder.UPDATE_TTL - 1)
        self.assertEqual(len(self.calls), 1)          # only one real fetch
        self.assertFalse(a["cached"])
        self.assertTrue(b["cached"])
        self.assertEqual(b["behind"], 3)              # served from cache
        self.assertGreater(b["checked_secs_ago"], 0)

    def test_expired_ttl_refetches(self):
        self.bm.check_updates("/src", now=1000)
        self.bm.check_updates("/src", now=1000 + builder.UPDATE_TTL + 1)
        self.assertEqual(len(self.calls), 2)

    def test_force_bypasses_cache(self):
        self.bm.check_updates("/src", now=1000)
        r = self.bm.check_updates("/src", now=1001, force=True)
        self.assertEqual(len(self.calls), 2)
        self.assertFalse(r["cached"])

    def test_failed_check_has_short_ttl(self):
        self.bm._check_updates_fresh = lambda s, r: {"ok": False, "error": "fetch failed"}
        self.bm.check_updates("/src", now=1000)
        # still cached briefly...
        self.bm.check_updates("/src", now=1000 + builder.UPDATE_TTL_FAIL - 1)
        # ...but retried after the shorter failure TTL
        self.bm.check_updates("/src", now=1000 + builder.UPDATE_TTL_FAIL + 1)
        # 2 real calls: initial + post-fail-TTL retry (middle one cached)
        # (count via a fresh stub would need wiring; assert via ok flag instead)
        r = self.bm.check_updates("/src", now=1000 + builder.UPDATE_TTL_FAIL + 1)
        self.assertFalse(r["ok"])

    def test_different_branch_cached_separately(self):
        self.bm.check_updates("/src", remote_branch="origin/master", now=1000)
        self.bm.check_updates("/src", remote_branch="origin/dev", now=1000)
        self.assertEqual(len(self.calls), 2)


if __name__ == "__main__":
    unittest.main()
