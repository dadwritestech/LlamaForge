import conftest_paths  # noqa: F401
import os, tempfile, unittest
import stats


class TrackerCase(unittest.TestCase):
    def setUp(self):
        self._orig = stats.STATS_FILE
        fd, self.path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.unlink(self.path)              # tracker should cope with no file
        stats.STATS_FILE = self.path
        self.tr = stats.StatsTracker()

    def tearDown(self):
        stats.STATS_FILE = self._orig
        if os.path.exists(self.path):
            os.unlink(self.path)


class TestGenSecs(TrackerCase):
    def test_gen_secs_accumulates_only_when_generating(self):
        with self.tr.lock:
            self.tr._record_tokens("m", 100, 50)    # generating
            self.tr._record_tokens("m", 100, 0)     # prompt only
            self.tr._record_tokens("m", 0, 50)      # generating
        m = self.tr.data["models"]["m"]
        self.assertEqual(m["gen_secs"], 2 * stats.POLL_SECS)

    def test_avg_tps_in_summary(self):
        with self.tr.lock:
            self.tr._record_tokens("m", 0, 100)     # 100 tok in one POLL_SECS window
        row = self.tr.summary()["per_model"][0]
        self.assertEqual(row["avg_tps"], round(100 / stats.POLL_SECS, 1))

    def test_legacy_model_without_gen_secs_reports_zero(self):
        with self.tr.lock:
            self.tr.data["models"]["old"] = {"prompt": 1, "generated": 2,
                                             "loaded_secs": 3, "runs": 1, "last_used": 0}
        row = self.tr.summary()["per_model"][0]
        self.assertEqual(row["avg_tps"], 0)


class TestReset(TrackerCase):
    def test_reset_zeroes_store_and_flushes(self):
        with self.tr.lock:
            self.tr._record_tokens("m", 10, 10)
        self.tr.reset()
        s = self.tr.summary()
        self.assertEqual(s["per_model"], [])
        self.assertEqual(s["totals"]["tokens"], 0)
        self.assertTrue(os.path.exists(self.path))  # flushed to disk


if __name__ == "__main__":
    unittest.main()
