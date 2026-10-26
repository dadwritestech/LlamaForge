import conftest_paths  # noqa: F401
import unittest

from forge import bench, prompts


class Pool(unittest.TestCase):
    def test_deterministic_unique_and_big_enough(self):
        a, b = prompts.pool(), prompts.pool()
        self.assertEqual(a, b)
        for cat, need in (("code", 60), ("prose", 60), ("editing", 20), ("summary", 20)):
            self.assertGreaterEqual(len(a[cat]), need, cat)
            self.assertEqual(len(set(a[cat])), len(a[cat]), cat)

    def test_summary_prompts_are_long_enough_to_time_prefill(self):
        self.assertTrue(all(len(p) > 6000 for p in prompts.pool()["summary"]))

    def test_select_is_fixed_per_round_and_rounds_differ(self):
        a = prompts.select("code", 8, "A")
        self.assertEqual(a, prompts.select("code", 8, "A"))
        self.assertEqual(len(set(a)), 8)
        self.assertLess(len(set(a) & set(prompts.select("code", 8, "B"))), 3)

    def test_select_never_repeats_by_wrapping(self):
        with self.assertRaises(ValueError):
            prompts.select("editing", 10_000, "A")


def timings(decode, prefill, n, prompt_n=900, draft=0, acc=0):
    return {"timings": {"predicted_per_second": decode, "prompt_per_second": prefill,
                        "predicted_n": n, "prompt_n": prompt_n, "draft_n": draft,
                        "draft_n_accepted": acc}}


class Measure(unittest.TestCase):
    def test_request_is_cache_off_seeded_and_uses_server_sampler(self):
        seen = {}

        def post(path, body):
            seen.update(path=path, body=body)
            return timings(100.0, 1500.0, 256)

        r = bench.measure(post, "hello", 256, {"enable_thinking": False})
        b = seen["body"]
        self.assertEqual(seen["path"], "/v1/chat/completions")
        self.assertFalse(b["cache_prompt"])
        self.assertEqual(b["max_tokens"], 256)
        self.assertIn("seed", b)
        self.assertNotIn("temperature", b)         # the user's own sampler, from models.ini
        self.assertEqual(b["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(r, {"decode": 100.0, "prefill": 1500.0, "n": 256, "prompt_n": 900,
                             "draft": 0, "accepted": 0})


class Summarize(unittest.TestCase):
    def test_token_weighted_median_spread_and_acceptance(self):
        s = bench.summarize([
            {"decode": 100.0, "prefill": 1000.0, "n": 200, "prompt_n": 50, "draft": 100, "accepted": 60},
            {"decode": 50.0, "prefill": 1200.0, "n": 100, "prompt_n": 50, "draft": 100, "accepted": 40},
            {"decode": 80.0, "prefill": 1100.0, "n": 200, "prompt_n": 50, "draft": 0, "accepted": 0},
        ])
        # 500 tokens over 2 + 2 + 2.5 s
        self.assertAlmostEqual(s["decode"], 500 / 6.5, places=2)
        self.assertEqual(s["median"], 80.0)
        self.assertAlmostEqual(s["spread"], (100 - 50) / 80, places=3)
        self.assertAlmostEqual(s["acceptance"], 0.5)
        self.assertEqual(s["n"], 3)

    def test_short_answers_dont_count(self):
        s = bench.summarize([{"decode": 300.0, "prefill": 1, "n": 5, "prompt_n": 1, "draft": 0, "accepted": 0},
                             {"decode": 90.0, "prefill": 1, "n": 200, "prompt_n": 1, "draft": 0, "accepted": 0}],
                            min_tokens=32)
        self.assertEqual(s["decode"], 90.0)
        self.assertEqual(s["n"], 1)

    def test_empty(self):
        self.assertIsNone(bench.summarize([]))


class Round(unittest.TestCase):
    def test_warms_up_then_measures_each_category_with_distinct_prompts(self):
        calls = []

        def post(path, body):
            p = body["messages"][0]["content"]
            calls.append(p)
            if p.startswith("Summarise"):
                return timings(60.0, 2000.0, 128, prompt_n=2000)
            return timings(100.0, 900.0, 256)

        r = bench.run_round(post, {"code": 3, "prose": 3, "summary": 2}, "A", n_predict=256)
        self.assertEqual(calls[0], bench.WARMUP)
        self.assertEqual(len(calls), 1 + 8)
        self.assertEqual(len(set(calls)), len(calls))   # nothing repeated in one process
        self.assertEqual(r["code"], 100.0)
        self.assertEqual(r["summary"], 60.0)
        self.assertEqual(r["prefill"], 2000.0)          # from the long summary prompts only
        self.assertIn("code", r["detail"])
        self.assertIsNone(r["error"])

    def test_request_error_ends_the_round_with_a_reason(self):
        def post(path, body):
            raise OSError("connection reset")

        r = bench.run_round(post, {"code": 2}, "A")
        self.assertIn("connection reset", r["error"])


if __name__ == "__main__":
    unittest.main()
