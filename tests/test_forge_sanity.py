import conftest_paths  # noqa: F401
import unittest

from forge import sanity


class Divergence(unittest.TestCase):
    def test_first_divergence(self):
        self.assertIsNone(sanity.first_divergence([1, 2, 3], [1, 2, 3]))
        self.assertIsNone(sanity.first_divergence([1, 2, 3], [1, 2]))   # a prefix isn't a split
        self.assertEqual(sanity.first_divergence([1, 2, 3], [1, 9, 3]), 1)


class Check(unittest.TestCase):
    def test_identical_passes(self):
        plain = {"p1": [1] * 64, "p2": [2] * 64}
        r = sanity.check(plain, dict(plain))
        self.assertTrue(r["ok"])
        self.assertEqual(r["early"], 0)

    def test_late_float_drift_is_tolerated(self):
        plain = {f"p{i}": list(range(64)) for i in range(6)}
        spec = {k: v[:50] + [999] * 14 for k, v in plain.items()}
        self.assertTrue(sanity.check(plain, spec)["ok"])

    def test_two_early_splits_fail(self):
        plain = {f"p{i}": list(range(64)) for i in range(6)}
        spec = dict(plain)
        spec["p0"] = [0, 1, 2, 77] + list(range(4, 64))
        spec["p3"] = [5] * 64
        r = sanity.check(plain, spec)
        self.assertFalse(r["ok"])
        self.assertEqual(r["early"], 2)
        self.assertIn("2 of 6", r["reason"])

    def test_missing_prompts_are_skipped(self):
        r = sanity.check({"a": [1, 2]}, {"b": [1, 2]})
        self.assertTrue(r["ok"])
        self.assertEqual(r["compared"], 0)


class Sample(unittest.TestCase):
    def test_greedy_tokens_cache_off(self):
        seen = {}

        def post(path, body):
            seen.update(path=path, body=body)
            return {"tokens": [5, 6, 7], "content": "x"}

        self.assertEqual(sanity.sample(post, "hi", 64), [5, 6, 7])
        b = seen["body"]
        self.assertEqual(seen["path"], "/completion")
        self.assertEqual(b["temperature"], 0)
        self.assertTrue(b["return_tokens"])
        self.assertFalse(b["cache_prompt"])
        self.assertEqual(b["n_predict"], 64)


if __name__ == "__main__":
    unittest.main()
