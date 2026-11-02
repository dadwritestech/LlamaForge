"""Proving ground: built-in tasks with answers a script can check, the
checkers, and the morning verdict. Model output is only ever compared."""
import unittest

import conftest_paths  # noqa: F401
import proving


def T(tid):
    return next(t for t in proving.TASKS if t["id"] == tid)


class Tasks(unittest.TestCase):
    def test_pack_shape(self):
        ids = [t["id"] for t in proving.TASKS]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreaterEqual(len(ids), 15)
        self.assertEqual(sorted({t["cat"] for t in proving.TASKS}), sorted(proving.CATEGORIES))
        for t in proving.TASKS:
            self.assertIn(t["check"], proving.CHECKS, t["id"])
            self.assertTrue(t["prompt"].strip())

    def test_every_expected_answer_passes_its_own_check(self):
        for t in proving.TASKS:
            self.assertTrue(proving.check(t, proving.sample_answer(t)), t["id"])

    def test_answers_worked_out_independently(self):
        self.assertEqual(T("math-kv")["expect"], (16 - 9.5 - 1) / 0.25 * 4096)
        self.assertEqual(T("math-pct")["expect"], round(2340 * 0.17, 2))
        b = next(b for b in range(40) if b + (b + 2) + 2 * (b + 2) == 34)
        self.assertEqual(T("reason-boxes")["expect"], b)
        days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        self.assertEqual(T("reason-days")["expect"], days[(2 + 100) % 7])
        self.assertEqual(T("json-prime")["expect"], {"ok": all(221 % d for d in range(2, 221))})
        self.assertEqual(T("follow-reverse")["expect"], "llama"[::-1].upper())


class Checks(unittest.TestCase):
    def test_exact_tolerates_case_punctuation_and_backticks(self):
        t = T("extract-invoice")
        self.assertTrue(proving.check(t, "`INV-20931`."))
        self.assertTrue(proving.check(t, "inv-20931"))
        self.assertFalse(proving.check(t, "INV-20391"))
        self.assertFalse(proving.check(t, "The invoice is INV-20931"))

    def test_exact_case_means_case(self):
        t = T("follow-reverse")
        self.assertTrue(proving.check(t, "AMALL"))
        self.assertFalse(proving.check(t, "amall"))

    def test_number_takes_the_last_number(self):
        t = T("math-kv")
        self.assertTrue(proving.check(t, "90,112"))
        self.assertTrue(proving.check(t, "5.5 GB left, so 90112"))
        self.assertFalse(proving.check(t, "90112 tokens? no wait, 86016"))
        self.assertFalse(proving.check(t, "lots"))

    def test_thinking_is_ignored(self):
        t = T("reason-boxes")
        self.assertTrue(proving.check(t, "<think>maybe 9</think>7"))
        self.assertFalse(proving.check(t, "<think>7</think>9"))

    def test_json_strips_fences_and_compares_values(self):
        t = T("json-person")
        self.assertTrue(proving.check(t, '```json\n{"gpus": 3, "name": "Rin"}\n```'))
        self.assertFalse(proving.check(t, '{"name": "Rin", "gpus": "3"}'))
        self.assertFalse(proving.check(t, "not json {"))

    def test_lines_set_words_and_csv(self):
        self.assertTrue(proving.check(T("follow-colors"), "Red\nGreen\nBlue"))
        self.assertFalse(proving.check(T("follow-colors"), "1. Red\n2. Green\n3. Blue\n4. Yellow"))
        self.assertTrue(proving.check(T("follow-words"), "The sea was calm and very blue."))
        self.assertFalse(proving.check(T("follow-words"), "The sea is blue."))
        self.assertTrue(proving.check(T("reason-sort"), "apple, banana, cherry, mango, pear"))
        self.assertFalse(proving.check(T("reason-sort"), "apple, banana, mango, cherry, pear"))

    def test_none_and_empty_never_pass(self):
        for t in proving.TASKS:
            self.assertFalse(proving.check(t, None), t["id"])
            self.assertFalse(proving.check(t, ""), t["id"])


def run(passing, tok_s, cut=()):
    """A model's results: passing = task ids it got right."""
    return {t["id"]: {"pass": t["id"] in passing, "tok_s": tok_s, "cut": t["id"] in cut}
            for t in proving.TASKS}


class Verdict(unittest.TestCase):
    def setUp(self):
        ids = [t["id"] for t in proving.TASKS]
        self.math = [i for i in ids if i.startswith("math-")]
        self.all = ids

    def test_score(self):
        s = proving.score(run(self.math, 50.0))
        self.assertEqual(s["passed"], len(self.math))
        self.assertEqual(s["total"], len(self.all))
        self.assertEqual(s["cats"]["math"], [len(self.math), len(self.math)])
        self.assertEqual(s["tok_s"], 50.0)

    def test_beats_driver_slower(self):
        res = {"big": run(self.all, 30.0), "daily": run(self.math, 50.0)}
        self.assertEqual(proving.verdict(res, "big", "daily"),
                         "big beats daily on 4 of 5 categories (1 tie), 40% slower.")

    def test_faster_and_worse(self):
        res = {"small": run([], 100.0), "daily": run(self.math, 50.0)}
        self.assertEqual(proving.verdict(res, "small", "daily"),
                         "small beats daily on 0 of 5 categories (4 ties), 2.0x faster.")

    def test_same_speed_and_missing(self):
        res = {"a": run(self.all, 51.0), "daily": run(self.all, 50.0)}
        self.assertEqual(proving.verdict(res, "a", "daily"),
                         "a beats daily on 0 of 5 categories (5 ties), about as fast.")
        self.assertIsNone(proving.verdict(res, "a", "nope"))
        self.assertIsNone(proving.verdict(res, "a", "a"))


if __name__ == "__main__":
    unittest.main()
