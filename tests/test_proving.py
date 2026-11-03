"""Proving ground: built-in tasks with answers a script can check, the
checkers, and the morning verdict. Model output is only ever compared."""
import os
import tempfile
import threading
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
        self.assertTrue(proving.check(t, '```json\n{"gpus": 3, "owner": "Rin"}\n```'))
        self.assertFalse(proving.check(t, '{"owner": "Rin", "gpus": "3"}'))
        self.assertFalse(proving.check(t, '{"owner": "Rin\'s workstation", "gpus": 3}'))
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
                         "small trails daily on 1 of 5 categories (4 ties), 2.0x faster.")

    def test_same_speed_and_missing(self):
        res = {"a": run(self.all, 51.0), "daily": run(self.all, 50.0)}
        self.assertEqual(proving.verdict(res, "a", "daily"),
                         "a matches daily on all 5 categories, about as fast.")
        self.assertIsNone(proving.verdict(res, "a", "nope"))
        self.assertIsNone(proving.verdict(res, "a", "a"))


class RunnerTests(unittest.TestCase):
    """The background run, with loading and asking passed in (no router)."""
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.remove(self.path)
        self.addCleanup(lambda: os.path.exists(self.path) and os.remove(self.path))
        self.loaded, self.released, self.asked = [], [], []

    def runner(self, answer=None, ensure=None):
        def ask(mid, prompt):
            self.asked.append(mid)
            t = next(t for t in proving.TASKS if t["prompt"] == prompt)
            return (answer or (lambda mid, t: {"text": proving.sample_answer(t), "tok_s": 40.0, "cut": False}))(mid, t)
        return proving.Runner(ask, ensure or self.loaded.append, lambda: self.released.append(1), path=self.path)

    def test_runs_every_task_on_every_model_and_saves(self):
        r = self.runner()
        self.assertTrue(r.start(["a", "b"]))
        r.join(5)
        st = r.status()
        self.assertFalse(st["running"])
        self.assertEqual((st["done"], st["total"]), (2 * len(proving.TASKS), 2 * len(proving.TASKS)))
        self.assertEqual(self.loaded, ["a", "b"])
        self.assertEqual(self.released, [1])
        saved = proving.load(self.path)["models"]
        self.assertEqual(proving.score(saved["a"]["results"])["passed"], len(proving.TASKS))
        self.assertIn("at", saved["b"])

    def test_wrong_and_cut_answers_are_kept_short_and_marked(self):
        r = self.runner(answer=lambda mid, t: {"text": "x" * 5000, "tok_s": 10.0, "cut": True})
        r.start(["a"])
        r.join(5)
        res = proving.load(self.path)["models"]["a"]["results"]
        one = res[proving.TASKS[0]["id"]]
        self.assertFalse(one["pass"])
        self.assertTrue(one["cut"])
        self.assertLessEqual(len(one["answer"]), proving.MAX_KEEP)

    def test_a_model_that_wont_load_is_skipped_with_its_error(self):
        def ensure(mid):
            if mid == "bad":
                raise RuntimeError("needs 28.7 GiB")
            self.loaded.append(mid)
        r = self.runner(ensure=ensure)
        r.start(["bad", "a"])
        r.join(5)
        st = r.status()
        self.assertEqual(st["errors"], {"bad": "needs 28.7 GiB"})
        self.assertEqual(list(proving.load(self.path)["models"]), ["a"])

    def test_an_answer_error_fails_that_task_only(self):
        def answer(mid, t):
            if t["id"] == "math-pct":
                raise RuntimeError("HTTP 500")
            return {"text": proving.sample_answer(t), "tok_s": 40.0, "cut": False}
        r = self.runner(answer=answer)
        r.start(["a"])
        r.join(5)
        res = proving.load(self.path)["models"]["a"]["results"]
        self.assertFalse(res["math-pct"]["pass"])
        self.assertEqual(res["math-pct"]["answer"], "error: HTTP 500")
        self.assertTrue(res["math-kv"]["pass"])

    def test_cancel_stops_and_keeps_old_results(self):
        gate = threading.Event()
        proving.save_model(self.path, "a", {"math-kv": {"pass": True, "tok_s": 1.0, "cut": False}})
        def answer(mid, t):
            gate.wait(5)
            return {"text": "", "tok_s": None, "cut": False}
        r = self.runner(answer=answer)
        r.start(["a"])
        self.assertFalse(r.start(["b"]), "one run at a time")
        r.cancel()
        gate.set()
        r.join(5)
        self.assertFalse(r.status()["running"])
        self.assertTrue(r.status()["cancelled"])
        self.assertEqual(proving.load(self.path)["models"]["a"]["results"], {"math-kv": {"pass": True, "tok_s": 1.0, "cut": False}})
        self.assertEqual(self.released, [1])

    def test_load_survives_junk(self):
        with open(self.path, "w") as f:
            f.write("[1,2")
        self.assertEqual(proving.load(self.path), {"models": {}})


class Routes(unittest.TestCase):
    """/api/proving*: the panel side, router mocked (same harness as the arena)."""
    def setUp(self):
        from unittest import mock
        import routes
        self.routes = routes
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.remove(path)
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        reg = mock.Mock()
        reg.state.return_value = {"models": [
            {"id": "qwen", "backend": "llamacpp", "settings": {}},
            {"id": "gemma", "backend": "llamacpp", "settings": {}},
            {"id": "nomic", "backend": "llamacpp", "settings": {"embeddings": "true"}}]}
        self.loads, self.unloads = [], []
        self.up = {"qwen"}
        by_prompt = {t["prompt"]: t for t in proving.TASKS}

        self.bodies = []

        def openai(body, stream=False, base=None):
            self.bodies.append(body)
            t = by_prompt[body["messages"][-1]["content"]]
            text = proving.sample_answer(t) if body["model"] == "qwen" else "no idea"
            return 200, {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                         "timings": {"predicted_n": 3, "predicted_per_second": 50.0 if body["model"] == "qwen" else 25.0}}
        runner = proving.Runner(routes._proving_ask, routes._ensure_loaded, routes._proving_release)
        for p in (mock.patch.object(proving, "PROVING_FILE", path),
                  mock.patch.object(routes, "REGISTRY", reg),
                  mock.patch.object(routes, "PROVING", runner),
                  mock.patch.object(routes.SLOTS, "_main", return_value="qwen"),
                  mock.patch.object(routes, "_model_status",
                                    side_effect=lambda m: {"value": "loaded" if m in self.up else "unloaded"}),
                  mock.patch.object(routes, "_load_and_wait", side_effect=lambda m, r: self.loads.append((m, r)) or self.up.add(m)),
                  mock.patch.object(routes, "_router_openai", side_effect=openai),
                  mock.patch.object(routes, "_model_base", return_value="http://127.0.0.1:8080"),
                  mock.patch.object(routes, "_unload_and_wait", side_effect=lambda m: self.unloads.append(m) or self.up.discard(m)),
                  mock.patch.object(routes, "_ARENA_LOADED", set())):
            p.start()
            self.addCleanup(p.stop)

    def run_all(self, body=None):
        code, out = self.routes.post_proving_run(self.routes.Req(body=body or {}))
        self.assertEqual(code, 200)
        self.routes.PROVING.join(5)
        return self.routes.get_proving(self.routes.Req())[1]

    def test_fresh_state_lists_tasks_and_models(self):
        _, out = self.routes.get_proving(self.routes.Req())
        self.assertFalse(out["status"]["running"])
        self.assertEqual(len(out["tasks"]), len(proving.TASKS))
        self.assertEqual(out["models"], ["gemma", "qwen"])
        self.assertEqual(out["scores"], {})
        self.assertEqual(out["driver"], "qwen")

    def test_schedule_round_trip(self):
        r = self.routes
        self.assertIsNone(r.get_proving(r.Req())[1]["schedule"])
        code, out = r.post_proving_schedule(r.Req(body={"at": "01:30", "models": ["qwen"]}))
        self.assertEqual(code, 200)
        self.assertEqual(out["schedule"]["at"], "01:30")
        got = r.get_proving(r.Req())[1]
        self.assertEqual(got["schedule"]["models"], ["qwen"])
        self.assertEqual(got["until"], proving.UNTIL)
        for bad in ({"at": "1:30"}, {"at": "01:30", "models": ["nomic"]}, {"at": "01:30", "models": "qwen"}):
            with self.assertRaises(r.ApiError) as e:
                r.post_proving_schedule(r.Req(body=bad))
            self.assertEqual(e.exception.status, 400)
        r.post_proving_schedule(r.Req(body={"at": None}))
        self.assertIsNone(r.get_proving(r.Req())[1]["schedule"])

    def test_last_activity_is_the_latest_use_of_any_model(self):
        from unittest import mock
        with mock.patch.object(self.routes.stats.TRACKER, "data",
                               {"models": {"a": {"last_used": 100}, "b": {"last_used": 250}, "c": {}}}):
            self.assertEqual(self.routes._last_activity(), 250)
        with mock.patch.object(self.routes.stats.TRACKER, "data", {}):
            self.assertEqual(self.routes._last_activity(), 0)

    def test_gpu_names_for_the_share_card(self):
        from unittest import mock
        with mock.patch.object(self.routes._GPU_TELEMETRY, "get",
                               return_value=[{"name": "NVIDIA GeForce RTX 5080"}, {"error": "x"}]):
            out = self.routes.get_proving(self.routes.Req())[1]
        self.assertEqual(out["gpus"], ["RTX 5080"])
        with mock.patch.object(self.routes._GPU_TELEMETRY, "get", side_effect=RuntimeError("no smi")):
            self.assertEqual(self.routes.get_proving(self.routes.Req())[1]["gpus"], [])

    def test_run_scores_every_chat_model_and_gives_the_verdict(self):
        out = self.run_all()
        self.assertEqual(out["scores"]["qwen"]["passed"], len(proving.TASKS))
        self.assertEqual(out["scores"]["gemma"]["passed"], 0)
        self.assertEqual(out["scores"]["gemma"]["tok_s"], 25.0)
        self.assertIn("at", out["scores"]["gemma"])
        self.assertEqual(out["verdicts"], {"gemma": "gemma trails qwen on 5 of 5 categories, 50% slower."})
        self.assertEqual(out["results"]["gemma"]["math-kv"]["answer"], "no idea")

    def test_thinking_models_get_room_to_finish(self):
        self.run_all({"models": ["qwen"]})
        self.assertEqual({b["max_tokens"] for b in self.bodies}, {proving.MAX_TOKENS})
        self.assertGreater(proving.MAX_TOKENS, 1024)

    def test_loads_what_it_needs_and_puts_it_back(self):
        self.run_all()
        self.assertEqual(self.loads, [("gemma", "worker")])
        self.assertEqual(self.unloads, ["gemma"], "the model the user had up stays up")
        self.assertEqual(self.up, {"qwen"})

    def test_picked_models_only_and_unknown_ones_refused(self):
        out = self.run_all({"models": ["qwen"]})
        self.assertEqual(list(out["scores"]), ["qwen"])
        with self.assertRaises(self.routes.ApiError) as cm:
            self.routes.post_proving_run(self.routes.Req(body={"models": ["nomic"]}))
        self.assertEqual(cm.exception.status, 400)

    def test_one_run_at_a_time_and_the_arena_waits(self):
        gate = threading.Event()
        real = self.routes._proving_ask
        from unittest import mock
        with mock.patch.object(self.routes.PROVING, "_ask", side_effect=lambda m, p: gate.wait(5) and real(m, p)):
            self.routes.post_proving_run(self.routes.Req(body={}))
            with self.assertRaises(self.routes.ApiError) as cm:
                self.routes.post_proving_run(self.routes.Req(body={}))
            self.assertEqual(cm.exception.status, 409)
            with self.assertRaises(self.routes.ApiError) as cm:
                self.routes.post_arena_answer(self.routes.Req(body={"model": "qwen", "prompt": "hi"}))
            self.assertEqual(cm.exception.status, 409)
            _, st = self.routes.post_proving_cancel(self.routes.Req())
            gate.set()
            self.routes.PROVING.join(5)
        self.assertTrue(self.routes.get_proving(self.routes.Req())[1]["status"]["cancelled"])


if __name__ == "__main__":
    unittest.main()


class Tonight(unittest.TestCase):
    """The overnight schedule: a time, a quiet check, once per night."""
    def setUp(self):
        from datetime import datetime
        self.dt = datetime
        fd, self.path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.remove(self.path)
        self.addCleanup(lambda: os.path.exists(self.path) and os.remove(self.path))
        self.started = []
        self.running = False
        self.last = 0.0
        outer = self

        class FakeRunner:
            def status(self):
                return {"running": outer.running}

            def start(self, models):
                outer.started.append(list(models))
                return True
        self.sched = proving.Scheduler(FakeRunner(), lambda: ["gemma", "qwen"], lambda: self.last,
                                       path=self.path)

    def at(self, h, m, day=2):
        return self.dt(2026, 11, day, h, m)

    def test_window_runs_from_the_set_time_to_six(self):
        n = proving.window_night
        self.assertEqual(n(self.at(1, 0), "01:00"), "2026-11-02")
        self.assertEqual(n(self.at(5, 59), "01:00"), "2026-11-02")
        self.assertIsNone(n(self.at(6, 0), "01:00"))
        self.assertIsNone(n(self.at(0, 59), "01:00"))
        # past midnight: 23:00 tonight belongs to the night that started today
        self.assertEqual(n(self.at(23, 30), "23:00"), "2026-11-02")
        self.assertEqual(n(self.at(2, 0, day=3), "23:00"), "2026-11-02")
        self.assertIsNone(n(self.at(22, 59), "23:00"))
        self.assertIsNone(n(self.at(12, 0), "bad"))

    def test_set_and_clear(self):
        self.assertEqual(proving.set_schedule(self.path, "01:30", ["qwen"])["at"], "01:30")
        self.assertEqual(proving.load(self.path)["schedule"], {"at": "01:30", "models": ["qwen"], "last_night": None})
        with self.assertRaises(ValueError):
            proving.set_schedule(self.path, "25:00", None)
        proving.set_schedule(self.path, None, None)
        self.assertIsNone(proving.load(self.path).get("schedule"))

    def test_starts_once_a_night_when_quiet(self):
        proving.set_schedule(self.path, "01:00", None)
        now = self.at(1, 5)
        self.last = now.timestamp() - 3600
        self.assertEqual(self.sched.tick(now), "started")
        self.assertEqual(self.started, [["gemma", "qwen"]])
        self.assertEqual(proving.load(self.path)["schedule"]["last_night"], "2026-11-02")
        self.assertIsNone(self.sched.tick(self.at(2, 0)))          # same night: done
        self.assertEqual(len(self.started), 1)
        self.assertEqual(self.sched.tick(self.at(1, 5, day=3)), "started")  # next night

    def test_waits_while_someone_is_using_the_router(self):
        proving.set_schedule(self.path, "01:00", ["qwen", "gone"])
        now = self.at(1, 5)
        self.last = now.timestamp() - 60
        self.assertEqual(self.sched.tick(now), "busy")
        self.assertEqual(self.started, [])
        self.last = now.timestamp() - proving.QUIET_SECS - 1
        self.assertEqual(self.sched.tick(now), "started")
        self.assertEqual(self.started, [["qwen"]])                   # models that are gone are skipped

    def test_nothing_outside_the_window_or_while_a_run_is_going(self):
        self.assertIsNone(self.sched.tick(self.at(1, 5)))            # no schedule
        proving.set_schedule(self.path, "01:00", None)
        self.assertIsNone(self.sched.tick(self.at(12, 0)))
        self.running = True
        self.assertIsNone(self.sched.tick(self.at(1, 5)))
        self.assertEqual(self.started, [])
