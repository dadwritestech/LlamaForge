"""backend/arena.py: the blind local arena. Two of your models answer one
prompt unlabeled, you pick, and an Elo board kept on this PC ranks them."""
import conftest_paths  # noqa: F401
import json, os, random, tempfile, unittest

import arena


class Elo(unittest.TestCase):
    def test_equal_ratings_move_by_half_k(self):
        a, b = arena.elo(1000, 1000, 1.0)
        self.assertAlmostEqual(a, 1016)
        self.assertAlmostEqual(b, 984)

    def test_tie_between_equals_changes_nothing(self):
        self.assertEqual(arena.elo(1000, 1000, 0.5), (1000, 1000))

    def test_upset_moves_more_than_expected_win(self):
        up, _ = arena.elo(900, 1100, 1.0)
        fav, _ = arena.elo(1100, 900, 1.0)
        self.assertGreater(up - 900, fav - 1100)

    def test_points_are_conserved(self):
        a, b = arena.elo(1234, 987, 0.0)
        self.assertAlmostEqual(a + b, 1234 + 987)


class Board(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.remove(self.path)
        self.addCleanup(lambda: os.path.exists(self.path) and os.remove(self.path))

    def test_empty_when_missing_or_corrupt(self):
        self.assertEqual(arena.load(self.path), {"models": {}, "games": 0})
        with open(self.path, "w") as f:
            f.write("{nope")
        self.assertEqual(arena.load(self.path), {"models": {}, "games": 0})

    def test_vote_updates_both_and_persists(self):
        b = arena.vote(self.path, "qwen", "gemma", "a")
        self.assertEqual(b["games"], 1)
        q, g = b["models"]["qwen"], b["models"]["gemma"]
        self.assertGreater(q["elo"], g["elo"])
        self.assertEqual((q["wins"], q["losses"], q["ties"]), (1, 0, 0))
        self.assertEqual((g["wins"], g["losses"], g["ties"]), (0, 1, 0))
        self.assertEqual(arena.load(self.path), b)

    def test_b_and_tie(self):
        arena.vote(self.path, "qwen", "gemma", "b")
        b = arena.vote(self.path, "qwen", "gemma", "tie")
        self.assertEqual(b["models"]["gemma"]["wins"], 1)
        self.assertEqual(b["models"]["qwen"]["ties"], 1)
        self.assertEqual(b["games"], 2)

    def test_bad_votes_are_refused(self):
        for a, bb, w in (("q", "q", "a"), ("", "g", "a"), ("q", "g", "x"), ("q", None, "a")):
            with self.assertRaises(ValueError):
                arena.vote(self.path, a, bb, w)
        self.assertFalse(os.path.exists(self.path))

    def test_leaderboard_is_sorted_by_elo(self):
        arena.vote(self.path, "a", "b", "a")
        arena.vote(self.path, "c", "a", "a")
        rows = arena.leaderboard(arena.load(self.path))
        self.assertEqual([r["model"] for r in rows], ["c", "a", "b"])
        self.assertEqual(rows[0]["games"], 1)
        self.assertIsInstance(rows[0]["elo"], int)

    def test_markdown_table(self):
        arena.vote(self.path, "qwen", "gemma", "a")
        md = arena.markdown(arena.load(self.path))
        self.assertIn("| 1 | qwen |", md)
        self.assertIn("github.com/dadwritestech/LlamaForge", md)


class Answer(unittest.TestCase):
    def test_request_body_is_plain_and_capped(self):
        body = arena.request_body("qwen", "  hi  ")
        self.assertEqual(body["model"], "qwen")
        self.assertEqual(body["messages"], [{"role": "user", "content": "hi"}])
        self.assertFalse(body["stream"])
        self.assertLessEqual(body["max_tokens"], arena.MAX_TOKENS)

    def test_request_body_refuses_junk(self):
        for p in ("", "   ", None, 5, "x" * (arena.MAX_PROMPT + 1)):
            with self.assertRaises(ValueError):
                arena.request_body("qwen", p)

    def test_reply_reads_content_reasoning_and_speed(self):
        out = arena.reply({"choices": [{"message": {"content": "hello", "reasoning_content": "hmm"},
                                        "finish_reason": "stop"}],
                           "timings": {"predicted_n": 40, "predicted_per_second": 80.25}})
        self.assertEqual(out, {"text": "hello", "thinking": "hmm", "tokens": 40,
                               "tok_s": 80.2, "cut": False})

    def test_reply_marks_cut_answers_and_survives_missing_fields(self):
        out = arena.reply({"choices": [{"message": {"content": "x"}, "finish_reason": "length"}]})
        self.assertTrue(out["cut"])
        self.assertIsNone(out["tok_s"])
        self.assertEqual(arena.reply({})["text"], "")

    def test_pick_two_is_random_and_distinct(self):
        rng = random.Random(1)
        for _ in range(20):
            a, b = arena.pick_two(["x", "y", "z"], rng)
            self.assertNotEqual(a, b)
        with self.assertRaises(ValueError):
            arena.pick_two(["only"], rng)


class Routes(unittest.TestCase):
    """/api/arena*: the panel side, with the router mocked."""
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
            {"id": "nomic", "backend": "llamacpp", "settings": {"embeddings": "true"}},
            {"id": "vl", "backend": "vllm", "settings": {}}]}
        self.calls, self.loads, self.unloads = [], [], []
        self.up = {"qwen"}
        reply = {"choices": [{"message": {"content": "<b>hi</b>"}, "finish_reason": "stop"}],
                 "timings": {"predicted_n": 3, "predicted_per_second": 50.0}}
        for p in (mock.patch.object(arena, "ARENA_FILE", path),
                  mock.patch.object(routes, "REGISTRY", reg),
                  mock.patch.object(routes, "_model_status",
                                    side_effect=lambda m: {"value": "loaded" if m in self.up else "unloaded"}),
                  mock.patch.object(routes, "_load_and_wait", side_effect=lambda m, r: self.loads.append((m, r)) or self.up.add(m)),
                  mock.patch.object(routes, "_router_openai",
                                    side_effect=lambda body, stream=False, base=None: self.calls.append(body) or (200, reply)),
                  mock.patch.object(routes, "_model_base", return_value="http://127.0.0.1:8080"),
                  mock.patch.object(routes, "_unload_and_wait", side_effect=lambda m: self.unloads.append(m) or self.up.discard(m)),
                  mock.patch.object(routes, "_ARENA_LOADED", set())):
            p.start()
            self.addCleanup(p.stop)

    def test_only_llama_chat_models_take_part(self):
        _, out = self.routes.get_arena(self.routes.Req())
        self.assertEqual(out["models"], ["gemma", "qwen"])
        self.assertEqual(out["board"], [])

    def test_board_names_the_gpus_for_the_share_card(self):
        from unittest import mock
        tele = mock.Mock()
        tele.get.return_value = [{"name": "NVIDIA GeForce RTX 5080", "total": 16303},
                                 {"error": "nvidia-smi failed"}]
        with mock.patch.object(self.routes, "_GPU_TELEMETRY", tele):
            _, out = self.routes.get_arena(self.routes.Req())
        self.assertEqual(out["gpus"], ["RTX 5080"])

    def test_answer_loads_when_needed_and_returns_plain_text(self):
        _, out = self.routes.post_arena_answer(self.routes.Req(body={"model": "qwen", "prompt": "hi"}))
        self.assertEqual(self.loads, [])
        self.assertEqual(out["text"], "<b>hi</b>")        # the UI escapes it
        self.assertEqual(out["tok_s"], 50.0)
        self.routes.post_arena_answer(self.routes.Req(body={"model": "gemma", "prompt": "hi"}))
        self.assertEqual([m for m, _ in self.loads], ["gemma"])
        self.assertEqual(self.calls[0]["messages"], [{"role": "user", "content": "hi"}])

    def test_next_round_unloads_only_what_the_arena_loaded(self):
        self.up = {"qwen"}                        # the user's own model, up before the arena
        self.routes._ARENA_LOADED.clear()
        self.routes.REGISTRY.state.return_value["models"].append(
            {"id": "phi", "backend": "llamacpp", "settings": {}})
        ask = lambda m: self.routes.post_arena_answer(self.routes.Req(body={"model": m, "prompt": "hi"}))
        ask("qwen")
        ask("gemma")                              # loads gemma, keeps qwen
        self.assertEqual(self.unloads, [])
        ask("phi")                                # gemma was the arena's: it goes
        self.assertEqual(self.unloads, ["gemma"])
        self.assertIn("qwen", self.up)

    def test_answer_refuses_unknown_models_and_empty_prompts(self):
        from routes import ApiError
        for body in ({"model": "nomic", "prompt": "x"}, {"model": "vl", "prompt": "x"},
                     {"model": "qwen", "prompt": ""}):
            with self.assertRaises(ApiError):
                self.routes.post_arena_answer(self.routes.Req(body=body))
        self.assertEqual(self.calls, [])

    def test_vote_pair_and_reset(self):
        _, p = self.routes.post_arena_pair(self.routes.Req())
        self.assertEqual({p["a"], p["b"]}, {"qwen", "gemma"})
        _, out = self.routes.post_arena_vote(self.routes.Req(body={"a": "qwen", "b": "gemma", "winner": "a"}))
        self.assertEqual(out["board"][0]["model"], "qwen")
        self.assertIn("| 1 | qwen |", out["markdown"])
        _, out = self.routes.post_arena_reset(self.routes.Req())
        self.assertEqual(out["games"], 0)

    def test_routes_are_registered(self):
        self.assertIn("/api/arena", self.routes.GET_ROUTES)
        for p in ("pair", "answer", "vote", "reset"):
            self.assertIn("/api/arena/" + p, self.routes.POST_ROUTES)


if __name__ == "__main__":
    unittest.main()
