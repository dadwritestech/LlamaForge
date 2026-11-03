import conftest_paths  # noqa: F401
import unittest

import prtry


SHA = "0123456789abcdef0123456789abcdef01234567"


def api(**kw):
    d = {"number": 1234, "title": "CUDA: faster MMQ for Blackwell", "state": "open",
         "merged_at": None, "draft": False, "user": {"login": "someone"},
         "head": {"sha": SHA}, "html_url": "https://github.com/ggml-org/llama.cpp/pull/1234"}
    d.update(kw)
    return d


class ParsePr(unittest.TestCase):
    def test_number_hash_and_url(self):
        for s in ["1234", " #1234 ", "https://github.com/ggml-org/llama.cpp/pull/1234",
                  "github.com/ggml-org/llama.cpp/pull/1234/files",
                  "https://github.com/ggerganov/llama.cpp/pull/1234#issuecomment-9"]:
            self.assertEqual(prtry.parse_pr(s), 1234, s)

    def test_rejects_other_repos_and_junk(self):
        for s in ["", "abc", "0", "-3", "12345678", None, 12,
                  "https://github.com/someone/llama.cpp/pull/1234",
                  "https://github.com/ggml-org/whisper.cpp/pull/1234",
                  "https://evil.example/ggml-org/llama.cpp/pull/1234"]:
            with self.assertRaises(ValueError, msg=repr(s)):
                prtry.parse_pr(s)


class PrInfo(unittest.TestCase):
    def test_open_pr(self):
        i = prtry.pr_info(api())
        self.assertEqual(i, {"number": 1234, "title": "CUDA: faster MMQ for Blackwell",
                             "author": "someone", "sha": SHA, "state": "open", "draft": False,
                             "url": "https://github.com/ggml-org/llama.cpp/pull/1234"})

    def test_merged_and_closed(self):
        self.assertEqual(prtry.pr_info(api(state="closed", merged_at="2026-11-01T00:00:00Z"))["state"], "merged")
        self.assertEqual(prtry.pr_info(api(state="closed"))["state"], "closed")

    def test_bad_sha_is_refused(self):
        with self.assertRaises(ValueError):
            prtry.pr_info(api(head={"sha": "nope; rm -rf"}))
        with self.assertRaises(ValueError):
            prtry.pr_info({"message": "Not Found"})

    def test_title_is_trimmed_and_capped(self):
        self.assertEqual(len(prtry.pr_info(api(title="x" * 500))["title"]), 200)


class Plan(unittest.TestCase):
    def test_install_name(self):
        self.assertEqual(prtry.install_name(1234, SHA), "pr-1234-0123456")

    def test_commands_pin_the_sha_and_build_two_targets(self):
        cmds = prtry.commands(1234, SHA, "W", "W/build", {"GGML_CUDA": "ON"}, jobs=8)
        flat = [" ".join(c) for c in cmds]
        self.assertEqual(flat[0], "git init -q W")
        self.assertEqual(flat[1], "git -C W fetch -q --depth 1 "
                                  "https://github.com/ggml-org/llama.cpp pull/1234/head")
        self.assertEqual(flat[2], "git -C W checkout -q --detach FETCH_HEAD")
        self.assertTrue(flat[3].startswith("cmake -B W/build -S W -DCMAKE_BUILD_TYPE=Release"))
        self.assertIn("-DGGML_CUDA=ON", flat[3])
        self.assertIn("-DLLAMA_CURL=OFF", flat[3])
        self.assertEqual(flat[4], "cmake --build W/build --config Release --parallel 8 "
                                  "--target llama-server llama-bench")
        # every command is a list of plain strings (no shell)
        self.assertTrue(all(isinstance(a, str) for c in cmds for a in c))

    def test_head_must_match(self):
        self.assertEqual(prtry.check_head(SHA + "\n", SHA), "")
        self.assertIn("changed", prtry.check_head("f" * 40, SHA))

    def test_manifest(self):
        m = prtry.manifest(prtry.pr_info(api()), {"GGML_CUDA": "ON"}, now=5.0)
        self.assertEqual(m["tag"], "PR #1234")
        self.assertEqual(m["label"], "PR #1234: CUDA: faster MMQ for Blackwell")
        self.assertEqual(m["channel"], "pr")
        self.assertEqual(m["pr"], 1234)
        self.assertEqual(m["sha"], SHA)
        self.assertEqual(m["installed_at"], 5.0)
        self.assertTrue(prtry.is_pr({"channel": "pr"}))
        self.assertFalse(prtry.is_pr({"channel": "nightly"}))


if __name__ == "__main__":
    unittest.main()
