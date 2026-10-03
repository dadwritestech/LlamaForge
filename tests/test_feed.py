import unittest

import conftest_paths  # noqa: F401
import feed


def rel(tag, title, date="2026-10-02T10:00:00Z", extra=""):
    return {"tag_name": tag, "published_at": date,
            "html_url": f"https://github.com/ggml-org/llama.cpp/releases/tag/{tag}",
            "body": f"<details open>\n\n{title}\n\n* detail line\n{extra}\n</details>\n\n**macOS:** ..."}


class ModelSupportTitles(unittest.TestCase):
    def test_flags_real_model_support_commits(self):
        for t in ["model: support nimble decision model (#29844)",
                  "add GLM-5.3-Flash (GLM5-Next) support (#27773)",
                  "model : add Kimi-Linear (#12345)",
                  "models: add support for Falcon-H2 (#1)",
                  "llama : add support for Qwen4 (#2)",
                  "convert: add MiMo-V2.6 support (#3)",
                  "model : add support for HrmTextForCausalLM (DFM Mimir 1B) (#4)"]:
            self.assertTrue(feed.is_model_support(t), t)

    def test_ignores_kernels_fixes_and_features(self):
        for t in ["spec : add probabilistic sampling for simple draft and MTP (#27694)",
                  "metal : add tensor API flash attention kernel for F16 KV (#29570)",
                  "mimo : support dflash (convert + feature extraction) (#29650)",
                  "model : re-enable -sm tensor for qwen4exp (#28569)",
                  "hexagon: add q2_k and q3_k quant type support (#29717)",
                  "webgpu: add bfloat16 support for MUL_MAT (#29358)",
                  "model : support classifier_pooling for rerankers (#29627)",
                  "server : support typed content (vision/audio/video) input (#29556)",
                  "jinja : add support for dict builtin (#5)",
                  "llama : add `llama_prec_policy` + model-driven W4A4 path (#6)",
                  "server: Add support for binding to multiple addresses (#7)"]:
            self.assertFalse(feed.is_model_support(t), t)


class ParseRelease(unittest.TestCase):
    def test_takes_first_line_inside_details(self):
        r = feed.parse_release(rel("b11364", "model: support nimble decision model (#29844)"))
        self.assertEqual(r["build"], 11364)
        self.assertEqual(r["title"], "model: support nimble decision model")
        self.assertEqual(r["pr"], "https://github.com/ggml-org/llama.cpp/pull/29844")
        self.assertEqual(r["date"], "2026-10-02")

    def test_non_build_tags_are_skipped(self):
        self.assertIsNone(feed.parse_release(rel("master-abc", "x")))
        self.assertIsNone(feed.parse_release({"tag_name": "b1", "body": None}))


class EngineNews(unittest.TestCase):
    RELS = [rel("b11366", "ggml-quants : avoid invalid rounding (#29817)"),
            rel("b11364", "model: support nimble decision model (#29844)"),
            rel("b11279", "add GLM-5.3-Flash (GLM5-Next) support (#27773)", "2026-09-29T00:00:00Z")]

    def test_filters_and_marks_against_current_build(self):
        news = feed.engine_news(self.RELS, current_build=11300)
        self.assertEqual([n["build"] for n in news], [11364, 11279])
        self.assertEqual([n["have"] for n in news], [False, True])

    def test_unknown_build_leaves_have_unset(self):
        news = feed.engine_news(self.RELS, current_build=None)
        self.assertEqual([n["have"] for n in news], [None, None])


class VersionOutput(unittest.TestCase):
    def test_reads_build_number(self):
        self.assertEqual(feed.parse_version("load_backend: ...\nversion: 11368 (0e1f2a3)\nbuilt with MSVC"), 11368)
        self.assertIsNone(feed.parse_version("garbage"))

    def test_build_zero_means_unknown(self):
        self.assertIsNone(feed.parse_version("version: 0 (unknown)"))


class AppUpdate(unittest.TestCase):
    def test_newer_release_is_offered(self):
        u = feed.app_update("v0.10.1", {"tag_name": "v0.11.0", "html_url": "u", "name": "v0.11.0: feed"})
        self.assertTrue(u["available"])
        self.assertEqual(u["latest"], "v0.11.0")

    def test_same_or_older_is_not(self):
        self.assertFalse(feed.app_update("v0.10.1", {"tag_name": "v0.10.1"})["available"])
        self.assertFalse(feed.app_update("v0.10.1", {"tag_name": "v0.9.9"})["available"])

    def test_git_checkout_or_branch_install_never_offers(self):
        self.assertFalse(feed.app_update(None, {"tag_name": "v9.0.0"})["available"])
        self.assertFalse(feed.app_update("master", {"tag_name": "v9.0.0"})["available"])


class TrendingSearch(unittest.TestCase):
    def setUp(self):
        import hub
        self.hub, self.orig = hub, hub._get_json
        self.urls = []
        def fake(url, timeout=25):
            self.urls.append(url)
            return [
                {"id": "a/New-LLM-GGUF", "createdAt": "2026-10-01T00:00:00.000Z", "pipeline_tag": "text-generation", "likes": 5, "downloads": 9},
                {"id": "a/Old-LLM-GGUF", "createdAt": "2026-08-01T00:00:00.000Z", "pipeline_tag": "text-generation"},
                {"id": "a/Image-GGUF", "createdAt": "2026-10-01T00:00:00.000Z", "pipeline_tag": "text-to-image"},
                {"id": "a/Untagged-GGUF", "createdAt": "2026-09-30T00:00:00.000Z"},
            ]
        hub._get_json = fake

    def tearDown(self):
        self.hub._get_json = self.orig

    def test_trending_keeps_recent_text_models(self):
        res = self.hub.search(sort="trending", now="2026-10-03")
        self.assertIn("sort=trendingScore", self.urls[0])
        self.assertEqual([r["repo"] for r in res], ["a/New-LLM-GGUF", "a/Untagged-GGUF"])
        self.assertEqual(res[0]["created"], "2026-10-01")

    def test_other_sorts_unfiltered(self):
        self.assertEqual(len(self.hub.search(sort="downloads")), 4)


if __name__ == "__main__":
    unittest.main()
