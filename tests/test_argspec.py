"""parse_help against the help layouts mainline and ik_llama.cpp actually print."""
import conftest_paths  # noqa: F401
import unittest

import argspec


def _one(help_text, key):
    return next(i for i in argspec.parse_help(help_text) if i["key"] == key)


class MainlineLayoutTest(unittest.TestCase):
    def test_enum_placeholder_in_its_own_column(self):
        it = _one("-rea,  --reasoning [on|off|auto]        Use reasoning/thinking in the chat\n",
                  "reasoning")
        self.assertEqual(it["type"], "enum")
        self.assertEqual(it["options"], ["on", "off", "auto"])
        self.assertEqual(it["desc"], "Use reasoning/thinking in the chat")

    def test_bool_with_both_polarities(self):
        it = _one("--mmproj-auto, --no-mmproj, --no-mmproj-auto\n"
                  "                                        whether to use a projector (default: enabled)\n",
                  "mmproj-auto")
        self.assertEqual(it["type"], "bool")
        self.assertEqual(it["flags"], ["--mmproj-auto", "--no-mmproj", "--no-mmproj-auto"])
        self.assertEqual(it["default"], "enabled")

    def test_bracketed_text_after_a_valued_flag_stays_in_the_description(self):
        it = _one("--log-file FNAME                        [deprecated|kept] log to file\n",
                  "log-file")
        self.assertEqual(it["type"], "path")
        self.assertEqual(it["desc"], "[deprecated|kept] log to file")


class IkLayoutTest(unittest.TestCase):
    def test_enum_placeholder_glued_to_the_description(self):
        """ik prints `--reasoning              [on|off|auto]Use reasoning...`:
        no space after the placeholder, so it landed in the description and
        the knob read as a bool, which turned reasoning=off into nothing."""
        it = _one("-rea,  --reasoning              [on|off|auto]Use reasoning/thinking in the chat "
                  "('on', 'off', or 'auto', default: 'auto' (detect from template))\n",
                  "reasoning")
        self.assertEqual(it["type"], "enum")
        self.assertEqual(it["options"], ["on", "off", "auto"])
        self.assertEqual(it["placeholder"], "[on|off|auto]")
        self.assertTrue(it["desc"].startswith("Use reasoning/thinking"))

    def test_plural_suffix_in_parentheses_is_two_spellings(self):
        """ik's `--embedding(s)` accepts --embedding and --embeddings."""
        items = argspec.parse_help("--embedding(s)           restrict to only support embedding "
                                   "use case (default: disabled)\n")
        self.assertEqual(len(items), 1)
        it = items[0]
        self.assertEqual(it["key"], "embedding")
        self.assertEqual(it["flags"], ["--embedding", "--embeddings"])
        self.assertEqual(it["aliases"], ["embedding", "embeddings"])
        self.assertEqual(it["type"], "bool")

    def test_enum_in_parentheses(self):
        it = _one("  -fa, --flash-attn (auto|on|off|0|1)\n"
                  "                                  set Flash Attention (default: on)\n", "flash-attn")
        self.assertEqual(it["options"], ["auto", "on", "off", "0", "1"])
        self.assertEqual(it["values"], ["auto", "on", "off", "0", "1"])
        self.assertEqual(it["default"], "on")


# ik's --spec-type: values on a "types:" line, payloads allowed, and example
# lines that start with a flag
IK_SPEC = """\
  --spec-type SPEC[:k=v,...]      canonical speculative stage entry; repeat for a supported two-stage chain.
                                  types: none, draft, dflash, mtp, ngram-cache, ngram-simple, ngram-map-k, ngram-map-k4v, ngram-mod, suffix
                                  examples: --spec-type mtp:n_max=1,p_min=0.0
                                            --model-draft draft.gguf --spec-type dflash:n_max=4,cross_ctx=512
                                  legacy --spec-stage, --draft-*, --spec-ngram-*, --suffix-* and -
  --spec-autotune                 automatically tune speculative params to maximize tokens/sec
"""
IK_TYPES = ["none", "draft", "dflash", "mtp", "ngram-cache", "ngram-simple", "ngram-map-k",
            "ngram-map-k4v", "ngram-mod", "suffix"]


class DocumentedValuesTest(unittest.TestCase):
    """`values`: what the build itself says an option takes (None when it doesn't
    say). A process slot checks a models.ini value against it before starting."""

    def test_a_types_line(self):
        it = _one(IK_SPEC, "spec-type")
        self.assertEqual(it["values"], IK_TYPES)
        self.assertEqual(it["type"], "str")        # mtp:n_max=1 is a value too: no select
        self.assertFalse(it["curated"])

    def test_example_lines_are_not_options(self):
        """They start with a flag but sit in the description column; they made a
        second spec-type (placeholder 'mtp:n_max=1,...') and a second model-draft."""
        keys = [i["key"] for i in argspec.parse_help(IK_SPEC)]
        self.assertEqual(keys, ["spec-type", "spec-autotune"])

    def test_a_flag_in_a_wrapped_description(self):
        items = argspec.parse_help(
            "-hf,   -hfr, --hf-repo <user>/<model>[:quant]\n"
            "                                        Hugging Face model repository; mmproj is also downloaded\n"
            "                                        --hf-repo (default: unused)\n")
        self.assertEqual([i["key"] for i in items], ["hf-repo"])
        self.assertEqual(items[0]["default"], "unused")

    def test_the_builds_own_list_beats_the_curated_one(self):
        it = _one("--spec-type none,draft-simple,draft-mtp,draft-dflash\n"
                  "                                        comma-separated list of types of speculative decoding\n",
                  "spec-type")
        self.assertEqual(it["options"], ["none", "draft-simple", "draft-mtp", "draft-dflash"])
        self.assertEqual(it["values"], it["options"])
        self.assertFalse(it["curated"])

    def test_curated_options_when_the_help_lists_none(self):
        it = _one("--spec-type TYPE                        type of speculative decoding\n", "spec-type")
        self.assertEqual(it["type"], "enum")
        self.assertIn("draft-mtp", it["options"])
        self.assertTrue(it["curated"])
        self.assertIsNone(it["values"])            # we listed them; this build didn't

    def test_an_allowed_values_line_replaces_the_curated_list(self):
        it = _one("-ctk,  --cache-type-k TYPE              KV cache data type for K\n"
                  "                                        allowed values: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1\n"
                  "                                        (default: f16)\n", "cache-type-k")
        want = ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"]
        self.assertEqual((it["type"], it["options"], it["values"]), ("enum", want, want))
        self.assertFalse(it["curated"])
        self.assertEqual(it["default"], "f16")

    def test_an_allowed_values_list_that_wraps_is_read_in_full(self):
        it = _one("-ctk,  --cache-type-k TYPE              KV cache data type for K\n"
                  "                                        allowed values: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1,\n"
                  "                                        q6_0, q6_1, q3_0, q3_1, q2_0, q2_1, kvarn2, kvarn3, kvarn4, kvarn5,\n"
                  "                                        kvarn6, kvarn8\n"
                  "                                        (default: f16)\n", "cache-type-k")
        want = ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1",
                "q6_0", "q6_1", "q3_0", "q3_1", "q2_0", "q2_1", "kvarn2", "kvarn3", "kvarn4", "kvarn5",
                "kvarn6", "kvarn8"]
        self.assertEqual((it["type"], it["values"]), ("enum", want))
        self.assertEqual(it["default"], "f16")

    def test_a_types_line_that_goes_on_is_prose(self):
        it = _one("         --override-kv KEY=TYPE:VALUE\n"
                  "                                  advanced option to override model metadata by key.\n"
                  "                                  types: int, float, bool, str. example: --override-kv a=bool:false\n",
                  "override-kv")
        self.assertIsNone(it["values"])

    def test_numbered_placeholders_are_not_choices(self):
        """`<dev1,dev2,..>` / `dev1,dev2` name the shape of a list; reading them as
        choices offered dev1/dev2/.. in a select and would turn CUDA1 away."""
        for line in ("-dev,  --device <dev1,dev2,..>          comma-separated list of devices\n",
                     "  -dev,   --device dev1,dev2      comma-separated list of devices\n"):
            it = _one(line, "device")
            self.assertEqual((it["type"], it["values"]), ("str", None), line)


if __name__ == "__main__":
    unittest.main()
