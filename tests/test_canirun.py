import conftest_paths  # noqa: F401
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import canirun

GIB = 1024 ** 3
KV = {"block_count": 4, "head_count": 8, "head_count_kv": 8, "embedding_length": 1024,
      "context_length": 32768}


def dense_layout(block_gib=1.0):
    t = [["token_embd.weight", 12, GIB // 2], ["output.weight", 12, GIB // 2]]
    t += [[f"blk.{i}.attn_q.weight", 12, int(block_gib * GIB)] for i in range(4)]
    return {"arch": "llama", "kv": dict(KV), "tensors": t}


def moe_layout(exp_gib=6.0):
    t = [["token_embd.weight", 12, GIB // 2], ["output.weight", 12, GIB // 2]]
    for i in range(4):
        t.append([f"blk.{i}.attn_q.weight", 12, GIB // 4])
        t.append([f"blk.{i}.ffn_up_exps.weight", 12, int(exp_gib * GIB)])
    return {"arch": "qwen3moe", "kv": dict(KV, expert_count=128), "tensors": t}


def model(slug="tiny-8b", name="Tiny 8B", layout=None, moe=False):
    lay = layout or dense_layout()
    return {"slug": slug, "name": name, "repo": "someone/Tiny-8B-GGUF", "moe": moe,
            "arch": lay["arch"], "kv": lay["kv"],
            "quants": [{"quant": "Q4_K_M", "file": "Tiny-8B-Q4_K_M.gguf",
                        "bytes": sum(t[2] for t in lay["tensors"]),
                        "tensors": canirun.compact(lay)["tensors"]}]}


class Compact(unittest.TestCase):
    def test_blocks_sum_to_experts_and_rest_and_keep_total_bytes(self):
        lay = moe_layout()
        lay["tensors"].append(["blk.0.ffn_down_exps.weight", 12, 100])
        lay["tensors"].append(["blk.0.attn_k.weight", 12, 50])
        c = canirun.compact(lay)
        names = {n: b for n, _, b in c["tensors"]}
        self.assertEqual(names["blk.0.ffn_up_exps.weight"], 6 * GIB + 100)
        self.assertEqual(names["blk.0.rest"], GIB // 4 + 50)
        self.assertIn("token_embd.weight", names)
        self.assertEqual(sum(names.values()), sum(t[2] for t in lay["tensors"]))


class Pick(unittest.TestCase):
    def test_one_file_per_rung_plain_names_first_shards_grouped(self):
        paths = ["M-Q8_0.gguf", "M-UD-Q4_K_M.gguf", "M-Q4_K_M.gguf", "M-UD-Q3_K_XL.gguf",
                 "M-BF16.gguf", "mmproj-F16.gguf", "MTP/mtp-M-Q8_0.gguf", "eagle3-M-Q8_0.gguf",
                 "Q6_K/M-Q6_K-00002-of-00002.gguf", "Q6_K/M-Q6_K-00001-of-00002.gguf"]
        got = canirun.pick_quants(paths)
        self.assertEqual([q["quant"] for q in got], ["Q8_0", "Q6_K", "Q4_K_M", "UD-Q3_K_XL"])
        self.assertEqual(got[1]["files"], ["Q6_K/M-Q6_K-00001-of-00002.gguf",
                                           "Q6_K/M-Q6_K-00002-of-00002.gguf"])
        self.assertEqual(got[2]["files"], ["M-Q4_K_M.gguf"])

    def test_mxfp4_only_repo(self):
        self.assertEqual([q["quant"] for q in canirun.pick_quants(["gpt-oss-20b-MXFP4.gguf"])],
                         ["MXFP4"])


class Answer(unittest.TestCase):
    def test_small_dense_fits_with_the_full_trained_context(self):
        a = canirun.answer(model(), model()["quants"][0], vram_gb=8)
        self.assertEqual(a["status"], "fits")
        self.assertEqual(a["max_ctx"], 32768)
        self.assertGreater(a["need_gb"], 4.5)

    def test_context_is_what_runs_out_first_on_a_tight_card(self):
        m = model(layout=dense_layout(block_gib=1.5))   # 6.5 GB on the GPU, 8 GB card
        a = canirun.answer(m, m["quants"][0], vram_gb=8)
        self.assertEqual(a["status"], "fits")
        self.assertEqual(a["max_ctx"], 16384)

    def test_big_moe_keeps_some_experts_in_ram(self):
        m = model(layout=moe_layout(), moe=True)
        a = canirun.answer(m, m["quants"][0], vram_gb=16)
        self.assertEqual(a["status"], "experts")
        self.assertTrue(1 <= a["cpu_moe"] <= 4)
        self.assertGreater(a["ram_gb"], 5)

    def test_big_dense_runs_partly_on_the_cpu(self):
        m = model(layout=dense_layout(block_gib=5))
        a = canirun.answer(m, m["quants"][0], vram_gb=12)
        self.assertEqual(a["status"], "partial")
        self.assertTrue(0 < a["gpu_layers"] < 4)

    def test_nothing_fits_a_card_smaller_than_the_overhead(self):
        a = canirun.answer(model(), model()["quants"][0], vram_gb=1)
        self.assertEqual(a["status"], "no")


class Flags(unittest.TestCase):
    def test_fits_offloads_every_layer_at_the_room_it_has(self):
        a = canirun.answer(model(), model()["quants"][0], vram_gb=8)
        self.assertEqual(canirun.flags(a), "-ngl 5 -c 32768")

    def test_moe_names_how_many_blocks_keep_experts_in_ram(self):
        m = model(layout=moe_layout(), moe=True)
        a = canirun.answer(m, m["quants"][0], vram_gb=16)
        self.assertEqual(canirun.flags(a), f"-ngl 5 --n-cpu-moe {a['cpu_moe']} -c 8192")

    def test_dense_partial_gives_the_gpu_layer_count(self):
        m = model(layout=dense_layout(block_gib=5))
        a = canirun.answer(m, m["quants"][0], vram_gb=12)
        self.assertEqual(canirun.flags(a), f"-ngl {a['gpu_layers']} -c 8192")

    def test_no_flags_when_nothing_fits(self):
        self.assertEqual(canirun.flags(canirun.answer(model(), model()["quants"][0], vram_gb=1)), "")


class Pages(unittest.TestCase):
    def setUp(self):
        big = model(slug="big-moe", name="Big <MoE>", layout=moe_layout(), moe=True)
        self.data = {"fetched": "2026-11-01", "models": [model(), big]}

    def test_pair_page_says_it_is_a_prediction_and_links_the_quant(self):
        gpu = canirun.GPU_BY_SLUG["rtx-5080"]
        p = canirun.pair_page(self.data["models"][0], gpu)
        self.assertEqual(p["slug"], "can-i-run-tiny-8b-on-rtx-5080")
        self.assertIn("Can I run Tiny 8B on an RTX 5080 (16 GB)?", p["title"])
        self.assertIn("prediction", p["html"].lower())
        self.assertIn("#discover?repo=someone%2FTiny-8B-GGUF&amp;file=Tiny-8B-Q4_K_M.gguf", p["html"])
        self.assertIn("huggingface.co/someone/Tiny-8B-GGUF", p["html"])
        self.assertNotRegex(p["html"], r"\d\s*tok/s")   # fit only, no speed guesses

    def test_pair_page_prints_the_flags_for_the_pick_and_every_quant(self):
        gpu = canirun.GPU_BY_SLUG["rtx-5080"]
        p = canirun.pair_page(self.data["models"][1], gpu)
        self.assertRegex(p["html"], r"<code>llama-server -hf someone/Tiny-8B-GGUF:Q4_K_M -ngl 5 --n-cpu-moe \d -c 8192 --no-mmproj</code>")
        self.assertIn("<th>Flags</th>", p["html"])
        self.assertIn("--n-cpu-moe", p["desc"])

    def test_names_are_escaped(self):
        gpu = canirun.GPU_BY_SLUG["rtx-4090"]
        p = canirun.pair_page(self.data["models"][1], gpu)
        self.assertNotIn("<MoE>", p["html"])
        self.assertIn("Big &lt;MoE&gt;", p["html"])

    def test_bad_slugs_are_dropped(self):
        bad = model(slug="../evil")
        self.assertEqual([m["slug"] for m in canirun.valid_models({"models": [bad, model()]})],
                         ["tiny-8b"])

    def test_index_has_a_cell_per_model_and_gpu(self):
        idx = canirun.index_page(self.data)
        self.assertEqual(idx["slug"], "can-i-run")
        for g in canirun.GPUS:
            self.assertIn(f"can-i-run-tiny-8b-on-{g['slug']}.html", idx["html"])

    def test_index_groups_cards_into_one_table_each(self):
        idx = canirun.index_page(self.data)["html"]
        groups = [g["group"] for g in canirun.GPUS]
        for name in dict.fromkeys(groups):
            self.assertIn(f"<h2>{name}</h2>", idx)
        self.assertEqual(idx.count("<table>"), len(set(groups)))
        self.assertEqual(groups, sorted(groups, key=list(dict.fromkeys(groups)).index))  # grouped in order

    def test_cards_include_laptops_and_intel_with_unique_slugs(self):
        slugs = [g["slug"] for g in canirun.GPUS]
        self.assertEqual(len(slugs), len(set(slugs)))
        self.assertEqual(canirun.GPU_BY_SLUG["rtx-4060-laptop"]["gb"], 8)
        self.assertEqual(canirun.GPU_BY_SLUG["arc-b580"]["gb"], 12)
        self.assertIn("Can I run Tiny 8B on an Intel Arc B580 (12 GB)?",
                      canirun.pair_page(model(), canirun.GPU_BY_SLUG["arc-b580"])["title"])

    def test_build_writes_every_pair(self):
        out = tempfile.mkdtemp()
        pages = canirun.build_pages(self.data)
        self.assertEqual(len(pages), 1 + 2 * len(canirun.GPUS))
        self.assertTrue(all(p["desc"] for p in pages))
        self.assertTrue(all(set(p["slug"]) <= set("abcdefghijklmnopqrstuvwxyz0123456789.-")
                            for p in pages))


if __name__ == "__main__":
    unittest.main()
