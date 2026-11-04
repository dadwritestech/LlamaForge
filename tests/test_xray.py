import conftest_paths  # noqa: F401
import unittest

import footprint
import xray

M = 1024 * 1024


def moe(n_layer=4):
    t = [("token_embd.weight", 12, 500 * M), ("output_norm.weight", 0, 1 * M),
         ("output.weight", 14, 500 * M)]
    for i in range(n_layer):
        t += [(f"blk.{i}.attn_norm.weight", 0, 1 * M),
              (f"blk.{i}.attn_q.weight", 12, 100 * M),
              (f"blk.{i}.attn_k.weight", 12, 20 * M),
              (f"blk.{i}.ffn_up_exps.weight", 12, 1000 * M),
              (f"blk.{i}.ffn_down_exps.weight", 14, 1000 * M),
              (f"blk.{i}.ffn_up_shexp.weight", 14, 50 * M)]
    return {"arch": "qwen3moe", "kv": {"block_count": n_layer}, "tensors": t}


def dense(n_layer=3):
    t = [("token_embd.weight", 12, 300 * M)]
    for i in range(n_layer):
        t += [(f"blk.{i}.attn_q.weight", 12, 40 * M), (f"blk.{i}.ffn_up.weight", 12, 80 * M),
              (f"blk.{i}.ssm_in.weight", 8, 10 * M)]
    return {"arch": "llama", "kv": {"block_count": n_layer}, "tensors": t}


class Placer(unittest.TestCase):
    def test_all_layers_by_default_but_embedding_stays_in_ram(self):
        on_gpu, info = footprint.placer({}, moe())
        self.assertFalse(on_gpu("token_embd.weight"))
        self.assertTrue(on_gpu("blk.0.attn_q.weight"))
        self.assertTrue(on_gpu("output.weight"))
        self.assertEqual(info["n_layer"], 4)
        self.assertTrue(info["confident"])

    def test_ngl_offloads_the_last_blocks(self):
        on_gpu, _ = footprint.placer({"ngl": "2"}, moe())
        self.assertFalse(on_gpu("blk.1.attn_q.weight"))
        self.assertTrue(on_gpu("blk.2.attn_q.weight"))
        self.assertFalse(on_gpu("output.weight"))

    def test_cpu_moe_pulls_experts_back(self):
        on_gpu, _ = footprint.placer({"cpu-moe": "true"}, moe())
        self.assertFalse(on_gpu("blk.3.ffn_up_exps.weight"))
        self.assertTrue(on_gpu("blk.3.ffn_up_shexp.weight"))

    def test_predict_is_unchanged_by_the_refactor(self):
        p = footprint.predict({"ngl": "2", "ctx-size": "4096"}, moe())
        # blocks 2,3 (2 x 2171 MiB) on GPU; embedding + blocks 0,1 + output_norm + output in RAM
        self.assertEqual(p["weights_gpu_mib"], 2 * 2171)
        self.assertEqual(p["weights_ram_mib"], 500 + 1 + 500 + 2 * 2171)


class View(unittest.TestCase):
    def test_rows_run_embedding_blocks_then_output(self):
        v = xray.view(moe(), {})
        self.assertEqual([r["label"] for r in v["rows"]],
                         ["embedding", "blk 0", "blk 1", "blk 2", "blk 3", "output"])

    def test_each_block_is_split_into_kinds(self):
        row = xray.view(moe(), {})["rows"][1]
        self.assertEqual(row["parts"], {"attention": 120 * M, "experts": 2000 * M,
                                        "shared": 50 * M, "norm": 1 * M})
        self.assertEqual(row["gpu"], 2171 * M)
        self.assertEqual(row["ram"], 0)

    def test_dense_ffn_and_ssm_kinds(self):
        row = xray.view(dense(), {})["rows"][1]
        self.assertEqual(row["parts"], {"attention": 40 * M, "ffn": 80 * M, "ssm": 10 * M})

    def test_mtp_head_and_gemma_per_layer_inputs_have_names(self):
        self.assertEqual(xray.kind("blk.64.nextn.eh_proj.weight"), "mtp")
        self.assertEqual(xray.kind("blk.64.nextn.shared_head_norm.weight"), "mtp")
        self.assertEqual(xray.kind("blk.3.inp_gate.weight"), "per-layer")
        self.assertEqual(xray.kind("blk.3.proj.weight"), "per-layer")
        self.assertEqual(xray.kind("blk.3.layer_output_scale.weight"), "other")

    def test_totals_follow_the_settings(self):
        v = xray.view(moe(), {"n-cpu-moe": "1"})
        blk0 = v["rows"][1]
        self.assertEqual(blk0["ram"], 2000 * M)
        self.assertEqual(blk0["gpu"], 171 * M)
        self.assertEqual(v["gpu"] + v["ram"], v["total"])
        self.assertEqual(v["ram"], 500 * M + 2000 * M)

    def test_ram_parts_say_which_kinds_stay_in_ram(self):
        rows = xray.view(moe(), {"n-cpu-moe": "1"})["rows"]
        self.assertEqual(rows[1]["ram_parts"], {"experts": 2000 * M})
        self.assertEqual(rows[2]["ram_parts"], {})
        self.assertEqual(rows[0]["ram_parts"], {"embedding": 500 * M})

    def test_types_ranked_by_bytes_with_names(self):
        types = xray.view(moe(), {})["types"]
        self.assertEqual(types[0]["type"], "Q4_K")
        self.assertEqual(types[0]["bytes"], 500 * M + 4 * 1120 * M)
        self.assertEqual({t["type"] for t in types}, {"Q4_K", "Q6_K", "F32"})
        self.assertEqual(sum(t["bytes"] for t in types), xray.view(moe(), {})["total"])

    def test_row_carries_its_main_type(self):
        self.assertEqual(xray.view(moe(), {})["rows"][1]["type"], "Q4_K")

    def test_layer_count_and_gpu_layers_reported(self):
        v = xray.view(moe(), {"ngl": "3"})
        self.assertEqual(v["n_layer"], 4)
        self.assertEqual(v["ngl"], 3)
        self.assertEqual(xray.view(moe(), {})["ngl"], 5)   # all blocks + output

    def test_tied_output_shows_the_gpu_copy_of_the_embedding(self):
        lay = dense()                                     # no output.weight: tied
        self.assertEqual(xray.view(lay, {})["tied_copy"], 300 * M)
        self.assertEqual(xray.view(lay, {"ngl": "3"})["tied_copy"], 0)   # output stays on the CPU
        self.assertEqual(xray.view(moe(), {})["tied_copy"], 0)            # has its own output.weight

    def test_unreadable_layout_is_none(self):
        self.assertIsNone(xray.view(None, {}))
        self.assertIsNone(xray.view({"arch": "x", "kv": {}, "tensors": []}, {}))


class Route(unittest.TestCase):
    def setUp(self):
        from unittest import mock
        import routes
        self.routes = routes
        self.secs = {"*": {"ctx-size": "8192"},
                     "qmoe": {"model": "D:/m/q.gguf", "ngl": "3", "cpu-moe": "true"}}
        mock.patch.object(routes.config, "read_sections", side_effect=lambda *a: self.secs).start()
        self.lay = mock.patch.object(routes, "_el_layout", return_value=moe()).start()
        self.addCleanup(mock.patch.stopall)

    def get(self, **qs):
        return self.routes.get_model_xray(self.routes.Req(qs=qs))

    def test_uses_the_model_settings(self):
        st, v = self.get(model="qmoe")
        self.assertEqual(st, 200)
        self.assertEqual(v["model"], "qmoe")
        self.assertEqual(v["ngl"], 3)
        self.assertEqual(v["rows"][4]["ram"], 2000 * M)      # cpu-moe: blk 3's experts stay in RAM
        self.lay.assert_called_with("D:/m/q.gguf")

    def test_ngl_query_overrides_any_spelling_of_the_setting(self):
        self.secs["qmoe"]["n-gpu-layers"] = "1"
        st, v = self.get(model="qmoe", ngl="0")
        self.assertEqual(v["ngl"], 0)
        self.assertEqual(v["gpu"], 0)
        self.assertEqual(v["configured_ngl"], 1)

    def test_bad_ngl_and_unknown_model_are_errors(self):
        with self.assertRaises(self.routes.ApiError) as e:
            self.get(model="nope")
        self.assertEqual(e.exception.status, 404)
        with self.assertRaises(self.routes.ApiError) as e:
            self.get(model="qmoe", ngl="lots")
        self.assertEqual(e.exception.status, 400)

    def test_unreadable_header(self):
        self.lay.return_value = None
        with self.assertRaises(self.routes.ApiError) as e:
            self.get(model="qmoe")
        self.assertEqual(e.exception.status, 422)

    def test_registered(self):
        self.assertIs(self.routes.GET_ROUTES["/api/model/xray"], self.routes.get_model_xray)


if __name__ == "__main__":
    unittest.main()
