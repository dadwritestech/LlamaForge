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
        mock.patch.object(routes, "total_vram_mib", return_value=8 * 1024).start()
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

    def test_experts_flags_and_fit_from_the_model_settings(self):
        st, v = self.get(model="qmoe")
        self.assertTrue(v["experts"])
        self.assertEqual((v["ncmoe"], v["configured_ncmoe"]), (4, 4))    # cpu-moe = every block
        self.assertEqual(v["flags"], "-ngl 3 --n-cpu-moe 4 -c 8192")
        self.assertEqual(v["vram"], 8 * G)
        s = xray.fit_settings({**self.secs["*"], **self.secs["qmoe"]}, moe())[0]
        self.assertEqual(v["best"], xray.placement(moe(), xray.usable_mib(8 * G), s))
        cur = footprint.predict(xray.with_ncmoe(xray.with_ngl(s, 3), 4), moe())
        self.assertEqual(v["fit"], {"ctx": 8192, "ctk": "f16", "ctv": "f16", "trained_ctx": 0,
                                    "usable": xray.usable_mib(8 * G) * M,
                                    "need": cur["need_mib"] * M, "kv": cur["kv_gpu_mib"] * M})

    def test_ctx_and_kv_queries_change_the_fit(self):
        self.lay.return_value = heavy()
        st, v = self.get(model="qmoe", ngl="5", ncmoe="0", ctx="32768", kv="q8_0")
        self.assertEqual((v["fit"]["ctx"], v["fit"]["ctk"], v["fit"]["ctv"]), (32768, "q8_0", "q8_0"))
        self.assertEqual(v["fit"]["trained_ctx"], 32768)
        self.assertEqual(v["flags"], "-ngl 5 -c 32768 -ctk q8_0 -ctv q8_0")
        s = xray.fit_settings({}, heavy(), ctx=32768, kv="q8_0")[0]
        self.assertEqual(v["best"], xray.placement(heavy(), xray.usable_mib(8 * G), s))
        st, f16 = self.get(model="qmoe", ngl="5", ncmoe="0", ctx="32768")
        self.assertGreater(f16["fit"]["kv"], v["fit"]["kv"])
        for bad in ({"ctx": "lots"}, {"ctx": "12"}, {"kv": "q9_9"}, {"kv": "f16;rm"}):
            with self.assertRaises(self.routes.ApiError) as e:
                self.get(model="qmoe", **bad)
            self.assertEqual(e.exception.status, 400, bad)

    def test_ncmoe_query_replaces_cpu_moe(self):
        st, v = self.get(model="qmoe", ngl="5", ncmoe="1")
        self.assertEqual((v["ngl"], v["ncmoe"], v["configured_ncmoe"]), (5, 1, 4))
        self.assertEqual(v["flags"], "-ngl 5 --n-cpu-moe 1 -c 8192")
        self.assertEqual(v["rows"][1]["ram"], 2000 * M)     # only blk 0's experts in RAM
        self.assertEqual(v["rows"][2]["ram"], 0)
        with self.assertRaises(self.routes.ApiError) as e:
            self.get(model="qmoe", ncmoe="x")
        self.assertEqual(e.exception.status, 400)

    def test_dense_model_and_no_gpu(self):
        self.lay.return_value = dense()
        self.routes.total_vram_mib.return_value = 0
        st, v = self.get(model="qmoe")
        self.assertFalse(v["experts"])
        self.assertEqual(v["ncmoe"], 0)
        self.assertEqual(v["flags"], "-ngl 3 -c 8192")
        self.assertEqual(v["vram"], 0)
        self.assertIsNone(v["best"])
        self.assertEqual(v["fit"]["usable"], 0)

    def apply(self, **body):
        return self.routes.post_model_xray_apply(self.routes.Req(body=body))

    def test_apply_writes_the_flags_over_any_spelling(self):
        from unittest import mock
        self.secs["qmoe"]["gpu-layers"] = "9"
        with mock.patch.object(self.routes, "_apply_knobs_and_reload", return_value=True) as w:
            st, out = self.apply(model="qmoe", ngl=5, ncmoe=2)
        self.assertEqual(st, 200)
        w.assert_called_once_with("qmoe", {"ngl": None, "gpu-layers": None, "cpu-moe": None,
                                           "n-gpu-layers": "5", "n-cpu-moe": "2"})
        self.assertEqual(out, {"ok": True, "flags": "-ngl 5 --n-cpu-moe 2", "was_running": True})

    def test_apply_zero_experts_drops_the_flag(self):
        from unittest import mock
        with mock.patch.object(self.routes, "_apply_knobs_and_reload", return_value=False) as w:
            self.apply(model="qmoe", ngl="5", ncmoe="0")
        w.assert_called_once_with("qmoe", {"ngl": None, "cpu-moe": None,
                                           "n-gpu-layers": "5", "n-cpu-moe": None})

    def test_apply_writes_context_and_cache_type_when_given(self):
        from unittest import mock
        self.secs["qmoe"].update({"c": "4096", "ctk": "q4_0"})
        with mock.patch.object(self.routes, "_apply_knobs_and_reload", return_value=False) as w:
            st, out = self.apply(model="qmoe", ngl=5, ncmoe=2, ctx=16384, kv="q8_0")
        w.assert_called_once_with("qmoe", {"ngl": None, "cpu-moe": None, "c": None, "ctk": None,
                                           "n-gpu-layers": "5", "n-cpu-moe": "2", "ctx-size": "16384",
                                           "cache-type-k": "q8_0", "cache-type-v": "q8_0"})
        self.assertEqual(out["flags"], "-ngl 5 --n-cpu-moe 2 -c 16384 -ctk q8_0 -ctv q8_0")
        with mock.patch.object(self.routes, "_apply_knobs_and_reload", return_value=False) as w:
            self.apply(model="qmoe", ngl=5, ctx="8192", kv="f16")     # f16 is the default: no line
        self.assertEqual(w.call_args[0][1]["cache-type-k"], None)
        self.assertEqual(w.call_args[0][1]["ctx-size"], "8192")

    def test_apply_checks_its_input(self):
        from unittest import mock
        with mock.patch.object(self.routes, "_apply_knobs_and_reload") as w:
            for body, code in (({"model": "nope", "ngl": 1}, 404), ({"model": "*", "ngl": 1}, 404),
                               ({"model": "qmoe", "ngl": "lots"}, 400), ({"model": "qmoe"}, 400),
                               ({"model": "qmoe", "ngl": 1, "ncmoe": -1}, 400),
                               ({"model": "qmoe", "ngl": True}, 400),
                               ({"model": "qmoe", "ngl": 1, "ctx": "lots"}, 400),
                               ({"model": "qmoe", "ngl": 1, "ctx": True}, 400),
                               ({"model": "qmoe", "ngl": 1, "kv": "q9_9"}, 400)):
                with self.assertRaises(self.routes.ApiError) as e:
                    self.apply(**body)
                self.assertEqual(e.exception.status, code, body)
            w.assert_not_called()

    def test_apply_registered(self):
        self.assertIs(self.routes.POST_ROUTES["/api/model/xray/apply"], self.routes.post_model_xray_apply)

    def test_registered(self):
        self.assertIs(self.routes.GET_ROUTES["/api/model/xray"], self.routes.get_model_xray)


class Remote(unittest.TestCase):
    """A quant that isn't on this disk: its header over HTTP Range, never the weights."""

    def setUp(self):
        import os
        import shutil
        import tempfile
        from unittest import mock
        from test_gguf_layout import kv_str, kv_u32, write_gguf
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.files = {}
        specs = {"q6.gguf": ["blk.0.attn_q.weight", "blk.1.attn_q.weight"],
                 "big-00001-of-00002.gguf": ["blk.0.attn_q.weight"],
                 "big-00002-of-00002.gguf": ["output.weight"]}
        for name, tensors in specs.items():
            path = os.path.join(self.dir, name)
            write_gguf(path, [kv_str("general.architecture", "llama"), kv_u32("llama.block_count", 2)],
                       [(t, 14, 5000) for t in tensors])
            with open(path, "rb") as f:
                self.files[name] = f.read()
        self.local = os.path.join(self.dir, "q6.gguf")
        self.calls = []
        mock.patch.object(xray, "_get", side_effect=self.fake).start()
        mock.patch.object(xray, "FIRST", 64).start()
        mock.patch.dict(xray._REMOTE, clear=True).start()
        self.addCleanup(mock.patch.stopall)

    def fake(self, url, headers):
        self.calls.append((url, headers))
        raw = self.files[url.rsplit("/", 1)[-1]]
        end = min(int(headers["Range"].split("-")[1]), len(raw) - 1)
        return raw[:end + 1], {"Content-Range": f"bytes 0-{end}/{len(raw)}"}

    def test_widens_the_range_until_the_header_fits(self):
        import gguf
        self.assertEqual(xray.remote_layout("o/r", "q6.gguf"), gguf.layout(self.local))
        self.assertGreater(len(self.calls), 1)
        self.assertEqual(self.calls[0], ("https://huggingface.co/o/r/resolve/main/q6.gguf",
                                         {"Range": "bytes=0-63"}))

    def test_cached_per_file(self):
        xray.remote_layout("o/r", "q6.gguf")
        n = len(self.calls)
        xray.remote_layout("o/r", "q6.gguf")
        self.assertEqual(len(self.calls), n)

    def test_shards_are_merged(self):
        lay = xray.remote_layout("o/r", "big-00001-of-00002.gguf", shards=2)
        self.assertEqual([t[0] for t in lay["tensors"]], ["blk.0.attn_q.weight", "output.weight"])

    def test_unreadable_is_none_and_not_cached(self):
        good = self.files["q6.gguf"]
        self.files["q6.gguf"] = b"nope" * 10
        self.assertIsNone(xray.remote_layout("o/r", "q6.gguf"))
        self.files["q6.gguf"] = good
        self.assertIsNotNone(xray.remote_layout("o/r", "q6.gguf"))

    def test_listing_drops_the_imatrix_and_is_cached(self):
        from unittest import mock
        listing = {"files": [{"path": "imatrix_unsloth.gguf", "size": 1, "shards": 1},
                             {"path": "m-Q4_K_M.gguf", "size": 9, "shards": 1}]}
        with mock.patch.object(xray.hub, "files", return_value=listing) as f,                 mock.patch.dict(xray._LISTS, clear=True):
            self.assertEqual([q["path"] for q in xray.quants("o/r")], ["m-Q4_K_M.gguf"])
            xray.quants("o/r")
            self.assertEqual(f.call_count, 1)

    def test_network_error_is_none(self):
        xray._get.side_effect = OSError("offline")
        self.assertIsNone(xray.remote_layout("o/r", "q6.gguf"))


class Compare(unittest.TestCase):
    def setUp(self):
        from unittest import mock
        import routes
        self.routes = routes
        self.secs = {"*": {}, "q": {"model": "D:/dl/unsloth--Gem-GGUF/Gem-Q4_K_M.gguf", "ngl": "3"},
                     "local": {"model": "D:/m/own.gguf"}}
        mock.patch.object(routes.config, "read_sections", side_effect=lambda *a: self.secs).start()
        mock.patch.object(routes, "_el_layout", return_value=moe()).start()
        self.quants = mock.patch.object(xray, "quants", return_value=[
            {"path": "Gem-Q4_K_M.gguf", "size": 9 * M, "shards": 1},
            {"path": "Gem-Q6_K.gguf", "size": 12 * M, "shards": 1},
            {"path": "Gem-Q8_0-00001-of-00002.gguf", "size": 20 * M, "shards": 2}]).start()
        self.remote = mock.patch.object(xray, "remote_layout", return_value=dense()).start()
        self.addCleanup(mock.patch.stopall)

    def get(self, route, **qs):
        return route(self.routes.Req(qs=qs))

    def test_lists_the_repo_quants_and_marks_this_file(self):
        st, v = self.get(self.routes.get_model_xray_quants, model="q")
        self.assertEqual(v["repo"], "unsloth/Gem-GGUF")
        self.assertEqual(v["current"], "Gem-Q4_K_M.gguf")
        self.assertEqual([f["path"] for f in v["files"]][1], "Gem-Q6_K.gguf")
        self.quants.assert_called_with("unsloth/Gem-GGUF")

    def test_a_file_with_no_repo_says_so(self):
        st, v = self.get(self.routes.get_model_xray_quants, model="local")
        self.assertEqual(v["files"], [])
        self.assertIn("Hugging Face", v["reason"])

    def test_listing_offline_says_so(self):
        self.quants.side_effect = OSError("offline")
        st, v = self.get(self.routes.get_model_xray_quants, model="q")
        self.assertEqual(v["files"], [])
        self.assertIn("reach", v["reason"])

    def test_vs_draws_the_other_quant_under_the_same_settings(self):
        st, v = self.get(self.routes.get_model_xray, model="q", vs="Gem-Q8_0-00001-of-00002.gguf", ngl="1")
        self.remote.assert_called_with("unsloth/Gem-GGUF", "Gem-Q8_0-00001-of-00002.gguf", shards=2)
        self.assertEqual(v["vs"]["file"], "Gem-Q8_0-00001-of-00002.gguf")
        self.assertEqual(v["vs"]["ngl"], 1)
        self.assertEqual(v["vs"]["arch"], "llama")
        self.assertEqual(v["file"], "Gem-Q4_K_M.gguf")
        self.assertEqual(v["ngl"], 1)

    def test_vs_must_be_a_file_of_that_repo(self):
        for bad in ("../x.gguf", "Other.gguf"):
            with self.assertRaises(self.routes.ApiError) as e:
                self.get(self.routes.get_model_xray, model="q", vs=bad)
            self.assertEqual(e.exception.status, 400)
        with self.assertRaises(self.routes.ApiError) as e:
            self.get(self.routes.get_model_xray, model="local", vs="Gem-Q6_K.gguf")
        self.assertEqual(e.exception.status, 400)

    def test_unreadable_remote_header(self):
        self.remote.return_value = None
        with self.assertRaises(self.routes.ApiError) as e:
            self.get(self.routes.get_model_xray, model="q", vs="Gem-Q6_K.gguf")
        self.assertEqual(e.exception.status, 502)

    def test_registered(self):
        self.assertIs(self.routes.GET_ROUTES["/api/model/xray/quants"], self.routes.get_model_xray_quants)


G = 1024 * M


def heavy(n_layer=4):
    """moe() with a real KV cache: 64 KiB a token at f16, 512 MiB at 8k, 2 GiB at 32k."""
    lay = moe(n_layer)
    lay["kv"] = {"block_count": n_layer, "context_length": 32768, "head_count": 16,
                 "head_count_kv": 16, "key_length": 256, "value_length": 256,
                 "embedding_length": 4096}
    return lay


def need(lay, s):
    return footprint.predict(s, lay)["need_mib"]


def canirun():
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
    import canirun as c
    return c


class Fit(unittest.TestCase):
    """Fit my VRAM: the whole need (weights, the KV cache at the chosen context
    and cache type, buffers) against the card less what the desktop holds, the
    same rule the "Can I run it?" pages judge every card by."""

    def test_usable_keeps_the_desktop_share_back(self):
        self.assertEqual(xray.usable_mib(12 * G), 12 * 1024 - xray.DESKTOP_MIB)
        self.assertEqual(xray.usable_mib(0), 0)
        self.assertEqual(xray.usable_mib(256 * M), 0)

    def test_same_constants_as_the_canirun_pages(self):
        c = canirun()
        self.assertEqual(xray.DESKTOP_MIB, c.DESKTOP_MIB)
        self.assertEqual(xray.DEFAULT_CTX, c.ANSWER_CTX)

    def test_fit_settings_context_and_cache_type(self):
        s, f = xray.fit_settings({}, heavy())
        self.assertEqual(f, {"ctx": 8192, "ctk": "f16", "ctv": "f16", "trained_ctx": 32768})
        self.assertEqual(s, {"ctx-size": "8192", "cache-type-k": "f16", "cache-type-v": "f16"})
        short = heavy()
        short["kv"]["context_length"] = 4096           # never past what the model was trained on
        self.assertEqual(xray.fit_settings({}, short)[1]["ctx"], 4096)
        self.assertEqual(xray.fit_settings({}, moe())[1]["ctx"], 8192)   # no trained length known
        own = {"c": "16384", "ctk": "q8_0", "ctv": "q4_0", "ngl": "3"}
        s, f = xray.fit_settings(own, heavy())        # the model's own, any spelling
        self.assertEqual((f["ctx"], f["ctk"], f["ctv"]), (16384, "q8_0", "q4_0"))
        self.assertEqual(s, {"ngl": "3", "ctx-size": "16384", "cache-type-k": "q8_0",
                             "cache-type-v": "q4_0"})
        s, f = xray.fit_settings(own, heavy(), ctx=32768, kv="f16")   # what you picked wins
        self.assertEqual((f["ctx"], f["ctk"], f["ctv"]), (32768, "f16", "f16"))
        self.assertEqual(xray.fit_settings({"ctx-size": "0"}, heavy())[1]["ctx"], 8192)
        self.assertEqual(xray.fit_settings({"ctk": "nope"}, heavy())[1]["ctk"], "f16")

    def test_most_layers_whose_whole_need_fits(self):
        lay, s = heavy(), xray.fit_settings({}, heavy())[0]
        for room in (1500, 3000, 5000, 7000, 100000):
            n = xray.fit_ngl(lay, s, room)
            if n:
                self.assertLessEqual(need(lay, xray.with_ngl(s, n)), room, room)
            if n < 5:
                self.assertGreater(need(lay, xray.with_ngl(s, n + 1)), room, room)
        self.assertEqual(xray.fit_ngl(lay, s, 100000), 5)
        self.assertEqual(xray.fit_ngl(lay, s, 0), 0)

    def test_fewest_experts_on_cpu_whose_whole_need_fits(self):
        lay, s = heavy(), xray.fit_settings({}, heavy())[0]
        self.assertTrue(xray.has_experts(lay))
        self.assertFalse(xray.has_experts(dense()))
        self.assertIsNone(xray.fit_ncmoe(dense(), s, 100000))
        self.assertEqual(xray.fit_ncmoe(lay, s, 100000), 0)
        full = xray.with_ngl(s, 5)
        for room in (4500, 6000, 7000, 8000):
            k = xray.fit_ncmoe(lay, s, room)
            self.assertLessEqual(need(lay, xray.with_ncmoe(full, k)), room, room)
            for j in range(k):
                self.assertGreater(need(lay, xray.with_ncmoe(full, j)), room, room)
        self.assertIsNone(xray.fit_ncmoe(lay, s, 1000))     # even every expert in RAM is too much

    def test_kv_cache_counts(self):
        lay, room = heavy(), 9000
        k = {name: xray.fit_ncmoe(lay, xray.fit_settings({}, lay, **kw)[0], room)
             for name, kw in (("4k", {"ctx": 4096}), ("32k", {"ctx": 32768}),
                              ("32k q8", {"ctx": 32768, "kv": "q8_0"}))}
        self.assertLess(k["4k"], k["32k"])          # 2 GiB of KV at 32k pushes experts out
        self.assertLess(k["32k q8"], k["32k"])      # q8_0 about halves it and pulls some back

    def test_placement_prefers_experts_in_ram(self):
        lay, s = heavy(), xray.fit_settings({}, heavy())[0]
        for room in (4500, 7000, 100000):
            self.assertEqual(xray.placement(lay, room, s), {"ngl": 5, "ncmoe": xray.fit_ncmoe(lay, s, room)})
        self.assertEqual(xray.placement(lay, 1000, s),
                         {"ngl": xray.fit_ngl(lay, s, 1000), "ncmoe": 0})
        self.assertEqual(xray.placement(dense(), 100000, s), {"ngl": 4, "ncmoe": 0})
        # the model's own cpu-moe doesn't stick to a whole-layer fit
        cm = dict(s, **{"cpu-moe": "true"})
        self.assertEqual(xray.placement(lay, 1000, cm), xray.placement(lay, 1000, s))

    def test_agrees_with_every_canirun_page(self):
        """Fit my VRAM and the published "Can I run it?" answers are one rule."""
        import json
        c = canirun()
        with open(c.DATA, encoding="utf-8") as f:
            data = json.load(f)
        checked = set()
        for m in c.valid_models(data):
            for q in m["quants"]:
                lay = {"arch": m["arch"], "kv": m["kv"], "tensors": q["tensors"]}
                s = xray.fit_settings({}, lay)[0]
                for gb in (6, 8, 12, 16, 24, 32):
                    a = c.answer(m, q, gb)
                    if a["status"] == "no":
                        continue
                    p = xray.placement(lay, gb * 1024 - xray.DESKTOP_MIB, s)
                    want = {"fits": {"ngl": a["layers"] + 1, "ncmoe": 0},
                            "experts": {"ngl": a["layers"] + 1, "ncmoe": a["cpu_moe"]},
                            "partial": {"ngl": a["gpu_layers"], "ncmoe": 0}}[a["status"]]
                    self.assertEqual(p, want, (m["slug"], q["quant"], gb, a["status"]))
                    checked.add(a["status"])
        self.assertEqual(checked, {"fits", "experts", "partial"})

    def test_with_ncmoe_replaces_any_spelling(self):
        s = xray.with_ncmoe({"ncmoe": "3", "cpu-moe": "", "ctx-size": "4096"}, 2)
        self.assertEqual(s, {"ctx-size": "4096", "n-cpu-moe": "2"})
        self.assertEqual(xray.with_ncmoe({"ncmoe": "3"}, 0), {})

    def test_ncmoe_of_reads_either_flag(self):
        self.assertEqual(xray.ncmoe_of({"cpu-moe": "true"}, 4), 4)
        self.assertEqual(xray.ncmoe_of({"cmoe": "on"}, 4), 4)
        self.assertEqual(xray.ncmoe_of({"cpu-moe": "false", "ncmoe": "2"}, 4), 2)
        self.assertEqual(xray.ncmoe_of({"n-cpu-moe": "99"}, 4), 4)
        self.assertEqual(xray.ncmoe_of({"n-cpu-moe": "x"}, 4), 0)
        self.assertEqual(xray.ncmoe_of({}, 4), 0)

    def test_flags(self):
        self.assertEqual(xray.flags(34, 0), "-ngl 34")
        self.assertEqual(xray.flags(41, 12), "-ngl 41 --n-cpu-moe 12")
        f = {"ctx": 8192, "ctk": "f16", "ctv": "f16"}
        self.assertEqual(xray.flags(41, 22, f), "-ngl 41 --n-cpu-moe 22 -c 8192")
        self.assertEqual(xray.flags(41, 0, dict(f, ctk="q8_0", ctv="q8_0")),
                         "-ngl 41 -c 8192 -ctk q8_0 -ctv q8_0")


class Hub(unittest.TestCase):
    def setUp(self):
        from unittest import mock
        import routes
        self.routes = routes
        mock.patch.object(routes, "total_vram_mib", return_value=8 * 1024).start()
        self.quants = mock.patch.object(xray, "quants", return_value=[
            {"path": "Gem-Q4_K_M.gguf", "size": 9 * G, "shards": 1},
            {"path": "Q8/Gem-Q8_0-00001-of-00002.gguf", "size": 20 * G, "shards": 2}]).start()
        self.remote = mock.patch.object(xray, "remote_layout", return_value=moe()).start()
        self.addCleanup(mock.patch.stopall)

    def get(self, **qs):
        return self.routes.get_hub_xray(self.routes.Req(qs=qs))

    def test_xrays_a_file_you_dont_have_and_starts_at_what_fits(self):
        st, v = self.get(repo="unsloth/Gem-GGUF", path="Q8/Gem-Q8_0-00001-of-00002.gguf")
        self.remote.assert_called_with("unsloth/Gem-GGUF", "Q8/Gem-Q8_0-00001-of-00002.gguf", shards=2)
        self.assertEqual(v["repo"], "unsloth/Gem-GGUF")
        self.assertEqual(v["file"], "Q8/Gem-Q8_0-00001-of-00002.gguf")
        self.assertEqual(v["vram"], 8 * G)
        self.assertEqual(v["fit"]["usable"], xray.usable_mib(8 * G) * M)
        s = xray.fit_settings({}, moe())[0]
        self.assertEqual(v["configured_ngl"], xray.placement(moe(), xray.usable_mib(8 * G), s)["ngl"])
        self.assertEqual(v["ngl"], v["configured_ngl"])
        self.assertNotIn("model", v)

    def test_ngl_and_vs(self):
        st, v = self.get(repo="unsloth/Gem-GGUF", path="Gem-Q4_K_M.gguf", ngl="2",
                         vs="Q8/Gem-Q8_0-00001-of-00002.gguf")
        self.assertEqual(v["ngl"], 2)
        self.assertEqual(v["vs"]["ngl"], 2)
        self.assertEqual(v["vs"]["file"], "Q8/Gem-Q8_0-00001-of-00002.gguf")

    def test_no_gpu_means_all_layers_and_no_budget(self):
        self.routes.total_vram_mib.return_value = 0
        st, v = self.get(repo="unsloth/Gem-GGUF", path="Gem-Q4_K_M.gguf")
        self.assertEqual(v["vram"], 0)
        self.assertEqual(v["configured_ngl"], 5)

    def test_only_files_the_repo_lists(self):
        for qs in ({"repo": "unsloth/Gem-GGUF", "path": "Other.gguf"},
                   {"repo": "unsloth/Gem-GGUF", "path": "Gem-Q4_K_M.gguf", "vs": "../x.gguf"},
                   {"repo": "../etc", "path": "Gem-Q4_K_M.gguf"},
                   {"repo": "a/b/c", "path": "Gem-Q4_K_M.gguf"},
                   {"repo": "unsloth/Gem-GGUF", "path": "Gem-Q4_K_M.gguf", "ngl": "lots"}):
            with self.assertRaises(self.routes.ApiError) as e:
                self.get(**qs)
            self.assertEqual(e.exception.status, 400, qs)

    def test_listing_or_header_unreachable(self):
        self.remote.return_value = None
        with self.assertRaises(self.routes.ApiError) as e:
            self.get(repo="unsloth/Gem-GGUF", path="Gem-Q4_K_M.gguf")
        self.assertEqual(e.exception.status, 502)
        self.quants.side_effect = OSError("offline")
        with self.assertRaises(self.routes.ApiError) as e:
            self.get(repo="unsloth/Gem-GGUF", path="Gem-Q4_K_M.gguf")
        self.assertEqual(e.exception.status, 502)

    def test_moe_starts_at_the_best_placement(self):
        self.routes.total_vram_mib.return_value = int(8.2 * 1024)
        st, v = self.get(repo="unsloth/Gem-GGUF", path="Gem-Q4_K_M.gguf")
        p = xray.placement(moe(), xray.usable_mib(int(8.2 * 1024) * M), xray.fit_settings({}, moe())[0])
        self.assertEqual((v["configured_ngl"], v["configured_ncmoe"]), (p["ngl"], p["ncmoe"]))
        self.assertEqual((v["ngl"], v["ncmoe"]), (p["ngl"], p["ncmoe"]))
        self.assertTrue(v["experts"])
        self.assertEqual(v["flags"], xray.flags(v["ngl"], v["ncmoe"], v["fit"]))

    def test_ctx_and_kv(self):
        self.remote.return_value = heavy()
        st, v = self.get(repo="unsloth/Gem-GGUF", path="Gem-Q4_K_M.gguf", ctx="32768", kv="q4_0")
        s = xray.fit_settings({}, heavy(), ctx=32768, kv="q4_0")[0]
        p = xray.placement(heavy(), xray.usable_mib(8 * G), s)
        self.assertEqual((v["ngl"], v["ncmoe"]), (p["ngl"], p["ncmoe"]))
        self.assertTrue(v["flags"].endswith("-c 32768 -ctk q4_0 -ctv q4_0"))
        with self.assertRaises(self.routes.ApiError) as e:
            self.get(repo="unsloth/Gem-GGUF", path="Gem-Q4_K_M.gguf", kv="q9")
        self.assertEqual(e.exception.status, 400)

    def test_ncmoe_param(self):
        st, v = self.get(repo="unsloth/Gem-GGUF", path="Gem-Q4_K_M.gguf", ngl="5", ncmoe="2",
                         vs="Q8/Gem-Q8_0-00001-of-00002.gguf")
        self.assertEqual(v["ncmoe"], 2)
        self.assertEqual(v["flags"], "-ngl 5 --n-cpu-moe 2 -c 8192")
        full = xray.view(moe(), {"n-gpu-layers": "5"})
        self.assertEqual(full["gpu"] - v["gpu"], 2 * 2000 * M)
        self.assertEqual(v["vs"]["gpu"], v["gpu"])
        with self.assertRaises(self.routes.ApiError) as e:
            self.get(repo="unsloth/Gem-GGUF", path="Gem-Q4_K_M.gguf", ncmoe="x")
        self.assertEqual(e.exception.status, 400)

    def test_dense_has_no_experts(self):
        self.remote.return_value = dense()
        st, v = self.get(repo="unsloth/Gem-GGUF", path="Gem-Q4_K_M.gguf")
        self.assertFalse(v["experts"])
        self.assertEqual(v["ncmoe"], 0)

    def test_registered(self):
        self.assertIs(self.routes.GET_ROUTES["/api/hub/xray"], self.routes.get_hub_xray)


if __name__ == "__main__":
    unittest.main()
