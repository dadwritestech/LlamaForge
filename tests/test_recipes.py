import conftest_paths  # noqa: F401
import json, os, unittest

import recipes


def prof(**kw):
    return dict({"model": "qwen-coder", "backend": "llamacpp", "preset": "", "engine": ""}, **kw)


INSTALLS = [
    {"tag": "b11000", "variant": "cuda-13", "dir": "/e/b11000-cuda-13", "server_bin": "/e/b11000-cuda-13/s", "active": False},
    {"tag": "b11300", "variant": "cuda-13", "dir": "/e/b11300-cuda-13", "server_bin": "/e/b11300-cuda-13/s", "active": True},
]


# shaped like argspec.schema(): aliases are long forms without dashes, short
# flags only ever appear in "flags", reserved (router-owned) knobs are absent
SCHEMA = {"groups": [{"knobs": [
    {"key": "temp", "flags": ["--temp"], "aliases": ["temp"], "type": "float"},
    {"key": "gpu-layers", "flags": ["-ngl", "--gpu-layers", "--n-gpu-layers"],
     "aliases": ["gpu-layers", "n-gpu-layers"], "type": "int"},
    {"key": "mmap", "flags": ["--mmap", "--no-mmap"], "aliases": ["mmap", "no-mmap"], "type": "bool"},
    {"key": "timeout", "flags": ["-to", "--timeout"], "aliases": ["timeout"], "type": "int"}]}]}


class Shareable(unittest.TestCase):
    def test_plain_tuning_knobs_pass(self):
        for k in ["ctx-size", "temp", "top-k", "cache-type-k", "gpu-layers", "flash-attn", "override-tensor"]:
            self.assertTrue(recipes.shareable(k), k)

    def test_paths_hosts_and_secrets_are_dropped(self):
        for k in ["model", "mmproj", "model-draft", "hf-repo", "lora", "lora-scaled", "control-vector",
                  "chat-template-file", "slot-save-path", "log-file", "rpc", "api-key", "host", "port",
                  "ssl-key-file", "models-dir", "grammar-file", "Temp", "temp;x", "", "../x"]:
            self.assertFalse(recipes.shareable(k), k)

    def test_tools_cors_downloads_and_templates_are_dropped(self):
        # found by auditing every knob in a live `llama-server --help` schema
        for k in ["spec-draft-model", "lookup-cache-static", "lookup-cache-dynamic", "mcp-servers-config",
                  "mcp-servers-json", "tools", "tools-runtime", "agent", "ui-mcp-proxy", "ui", "ui-config",
                  "cors-origins", "cors-credentials", "docker-repo", "offline", "list-devices",
                  "gpt-oss-20b-default", "fim-qwen-7b-spec", "spec-default", "embd-gemma-default",
                  "chat-template", "reuse-port", "threads-http", "embedding", "rerank"]:
            self.assertFalse(recipes.shareable(k), k)

    def test_machine_specific_knobs_are_not_shared(self):
        for k in ["main-gpu", "tensor-split", "split-mode", "device", "threads", "threads-batch",
                  "cpu-mask", "cpu-range-batch", "numa", "prio", "poll", "spec-draft-device",
                  "spec-draft-threads", "spec-draft-cpu-mask"]:
            self.assertFalse(recipes.shareable(k), k)

    def test_any_knob_typed_path_is_dropped(self):
        self.assertTrue(recipes.shareable("ctx-size"))
        self.assertFalse(recipes.shareable("ctx-size", "path"))

    def test_still_shares_the_useful_stuff(self):
        for k in ["spec-type", "spec-draft-n-max", "reasoning", "reasoning-budget", "chat-template-kwargs",
                  "mmap", "fit", "n-cpu-moe", "spec-draft-n-cpu-moe", "jinja", "parallel",
                  "gpu-layers", "dry-multiplier", "yarn-orig-ctx", "mirostat-lr", "xtc-probability",
                  "mmproj-auto", "mmproj-offload"]:
            self.assertTrue(recipes.shareable(k), k)

    def test_only_allowlisted_knobs_are_shared(self):
        # a denylist can't anticipate the next upstream flag; recipes only carry
        # knobs someone decided are tuning, not server/runtime behaviour
        for k in ["timeout", "verbose", "verbosity", "cache-ram", "sleep-idle-seconds", "perf",
                  "escape", "check-tensors", "some-new-knob", "tags", "sse-ping-interval"]:
            self.assertFalse(recipes.shareable(k), k)


class Origin(unittest.TestCase):
    def test_llamaforge_download_folder(self):
        p = os.path.join("D:", os.sep, "models", "unsloth--Qwen3-GGUF", "Qwen3-Q4_K_M.gguf")
        self.assertEqual(recipes.origin(p), "unsloth/Qwen3-GGUF")

    def test_hf_cache_layout(self):
        p = "/home/u/.cache/huggingface/hub/models--bartowski--gemma-3-GGUF/snapshots/abc123/gemma-Q4.gguf"
        self.assertEqual(recipes.origin(p), "bartowski/gemma-3-GGUF")

    def test_unknown_layout_is_blank(self):
        self.assertEqual(recipes.origin("/models/gemma-Q4.gguf"), "")
        self.assertEqual(recipes.origin(""), "")


class Export(unittest.TestCase):
    section = {"model": "/m/unsloth--Qwen3-Coder-GGUF/Qwen3-Coder-Q4_K_M.gguf",
               "mmproj": "/m/x/mmproj.gguf", "ctx-size": "32768", "temp": "0.7", "rpc": "10.0.0.5:50052"}

    def test_section_knobs_overlaid_by_preset(self):
        r = recipes.export("coder", prof(preset="coding"), self.section,
                           {"coding": {"temp": "0.2", "top-k": ""}}, INSTALLS)
        self.assertEqual(r["llamaforge_recipe"], 1)
        self.assertEqual(r["name"], "coder")
        self.assertEqual(r["model"], {"id": "qwen-coder", "file": "Qwen3-Coder-Q4_K_M.gguf",
                                      "hf_repo": "unsloth/Qwen3-Coder-GGUF"})
        self.assertEqual(r["settings"], {"ctx-size": "32768", "temp": "0.2"})

    def test_pinned_engine_is_tag_and_variant_not_a_path(self):
        r = recipes.export("coder", prof(engine="b11000-cuda-13"), self.section, {}, INSTALLS)
        self.assertEqual(r["engine"], {"tag": "b11000", "variant": "cuda-13"})

    def test_unpinned_engine_reports_the_active_build(self):
        r = recipes.export("coder", prof(), self.section, {}, INSTALLS)
        self.assertEqual(r["engine"], {"tag": "b11300", "variant": "cuda-13"})

    def test_own_build_has_no_engine(self):
        r = recipes.export("coder", prof(), self.section, {}, [])
        self.assertIsNone(r["engine"])

    def test_vllm_profiles_cannot_be_exported(self):
        with self.assertRaises(ValueError):
            recipes.export("v", prof(backend="vllm"), {}, {}, INSTALLS)


class Parse(unittest.TestCase):
    def good(self, **kw):
        r = {"llamaforge_recipe": 1, "name": "coder",
             "model": {"id": "qwen-coder", "file": "Qwen3-Coder-Q4_K_M.gguf", "hf_repo": "unsloth/Qwen3-Coder-GGUF"},
             "settings": {"temp": "0.2", "ctx-size": 32768}, "engine": {"tag": "b11000", "variant": "cuda-13"}}
        r.update(kw)
        return r

    def test_accepts_dict_or_json_text(self):
        for src in [self.good(), json.dumps(self.good())]:
            p = recipes.parse(src)
            self.assertEqual(p["settings"], {"temp": "0.2", "ctx-size": "32768"})
            self.assertEqual(p["model"]["file"], "Qwen3-Coder-Q4_K_M.gguf")
            self.assertEqual(p["dropped"], [])

    def test_dangerous_keys_are_dropped_and_reported(self):
        p = recipes.parse(self.good(settings={"temp": "0.2", "rpc": "evil:1", "log-file": "C:/x"}))
        self.assertEqual(p["settings"], {"temp": "0.2"})
        self.assertEqual(sorted(p["dropped"]), ["log-file", "rpc"])

    def test_unknown_keys_dropped_when_schema_known(self):
        p = recipes.parse(self.good(), recipes.knob_index(SCHEMA))
        self.assertEqual((p["settings"], p["dropped"]), ({"temp": "0.2"}, ["ctx-size"]))

    def test_schema_types_drop_path_knobs(self):
        schema = {"groups": [{"knobs": [{"key": "temp", "type": "float"},
                                        {"key": "top-k", "type": "path"}]}]}
        p = recipes.parse(self.good(settings={"temp": "0.2", "top-k": "C:/x"}), recipes.knob_index(schema))
        self.assertEqual((p["settings"], p["dropped"]), ({"temp": "0.2"}, ["top-k"]))

    def test_aliases_are_canonicalized_through_the_schema(self):
        # n-gpu-layers is an alias of gpu-layers: allowlisting the canonical name
        # must not let an unlisted alias spelling slip past, nor drop a listed one
        p = recipes.parse(self.good(settings={"n-gpu-layers": "99", "no-mmap": "true", "temp": "0.2"}),
                          recipes.knob_index(SCHEMA))
        self.assertEqual(p["settings"], {"gpu-layers": "99", "no-mmap": "true", "temp": "0.2"})
        self.assertEqual(p["dropped"], [])

    def test_short_flags_and_aliases_of_denied_knobs_are_dropped(self):
        p = recipes.parse(self.good(settings={"hf-repo-draft": "x/y", "hfd": "x/y", "ngl": "99",
                                              "to": "5", "timeout": "5", "temp": "0.2"}),
                          recipes.knob_index(SCHEMA))
        self.assertEqual(p["settings"], {"temp": "0.2"})
        self.assertEqual(sorted(p["dropped"]), ["hf-repo-draft", "hfd", "ngl", "timeout", "to"])

    def test_without_a_schema_only_canonical_allowlisted_names_pass(self):
        p = recipes.parse(self.good(settings={"n-gpu-layers": "99", "gpu-layers": "99", "temp": "0.2"}))
        self.assertEqual((p["settings"], p["dropped"]), ({"gpu-layers": "99", "temp": "0.2"}, ["n-gpu-layers"]))

    def test_clean_filters_a_saved_preset_the_same_way(self):
        out = recipes.clean({"n-gpu-layers": "99", "timeout": "5", "temp": None}, recipes.knob_index(SCHEMA))
        self.assertEqual(out, ({"gpu-layers": "99", "temp": None}, ["timeout"]))

    def test_newlines_in_values_are_rejected(self):
        # a newline would let a recipe write arbitrary models.ini lines
        with self.assertRaises(ValueError):
            recipes.parse(self.good(settings={"temp": "0.2\nmodel = C:/evil.gguf"}))

    def test_file_must_be_a_bare_gguf_name(self):
        for bad in ["../../x.gguf", "C:\\m\\x.gguf", "x.bin", ""]:
            with self.assertRaises(ValueError, msg=bad):
                recipes.parse(self.good(model={"id": "m", "file": bad, "hf_repo": ""}))

    def test_repo_must_look_like_owner_slash_name(self):
        with self.assertRaises(ValueError):
            recipes.parse(self.good(model={"id": "m", "file": "x.gguf", "hf_repo": "https://evil/x"}))

    def test_not_a_recipe(self):
        for bad in ["{not json", json.dumps({"name": "x"}), json.dumps(self.good(llamaforge_recipe=99))]:
            with self.assertRaises(ValueError):
                recipes.parse(bad)


class Matching(unittest.TestCase):
    sections = {"*": {"ctx-size": "8192"},
                "my-qwen": {"model": "D:/models/Qwen3-Coder-Q4_K_M.gguf"},
                "gemma": {"model": "D:/models/gemma.gguf"}}

    def test_local_model_found_by_file_name(self):
        m = {"id": "qwen-coder", "file": "qwen3-coder-q4_k_m.gguf", "hf_repo": ""}
        self.assertEqual(recipes.match_local(m, self.sections), "my-qwen")

    def test_local_model_found_by_id(self):
        self.assertEqual(recipes.match_local({"id": "gemma", "file": "other.gguf", "hf_repo": ""}, self.sections), "gemma")

    def test_missing_model(self):
        self.assertIsNone(recipes.match_local({"id": "x", "file": "x.gguf", "hf_repo": ""}, self.sections))

    def test_engine_matched_to_installed_dir(self):
        self.assertEqual(recipes.match_engine({"tag": "b11000", "variant": "cuda-13"}, INSTALLS), "b11000-cuda-13")
        self.assertEqual(recipes.match_engine({"tag": "b9", "variant": "cpu"}, INSTALLS), "")
        self.assertEqual(recipes.match_engine(None, INSTALLS), "")

    def test_unique_name(self):
        self.assertEqual(recipes.unique_name("coder", {}), "coder")
        self.assertEqual(recipes.unique_name("coder", {"coder": 1, "coder 2": 1}), "coder 3")


class Routes(unittest.TestCase):
    """export -> import round trip through the real config store."""

    def setUp(self):
        import shutil, tempfile
        from unittest import mock
        import config, routes, prebuilt
        self.config, self.routes = config, routes
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        orig = config.CONFIG
        config.CONFIG = os.path.join(self.dir, "config.json")
        self.addCleanup(setattr, config, "CONFIG", orig)
        self.known = recipes.knob_index({"groups": [{"knobs": [
            {"key": k, "aliases": [k], "type": "str"} for k in ["ctx-size", "temp", "flash-attn", "timeout"]]}]})
        self.sections = {"qwen-coder": {"model": "D:/m/unsloth--Qwen3-Coder-GGUF/Qwen3-Coder-Q4_K_M.gguf",
                                        "ctx-size": "32768", "rpc": "10.0.0.5:1"}}
        for target, attr, val in [(config, "read_sections", lambda *a: self.sections),
                                  (prebuilt, "list_installs", lambda *a, **k: INSTALLS),
                                  (routes, "_known_knobs", lambda: self.known)]:
            pt = mock.patch.object(target, attr, val)
            pt.start(); self.addCleanup(pt.stop)
        config.save_preset("coding", {"temp": "0.2"})
        config.save_profile("coder", {"model": "qwen-coder", "preset": "coding", "engine": "b11000-cuda-13"})

    def call(self, fn, **body):
        return fn(self.routes.Req(body=body))[1]

    def test_round_trip_creates_preset_and_profile(self):
        recipe = self.call(self.routes.post_profiles_export, name="coder")["recipe"]
        self.assertNotIn("rpc", recipe["settings"])
        out = self.call(self.routes.post_profiles_import, recipe=json.dumps(recipe))
        self.assertTrue(out["ok"], out)
        self.assertEqual((out["name"], out["model"], out["engine"]), ("coder 2", "qwen-coder", "b11000-cuda-13"))
        self.assertEqual(self.config.get_presets()[out["preset"]], {"ctx-size": "32768", "temp": "0.2"})
        self.assertEqual(self.config.get_profiles()["coder 2"]["preset"], out["preset"])

    def test_export_includes_global_defaults_under_the_section(self):
        self.sections["*"] = {"flash-attn": "on", "ctx-size": "150000"}
        recipe = self.call(self.routes.post_profiles_export, name="coder")["recipe"]
        self.assertEqual(recipe["settings"], {"ctx-size": "32768", "flash-attn": "on", "temp": "0.2"})

    def test_missing_model_reports_where_to_get_it(self):
        recipe = self.call(self.routes.post_profiles_export, name="coder")["recipe"]
        self.sections = {}
        out = self.call(self.routes.post_profiles_import, recipe=recipe)
        self.assertFalse(out["ok"])
        self.assertEqual(out["missing"]["hf_repo"], "unsloth/Qwen3-Coder-GGUF")
        self.assertEqual(set(self.config.get_profiles()), {"coder"})

    def test_download_fetches_all_shards_from_the_repo(self):
        from unittest import mock
        recipe = self.call(self.routes.post_profiles_export, name="coder")["recipe"]
        recipe["model"]["file"] = "Big-00001-of-00002.gguf"
        self.sections = {}
        listing = {"files": [{"path": "Q4/Big-00001-of-00002.gguf", "shards": 2}]}
        with mock.patch.object(self.routes.hub, "files", return_value=listing), \
             mock.patch.object(self.routes.DOWNLOADS, "start", return_value=True) as start:
            out = self.call(self.routes.post_profiles_import, recipe=recipe, download=True)
        self.assertTrue(out["downloading"], out)
        repo, paths, _dest = start.call_args[0]
        self.assertEqual((repo, paths), ("unsloth/Qwen3-Coder-GGUF",
                                         ["Q4/Big-00001-of-00002.gguf", "Q4/Big-00002-of-00002.gguf"]))

    def test_import_is_refused_without_a_knob_schema(self):
        recipe = self.call(self.routes.post_profiles_export, name="coder")["recipe"]
        self.known = None
        with self.assertRaises(self.routes.ApiError) as cm:
            self.call(self.routes.post_profiles_import, recipe=recipe)
        self.assertEqual(cm.exception.status, 409)
        self.assertEqual(set(self.config.get_profiles()), {"coder"})

    def test_imported_profile_is_refiltered_at_launch(self):
        from unittest import mock
        recipe = self.call(self.routes.post_profiles_export, name="coder")["recipe"]
        out = self.call(self.routes.post_profiles_import, recipe=recipe)
        self.assertEqual(self.config.get_profiles()[out["name"]]["source"], "recipe")
        # re-saving it from the profile editor (no "source" field) keeps the mark
        self.config.save_profile(out["name"], dict(self.config.get_profiles()[out["name"]], source=""))
        self.assertEqual(self.config.get_profiles()[out["name"]]["source"], "recipe")
        # someone edits the imported preset to add a server knob afterwards
        self.config.save_preset(out["preset"], {"temp": "0.2", "timeout": "1"})
        applied = {}
        loader = mock.Mock(); loader.load.return_value = (True, "")
        with mock.patch.object(self.routes, "_apply_knobs_and_reload",
                               side_effect=lambda m, s: applied.update(s)),              mock.patch.object(self.routes, "_activate_prebuilt", return_value=(True, "")),              mock.patch.object(self.routes, "_wait_router", return_value=True),              mock.patch.object(self.routes.REGISTRY, "for_model", return_value=loader):
            res = self.call(self.routes.post_profiles_launch, name=out["name"])
        self.assertTrue(res["ok"], res)
        self.assertEqual(applied, {"temp": "0.2"})

    def test_evil_recipe_is_refused(self):
        evil = {"llamaforge_recipe": 1, "name": "x", "model": {"id": "x", "file": "x.gguf", "hf_repo": ""},
                "settings": {"temp": "1\n[evil]\nmodel = C:/x.gguf"}}
        with self.assertRaises(self.routes.ApiError):
            self.call(self.routes.post_profiles_import, recipe=evil)


if __name__ == "__main__":
    unittest.main()
