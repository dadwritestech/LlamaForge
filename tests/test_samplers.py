import conftest_paths  # noqa: F401
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import config
import routes
import samplers

QWEN = {"bos_token_id": 1, "do_sample": True, "temperature": 1.0, "top_k": 20, "top_p": 0.95}


class FromConfig(unittest.TestCase):
    def test_maps_hf_keys_to_llama_knobs(self):
        out = samplers.from_config({"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                                    "repetition_penalty": 1.05, "presence_penalty": 1.5, "max_length": 9})
        self.assertEqual(out, {"temp": "0.6", "top-p": "0.95", "top-k": "20", "min-p": "0",
                               "repeat-penalty": "1.05", "presence-penalty": "1.5"})

    def test_greedy_or_empty_or_junk_gives_nothing_but_greedy_is_named(self):
        self.assertEqual(samplers.from_config({"bos_token_id": 1}), {})
        self.assertEqual(samplers.from_config({"temperature": "hot", "top_k": True}), {})
        self.assertEqual(samplers.from_config({"do_sample": False}), {"temp": "0"})
        self.assertEqual(samplers.from_config({"do_sample": False, "temperature": 0.7}), {"temp": "0"})

    def test_formatting_keeps_ints_and_trims_floats(self):
        self.assertEqual(samplers.from_config({"temperature": 1, "top_p": 0.8000000001})["temp"], "1")
        self.assertEqual(samplers.from_config({"top_p": 0.8000000001})["top-p"], "0.8")


class Recommend(unittest.TestCase):
    def test_gguf_repo_first_then_base(self):
        def js(url):
            if "unsloth/Q-GGUF" in url:
                raise OSError("404")
            return QWEN
        with mock.patch.object(samplers.tmpldoctor, "_json", side_effect=js), \
                mock.patch.object(samplers.tmpldoctor, "base_repo", return_value="Qwen/Q"):
            r = samplers.recommend("D:/m/unsloth--Q-GGUF/Q.gguf")
        self.assertEqual(r["source"], "Qwen/Q")
        self.assertEqual(r["values"], {"temp": "1", "top-k": "20", "top-p": "0.95"})
        self.assertIn("Qwen/Q/blob/main/generation_config.json", r["url"])

    def test_nothing_published(self):
        with mock.patch.object(samplers.tmpldoctor, "_json", return_value={"bos_token_id": 1}), \
                mock.patch.object(samplers.tmpldoctor, "base_repo", return_value="m/base"):
            r = samplers.recommend("D:/m/a--b/x.gguf")
        self.assertEqual(r["values"], {})
        self.assertEqual(r["source"], "")
        self.assertIn("a/b", r["checked"])
        self.assertIn("m/base", r["checked"])

    def test_unknown_origin(self):
        r = samplers.recommend("D:/loose/x.gguf")
        self.assertEqual(r["values"], {})
        self.assertTrue(r["error"])


class Route(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.ini = os.path.join(self.d, "models.ini")
        self.cfg = os.path.join(self.d, "config.json")
        self._orig = config.CONFIG
        config.CONFIG = self.cfg
        with open(self.cfg, "w", encoding="utf-8") as f:
            json.dump({"models_ini": self.ini, "active_engine": "llamacpp"}, f)
        with open(self.ini, "w", encoding="utf-8") as f:
            f.write("[*]\ntemp = 0.7\n\n[m]\nmodel = D:/m/a--b/x.gguf\ntop-k = 40\n")

    def tearDown(self):
        config.CONFIG = self._orig
        shutil.rmtree(self.d, ignore_errors=True)

    def test_answer_carries_what_is_set_now(self):
        with mock.patch.object(samplers.tmpldoctor, "_json", return_value=QWEN):
            st, out = routes.GET_ROUTES["/api/samplers"](routes.Req(qs={"model": "m"}))
        self.assertEqual(st, 200)
        self.assertEqual(out["values"]["top-k"], "20")
        self.assertEqual(out["current"], {"top-k": "40", "temp": "0.7"})
        self.assertEqual(out["inherited"], ["temp"])

    def test_unknown_model(self):
        with self.assertRaises(routes.ApiError):
            routes.GET_ROUTES["/api/samplers"](routes.Req(qs={"model": "zz"}))


if __name__ == "__main__":
    unittest.main()
