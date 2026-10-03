"""apply_ctx_defaults() must backfill without clobbering.

It runs on every panel startup (server.py main()), so anything it rewrites it
rewrites behind the user's back. Its job is to give models a sane ctx-size when
they have none - not to overrule one the user chose deliberately.

The regression this pins: a model whose GGUF trained length exceeds the global
default had its explicit per-model ctx-size DELETED so it would "inherit the
global". Trained length is not a VRAM budget - a 27B Q6_K that trains to 262144
still OOMs at 150000 on a 32 GB box - so the deletion silently reimposed a
config that could not load.
"""
import conftest_paths  # noqa: F401
import os, tempfile, unittest
from unittest import mock

import config


class ApplyCtxDefaultsTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "models.ini")

    def _write(self, text):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(text)

    def _read(self):
        return config.read_sections(self.path)

    def _run(self, default_ctx):
        """default_ctx: what gguf.default_ctx returns for every model here."""
        with mock.patch.object(config.gguf, "default_ctx", return_value=default_ctx):
            return config.apply_ctx_defaults(self.path)

    def test_never_pins_a_global_ctx(self):
        """A pinned ctx-size turns llama.cpp's --fit off (review 03 H1)."""
        self._write("[a]\nmodel = /m/a.gguf\n")
        self._run(0)
        self.assertNotIn("ctx-size", self._read().get("*", {}))

    def test_keeps_the_users_global_ctx(self):
        """It used to force [*] back to 150000 on every startup (review 03 H2)."""
        self._write("[*]\nctx-size = 32768\n\n[a]\nmodel = /m/a.gguf\n")
        self._run(0)
        self.assertEqual(self._read()["*"]["ctx-size"], "32768")

    def test_keeps_an_explicit_ctx_size_on_a_model_that_could_reach_the_global(self):
        """The regression. 65536 is there because 150000 does not fit in VRAM."""
        self._write("[*]\nctx-size = 150000\n\n[a]\nctx-size = 65536\nmodel = /m/a.gguf\n")
        self._run(0)
        self.assertEqual(self._read()["a"].get("ctx-size"), "65536")

    def test_does_not_backfill_per_model_pins(self):
        self._write("[*]\nctx-size = 150000\n\n[a]\nmodel = /m/a.gguf\n")
        self._run(40000)
        self.assertNotIn("ctx-size", self._read()["a"])

    def test_clamps_a_value_that_over_extends_the_trained_length(self):
        """Safety kept: never ask for more context than the model was trained on."""
        self._write("[*]\nctx-size = 150000\n\n[a]\nctx-size = 99999\nmodel = /m/a.gguf\n")
        self._run(40000)
        self.assertEqual(self._read()["a"].get("ctx-size"), "40000")

    def test_leaves_a_smaller_deliberate_value_below_the_cap(self):
        self._write("[*]\nctx-size = 150000\n\n[a]\nctx-size = 8192\nmodel = /m/a.gguf\n")
        self._run(40000)
        self.assertEqual(self._read()["a"].get("ctx-size"), "8192")

    def test_leaves_models_with_an_unreadable_trained_length_alone(self):
        self._write("[*]\nctx-size = 150000\n\n[a]\nctx-size = 4096\nmodel = /m/a.gguf\n")
        self._run(None)
        self.assertEqual(self._read()["a"].get("ctx-size"), "4096")

    def test_is_idempotent(self):
        self._write("[*]\nctx-size = 150000\n\n[a]\nctx-size = 65536\nmodel = /m/a.gguf\n")
        self._run(0)
        second = self._run(0)
        self.assertEqual(second["changed"], [],
                         "a second pass rewrote sections it had already settled")


class ReleaseLegacyCtxPinTest(unittest.TestCase):
    """Old versions forced [*] ctx-size = 150000; release it exactly once."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "models.ini")
        self._orig = config.CONFIG
        config.CONFIG = os.path.join(self.dir, "config.json")
        self.addCleanup(setattr, config, "CONFIG", self._orig)

    def _write(self, text):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(text)

    def test_drops_the_old_forced_value_once(self):
        self._write("[*]\nctx-size = 150000\nflash-attn = on\n\n[a]\nmodel = /m/a.gguf\n")
        self.assertTrue(config.release_legacy_ctx_pin(self.path))
        self.assertEqual(config.read_sections(self.path)["*"], {"flash-attn": "on"})
        # the user sets it back on purpose: later startups leave it alone
        config.set_keys("*", {"ctx-size": "150000"}, self.path)
        self.assertFalse(config.release_legacy_ctx_pin(self.path))
        self.assertEqual(config.read_sections(self.path)["*"]["ctx-size"], "150000")

    def test_leaves_any_other_global_ctx(self):
        self._write("[*]\nctx-size = 32768\n")
        self.assertFalse(config.release_legacy_ctx_pin(self.path))
        self.assertEqual(config.read_sections(self.path)["*"]["ctx-size"], "32768")
