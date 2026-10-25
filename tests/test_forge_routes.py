import conftest_paths  # noqa: F401
import json
import os
import shutil
import tempfile
import threading
import types
import unittest
from unittest import mock

import config
import forge_panel


class ApiError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status, self.message = status, message


class Req:
    def __init__(self, body=None, qs=None):
        self.body, self.qs = body or {}, qs or {}

    def q(self, name, default=""):
        return self.qs.get(name, default)


def deps(tmp, loaded=("m",)):
    calls = types.SimpleNamespace(unloaded=0, loaded=[], applied=[])
    d = types.SimpleNamespace(
        ApiError=ApiError, Req=Req, LOGDIR=tmp,
        _game_loaded=lambda: list(loaded),
        _game_load=lambda mid: calls.loaded.append(mid),
        post_unload_all=lambda req: setattr(calls, "unloaded", calls.unloaded + 1),
        _apply_knobs_and_reload=lambda mid, upd: calls.applied.append((mid, dict(upd))),
        _active_server_bin=lambda c=None: "llama-server",
        SLOTS=types.SimpleNamespace(help_items=lambda sbin: []))
    return d, calls


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.d, self.calls = deps(self.tmp)
        self.panel = forge_panel.Panel(self.d)
        self.sections = {"*": {}, "m": {"model": "/m/q.gguf", "tensor-split": "1,1", "temp": "0.7"}}
        p = mock.patch.object(config, "read_sections", lambda *a, **k: self.sections)
        p.start()
        self.addCleanup(p.stop)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_result(self, mid="m"):
        st = {"stage": "done", "cands": [], "rows": {}, "req": {"intent": "coding"},
              "result": {"winner": {"id": "w", "settings": {
                  "model": "/m/q.gguf", "temp": "0.7", "split-mode": "none", "device": "CUDA0"}}}}
        os.makedirs(self.panel.dir, exist_ok=True)
        with open(self.panel._state_path(mid), "w", encoding="utf-8") as f:
            json.dump(st, f)


class Lease(Base):
    def test_unloads_then_loads_back_even_on_error(self):
        with self.assertRaises(RuntimeError):
            with self.panel._lease():
                self.assertEqual(self.calls.unloaded, 1)
                raise RuntimeError("boom")
        self.assertEqual(self.calls.loaded, ["m"])


class Start(Base):
    def test_rejects_bad_ctx(self):
        with self.assertRaises(ApiError) as e:
            self.panel.post_start(Req({"model": "m", "ctx": "lots"}))
        self.assertEqual(e.exception.status, 400)

    def test_no_help_items_is_a_clear_error(self):
        with self.assertRaises(ApiError) as e:
            self.panel.post_start(Req({"model": "m"}))
        self.assertIn("arguments", e.exception.message)

    def test_one_run_at_a_time(self):
        gate = threading.Event()
        fake = types.SimpleNamespace(run=lambda: gate.wait(2), snapshot=lambda: {"stage": "fit"},
                                     cancel=threading.Event())
        with mock.patch.object(self.panel, "_make_job", lambda mid, r: fake):
            self.assertEqual(self.panel.post_start(Req({"model": "m"}))[0], 200)
            with self.assertRaises(ApiError) as e:
                self.panel.post_start(Req({"model": "m"}))
            self.assertEqual(e.exception.status, 409)
            gate.set()
            self.panel.runner.join(2)

    def test_status_reads_the_saved_state_when_idle(self):
        self.write_result()
        st = self.panel.get_status(Req(qs={"model": "m"}))[1]
        self.assertEqual(st["stage"], "done")
        self.assertFalse(st["running"])


class ApplyUndo(Base):
    def test_apply_then_undo_round_trips(self):
        self.write_result()
        self.panel.post_apply(Req({"model": "m"}))
        mid, upd = self.calls.applied[-1]
        self.assertEqual(upd, {"split-mode": "none", "device": "CUDA0", "tensor-split": None})
        self.panel.post_undo(Req({"model": "m"}))
        self.assertEqual(self.calls.applied[-1][1],
                         {"split-mode": None, "device": None, "tensor-split": "1,1"})
        with self.assertRaises(ApiError):
            self.panel.post_undo(Req({"model": "m"}))

    def test_apply_without_a_result(self):
        with self.assertRaises(ApiError) as e:
            self.panel.post_apply(Req({"model": "m"}))
        self.assertEqual(e.exception.status, 400)


class Estimate(unittest.TestCase):
    GPUS = 2

    def est(self, settings, layout=None):
        lay = layout if layout is not None else {"arch": "x", "kv": {}, "tensors": []}
        sizes = {"/m/q.gguf": 10000 << 20, "/m/d.gguf": 2000 << 20, "/m/mm.gguf": 800 << 20}
        fake = lambda s, l, mmproj_bytes=0: {"need_mib": 12000, "first_gpu_mib": 1000 if mmproj_bytes else 0}
        with mock.patch.object(forge_panel.footprint, "predict", fake):
            e = forge_panel.make_estimate(self.GPUS, layout_of=lambda p: lay, size_of=lambda p: sizes[p])
            return e({"id": "c", "settings": settings})

    def test_solo_with_drafter_on_the_other_gpu(self):
        s = {"model": "/m/q.gguf", "split-mode": "none", "device": "CUDA1",
             "spec-draft-model": "/m/d.gguf", "spec-draft-device": "CUDA0", "mmproj": "/m/mm.gguf"}
        need = self.est(s)
        self.assertGreater(need[0], 2000 + 1000)     # drafter + projector on CUDA0
        self.assertEqual(need[1], 12000)

    def test_unreadable_model_is_left_to_the_probe(self):
        self.assertIsNone(self.est({"model": "/m/q.gguf"}, layout=False))


class Wiring(unittest.TestCase):
    def test_routes_registered(self):
        import routes
        self.assertIn("/api/forge/status", routes.GET_ROUTES)
        for p in ("start", "cancel", "apply", "undo"):
            self.assertIn(f"/api/forge/{p}", routes.POST_ROUTES)


if __name__ == "__main__":
    unittest.main()
