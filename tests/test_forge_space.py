import conftest_paths  # noqa: F401
import unittest

from forge import space


def item(key, flags, typ="str", options=None):
    return {"key": key, "flags": flags, "type": typ, "options": options, "values": options}


FULL = [
    item("split-mode", ["-sm", "--split-mode"], "enum", ["none", "layer", "row", "tensor"]),
    item("device", ["-dev", "--device"]),
    item("tensor-split", ["-ts", "--tensor-split"]),
    item("cache-type-k", ["-ctk", "--cache-type-k"], "enum", ["f16", "q8_0", "q4_0"]),
    item("cache-type-v", ["-ctv", "--cache-type-v"], "enum", ["f16", "q8_0", "q4_0"]),
    item("spec-type", ["--spec-type"], "enum", ["none", "draft-simple", "draft-mtp", "draft-dflash"]),
    item("spec-draft-model", ["--spec-draft-model", "-md", "--model-draft"], "path"),
    item("spec-draft-device", ["--spec-draft-device", "-devd", "--device-draft"]),
    item("spec-draft-ngl", ["--spec-draft-ngl", "-ngld", "--gpu-layers-draft"]),
    item("spec-draft-n-max", ["--spec-draft-n-max"], "int"),
    item("ubatch-size", ["-ub", "--ubatch-size"], "int"),
    item("batch-size", ["-b", "--batch-size"], "int"),
    item("fit", ["-fit", "--fit"], "enum", ["on", "off"]),
]
OLD = [  # an older build: no tensor mode, old draft spellings, no spec-type
    item("split-mode", ["-sm", "--split-mode"], "enum", ["none", "layer", "row"]),
    item("device", ["-dev", "--device"]),
    item("tensor-split", ["-ts", "--tensor-split"]),
    item("model-draft", ["-md", "--model-draft"], "path"),
    item("device-draft", ["-devd", "--device-draft"]),
]
GPU16 = [{"index": 0, "name": "RTX 5080", "vram_mib": 16303},
         {"index": 1, "name": "RTX 5060 Ti", "vram_mib": 16311}]


class Caps(unittest.TestCase):
    def test_reads_modes_and_types_from_help(self):
        c = space.caps_from_items(FULL)
        self.assertEqual(c["split_modes"], ["none", "layer", "tensor"])  # row fails on CUDA
        self.assertIn("draft-dflash", c["spec_types"])
        self.assertEqual(c["kv_types"], ["f16", "q8_0", "q4_0"])
        self.assertEqual(c["keys"]["draft_model"], "spec-draft-model")
        self.assertEqual(c["keys"]["draft_ngl"], "spec-draft-ngl")

    def test_older_spellings_and_missing_flags(self):
        c = space.caps_from_items(OLD)
        self.assertEqual(c["split_modes"], ["none", "layer"])
        self.assertEqual(c["keys"]["draft_model"], "model-draft")
        self.assertEqual(c["keys"]["draft_device"], "device-draft")
        self.assertIsNone(c["keys"]["draft_n_max"])
        self.assertEqual(c["spec_types"], [])


class Placements(unittest.TestCase):
    def test_no_gpu_no_placements(self):
        self.assertEqual(space.placements([], space.caps_from_items(FULL)), [])

    def test_single_gpu_is_full_offload(self):
        ps = space.placements(GPU16[:1], space.caps_from_items(FULL))
        self.assertEqual(len(ps), 1)
        self.assertEqual(ps[0]["devices"], [0])

    def test_two_gpus_solo_each_and_splits_per_mode(self):
        ps = space.placements(GPU16, space.caps_from_items(FULL), steps=(0, 4))
        solos = [p for p in ps if p["settings"].get("split-mode") == "none"]
        self.assertEqual([p["settings"]["device"] for p in solos], ["CUDA0", "CUDA1"])
        splits = {(p["settings"]["split-mode"], p["settings"]["tensor-split"]) for p in ps
                  if p["settings"].get("split-mode") in ("layer", "tensor")}
        self.assertEqual(splits, {("layer", "16,16"), ("layer", "20,12"),
                                  ("tensor", "16,16"), ("tensor", "20,12")})

    def test_split_ratios_follow_vram_and_clamp(self):
        gpus = [{"index": 0, "name": "a", "vram_mib": 24000}, {"index": 1, "name": "b", "vram_mib": 8000}]
        ps = space.placements(gpus, space.caps_from_items(OLD), steps=(0, 8))
        ratios = sorted(p["settings"]["tensor-split"] for p in ps if "tensor-split" in p["settings"])
        self.assertEqual(ratios, ["24,8", "30,2"])  # 32 would leave GPU1 nothing: clamped

    def test_no_device_flag_means_no_solo(self):
        caps = space.caps_from_items([i for i in OLD if i["key"] != "device"])
        ps = space.placements(GPU16, caps, steps=(0,))
        self.assertFalse(any(p["settings"].get("split-mode") == "none" for p in ps))


INV = {"gpus": GPU16, "caps": space.caps_from_items(FULL),
       "base": {"model": "/m/q.gguf", "cache-type-k": "q8_0", "cache-type-v": "q8_0",
                "ctx-size": "8192", "temp": "0.7", "tensor-split": "1,1"},
       "drafters": [{"path": "/m/dflash.gguf", "kind": "dflash", "mib": 1800}],
       "mtp": False}


class Candidates(unittest.TestCase):
    def cands(self, **req):
        r = {"intent": "coding", "ctx": 32768, "allow_loss": False, "time": "quick"}
        r.update(req)
        return space.candidates(INV, r)

    def test_keeps_user_settings_but_owns_placement_and_ctx(self):
        c = self.cands()[0]
        s = c["settings"]
        self.assertEqual(s["temp"], "0.7")
        self.assertEqual(s["ctx-size"], "32768")
        self.assertEqual(s["fit"], "off")       # a silent fit-down would fake a pass
        self.assertNotEqual(s.get("tensor-split"), "1,1")

    def test_lossless_by_default(self):
        cs = self.cands()
        self.assertTrue(all(not c["lossy"] for c in cs))
        self.assertTrue(all(c["settings"]["cache-type-v"] == "q8_0" for c in cs))

    def test_allow_loss_adds_marked_kv_variants(self):
        cs = self.cands(allow_loss=True)
        lossy = [c for c in cs if c["lossy"]]
        self.assertTrue(lossy)
        self.assertTrue(all(c["settings"]["cache-type-v"] == "q4_0" for c in lossy))

    def test_drafter_on_each_gpu_with_build_spellings(self):
        cs = [c for c in self.cands() if "spec-draft-model" in c["settings"]]
        solo0 = [c for c in cs if c["settings"].get("device") == "CUDA0"]
        self.assertEqual(sorted(c["settings"]["spec-draft-device"] for c in solo0), ["CUDA0", "CUDA1"])
        c = solo0[0]["settings"]
        self.assertEqual(c["spec-type"], "draft-dflash")
        self.assertEqual(c["spec-draft-ngl"], "99")

    def test_drafter_skipped_when_build_lacks_its_type(self):
        inv = dict(INV, caps=space.caps_from_items(OLD))
        cs = space.candidates(inv, {"intent": "chat", "ctx": 8192, "allow_loss": False, "time": "quick"})
        self.assertFalse(any("model-draft" in c["settings"] for c in cs))

    def test_ids_are_stable_and_unique(self):
        a, b = self.cands(), self.cands()
        self.assertEqual([c["id"] for c in a], [c["id"] for c in b])
        self.assertEqual(len({c["id"] for c in a}), len(a))

    def test_spec_variants_expand_draft_window(self):
        c = next(c for c in self.cands(time="thorough") if "spec-draft-model" in c["settings"])
        ns = sorted(int(v["settings"]["spec-draft-n-max"]) for v in space.spec_variants(c, "thorough"))
        self.assertEqual(ns, [5, 7, 10])
        for v in space.spec_variants(c, "thorough", gpus=INV["gpus"]):
            self.assertIn(f"({v['settings']['spec-draft-n-max']} ahead)", v["label"])
        plain = next(c for c in self.cands() if "spec-draft-model" not in c["settings"])
        self.assertEqual(space.spec_variants(plain, "thorough"), [plain])

    def test_plain_only_then_drafters_on_one_placement(self):
        r = {"intent": "coding", "ctx": 32768, "allow_loss": False, "time": "quick"}
        plain = space.candidates(INV, r, drafters=False)
        self.assertTrue(plain and all(not c["spec"] for c in plain))
        self.assertEqual(len(plain), len([c for c in self.cands() if not c["spec"]]))
        solo0 = next(c for c in plain if c["settings"].get("device") == "CUDA0")
        drafted = space.with_drafters(solo0, INV)
        self.assertEqual(sorted(d["settings"]["spec-draft-device"] for d in drafted), ["CUDA0", "CUDA1"])
        full = {c["id"] for c in self.cands()}
        for d in drafted:                 # the same candidates the one-shot search makes
            self.assertIn(d["id"], full)
            self.assertEqual(d["spec"], "dflash")
            self.assertEqual(d["parts"], solo0["parts"] + 1)
            self.assertEqual(d["settings"]["temp"], "0.7")


class NoVision(unittest.TestCase):
    VIS = dict(INV, base=dict(INV["base"], mmproj="/m/mmproj-F16.gguf", **{"no-mmproj-offload": "true"}))
    REQ = {"intent": "coding", "ctx": 32768, "allow_loss": False, "time": "quick"}

    def test_off_unless_asked(self):
        cs = space.candidates(self.VIS, self.REQ, drafters=False)
        self.assertTrue(all(c["settings"].get("mmproj") for c in cs))
        self.assertFalse(any(c.get("novision") for c in cs))

    def test_each_placement_also_without_the_projector(self):
        with_v = space.candidates(self.VIS, self.REQ, drafters=False)
        cs = space.candidates(self.VIS, dict(self.REQ, drop_vision=True), drafters=False)
        off = [c for c in cs if c.get("novision")]
        self.assertEqual(len(off), len(with_v))
        self.assertEqual(len(cs), 2 * len(with_v))
        for c in off:
            self.assertFalse(any("mmproj" in k for k in c["settings"]))
            self.assertEqual(c["settings"]["temp"], "0.7")
            self.assertIn("no vision", c["label"])
        self.assertEqual(len({c["id"] for c in cs}), len(cs))

    def test_nothing_to_drop_without_a_projector(self):
        cs = space.candidates(INV, dict(self.REQ, drop_vision=True), drafters=False)
        self.assertFalse(any(c.get("novision") for c in cs))

    def test_drafters_and_windows_keep_it(self):
        cs = space.candidates(self.VIS, dict(self.REQ, drop_vision=True), drafters=False)
        solo = next(c for c in cs if c.get("novision") and c["settings"].get("device") == "CUDA0")
        for d in space.with_drafters(solo, self.VIS):
            self.assertTrue(d["novision"])
            self.assertIn("no vision", d["label"])
            for v in space.spec_variants(d, "thorough", GPU16):
                self.assertTrue(v["novision"])
                self.assertIn("no vision", v["label"])


class Prune(unittest.TestCase):
    def test_rejects_predicted_overflow_with_reason(self):
        cs = [{"id": "a", "settings": {}}, {"id": "b", "settings": {}}, {"id": "c", "settings": {}}]
        est = {"a": [15000, 2000], "b": [16200, 0], "c": None}
        keep, rej = space.prune(cs, lambda c: est[c["id"]], GPU16)
        self.assertEqual([c["id"] for c in keep], ["a", "c"])   # unknown estimate: let the probe decide
        self.assertEqual(rej[0]["id"], "b")
        self.assertIn("GPU0", rej[0]["reason"])

    def test_split_estimate(self):
        solo = {"split-mode": "none", "device": "CUDA1"}
        self.assertEqual(space.split_estimate(solo, 10000, 0, 2), [0, 10000])
        split = {"split-mode": "layer", "tensor-split": "24,8", "spec-draft-device": "CUDA1"}
        self.assertEqual(space.split_estimate(split, 8000, 1000, 2), [6000, 3000])


class Rank(unittest.TestCase):
    def row(self, id, code, prose, prefill, vram, parts=1, ctx=32768):
        return {"id": id, "code": code, "prose": prose, "prefill": prefill,
                "vram_mib": vram, "parts": parts, "ctx": ctx}

    def test_intent_changes_winner(self):
        rows = [self.row("fastcode", 130, 50, 1500, 20000), self.row("fastprose", 100, 70, 1500, 20000)]
        self.assertEqual(space.rank(rows, "coding")[0]["id"], "fastcode")
        self.assertEqual(space.rank(rows, "chat")[0]["id"], "fastprose")

    def test_near_tie_prefers_less_vram_then_fewer_parts(self):
        rows = [self.row("big", 100, 50, 1500, 30000, parts=2), self.row("small", 98, 49, 1490, 16000, parts=2),
                self.row("simple", 98, 49, 1490, 16000, parts=1)]
        self.assertEqual(space.rank(rows, "coding")[0]["id"], "simple")

    def test_clear_winner_beats_tie_break(self):
        rows = [self.row("big", 130, 60, 1500, 30000), self.row("small", 100, 50, 1500, 16000)]
        self.assertEqual(space.rank(rows, "coding")[0]["id"], "big")

    def test_missing_metric_is_reweighted_not_zeroed(self):
        rows = [{"id": "a", "code": 100, "prose": 50, "vram_mib": 1, "parts": 1, "ctx": 1},
                {"id": "b", "code": 90, "prose": 45, "vram_mib": 1, "parts": 1, "ctx": 1}]
        self.assertEqual(space.rank(rows, "coding")[0]["id"], "a")


class Label(unittest.TestCase):
    def test_plain_words(self):
        s = {"split-mode": "none", "device": "CUDA0", "spec-draft-model": "/m/dflash.gguf",
             "spec-draft-device": "CUDA1", "cache-type-k": "q8_0", "cache-type-v": "q8_0", "ctx-size": "32768"}
        self.assertEqual(space.label(s, GPU16), "RTX 5080 alone + drafter dflash on RTX 5060 Ti, KV 8-bit, 32k")
        s2 = dict(s, **{"spec-draft-model": "/m/Qwen3.8-27B-DFlash2-Q8_0-selfc.gguf", "spec-draft-n-max": "10"})
        self.assertEqual(space.label(s2, GPU16),
                         "RTX 5080 alone + drafter Qwen3.8-27B-DFlash2-Q8_0-selfc (10 ahead) on RTX 5060 Ti, KV 8-bit, 32k")
        s = {"split-mode": "layer", "tensor-split": "22,10", "cache-type-k": "q8_0",
             "cache-type-v": "q4_0", "ctx-size": "131072"}
        self.assertEqual(space.label(s, GPU16), "layer split 22:10, KV 8/4-bit, 128k")


if __name__ == "__main__":
    unittest.main()
