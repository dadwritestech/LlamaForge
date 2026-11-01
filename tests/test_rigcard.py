import conftest_paths  # noqa: F401
import unittest

import rigcard

GPUS = [{"index": 0, "name": "NVIDIA GeForce RTX 5080", "used": 1000, "total": 16303},
        {"index": 1, "name": "NVIDIA GeForce RTX 5060 Ti", "used": 10, "total": 16311}]
STATS = {"totals": {"generated": 41_234_567, "prompt": 90_000_000, "loaded_hours": 312.4},
         "per_model": [
             {"id": "qwen3.6-35b-a3b", "generated": 30_000_000, "tokens": 90_000_000, "avg_tps": 999, "decode_tps": 131.6},
             {"id": "gemma-4-12b", "generated": 11_000_000, "tokens": 40_000_000, "avg_tps": 999, "decode_tps": 88.04},
             {"id": "never-used", "generated": 0, "tokens": 0, "avg_tps": 0},
         ]}


def build(**kw):
    a = dict(gpus=GPUS, ram={"used": 1, "total": 65406}, platform="windows",
             version="0.18.11", stats=STATS)
    a.update(kw)
    return rigcard.build(**a)


class RigCardTest(unittest.TestCase):
    def test_hardware_line(self):
        d = build()
        self.assertEqual(d["gpus"], ["RTX 5080 16 GB", "RTX 5060 Ti 16 GB"])
        self.assertEqual(d["ram_gb"], 64)
        self.assertEqual(d["os"], "Windows")
        self.assertEqual(d["version"], "0.18.11")

    def test_models_by_use_with_speed_and_unused_dropped(self):
        d = build()
        self.assertEqual([m["name"] for m in d["models"]], ["qwen3.6-35b-a3b", "gemma-4-12b"])
        self.assertEqual(d["models"][1]["tps"], 88.0)
        self.assertEqual(d["models"][0]["generated"], 30_000_000)

    def test_at_most_five_models(self):
        many = {"totals": STATS["totals"], "per_model": [
            {"id": f"m{i}", "generated": 100 - i, "tokens": 100 - i, "avg_tps": 1} for i in range(9)]}
        self.assertEqual(len(build(stats=many)["models"]), 5)

    def test_totals(self):
        d = build()
        self.assertEqual(d["generated"], 41_234_567)
        self.assertEqual(d["hours"], 312)

    def test_paths_and_markup_never_leave(self):
        s = {"totals": STATS["totals"], "per_model": [
            {"id": "C:\\Users\\someone\\models\\secret-model.gguf", "generated": 5, "tokens": 5, "avg_tps": 1},
            {"id": "a|b](http://x)<img>`", "generated": 4, "tokens": 4, "avg_tps": 1}]}
        d = build(stats=s)
        self.assertEqual(d["models"][0]["name"], "secret-model.gguf")
        self.assertNotIn("someone", d["markdown"])
        for ch in "|[]<>`":
            self.assertNotIn(ch, d["models"][1]["name"])

    def test_gpu_errors_and_no_ram_are_skipped(self):
        d = build(gpus=[{"error": "nvidia-smi not found"}], ram=None, platform="linux")
        self.assertEqual(d["gpus"], [])
        self.assertIsNone(d["ram_gb"])
        self.assertEqual(d["os"], "Linux")

    def test_human_counts(self):
        self.assertEqual(rigcard.human(41_234_567), "41.2M")
        self.assertEqual(rigcard.human(1_500_000_000), "1.5B")
        self.assertEqual(rigcard.human(12_345), "12.3k")
        self.assertEqual(rigcard.human(999), "999")

    def test_markdown(self):
        md = build()["markdown"]
        self.assertIn("RTX 5080 16 GB + RTX 5060 Ti 16 GB", md)
        self.assertIn("| qwen3.6-35b-a3b | 131.6 | 30.0M |", md)
        self.assertIn("41.2M tokens generated", md)
        self.assertIn("github.com/dadwritestech/LlamaForge", md)

    def test_empty_stats(self):
        d = build(stats={"totals": {}, "per_model": []})
        self.assertEqual(d["models"], [])
        self.assertEqual(d["generated"], 0)
        self.assertIn("LlamaForge", d["markdown"])


class RouteTest(unittest.TestCase):
    def test_route_registered(self):
        import routes
        self.assertIn("/api/rig", routes.GET_ROUTES)


if __name__ == "__main__":
    unittest.main()
