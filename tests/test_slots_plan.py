import conftest_paths  # noqa: F401
import unittest

import slots

G = 1024                      # MiB per GiB, to keep the numbers readable


def gpus(*used, total=16 * G):
    return [{"index": i, "name": f"gpu{i}", "total_mib": total, "used_mib": u}
            for i, u in enumerate(used)]


def cand(model="m", need=6 * G, pins=None, measured=None, ram_mib=0, first_gpu_mib=0):
    return {"model": model, "need_mib": need, "pins": pins,
            "measured": measured or {}, "ram_mib": ram_mib, "first_gpu_mib": first_gpu_mib}


class Pins(unittest.TestCase):
    def test_unpinned(self):
        self.assertIsNone(slots.pins({"n-gpu-layers": "99", "ctx-size": "8192"}))

    def test_device_list(self):
        self.assertEqual(slots.pins({"device": "CUDA1"}), {"devices": [1], "share": {1: 1.0}})
        self.assertEqual(slots.pins({"device": "CUDA0,CUDA1"}),
                         {"devices": [0, 1], "share": {0: 0.5, 1: 0.5}})

    def test_device_none_is_cpu_only(self):
        self.assertEqual(slots.pins({"device": "none"}), {"devices": [], "share": {}})

    def test_split_mode_none_uses_main_gpu(self):
        self.assertEqual(slots.pins({"split-mode": "none", "main-gpu": "1"}),
                         {"devices": [1], "share": {1: 1.0}})
        self.assertEqual(slots.pins({"split-mode": "none"}), {"devices": [0], "share": {0: 1.0}})

    def test_tensor_split_shares(self):
        self.assertEqual(slots.pins({"tensor-split": "3,1"}),
                         {"devices": [0, 1], "share": {0: 0.75, 1: 0.25}})
        self.assertEqual(slots.pins({"tensor-split": "0,1"}), {"devices": [1], "share": {1: 1.0}})

    def test_tensor_split_narrows_a_device_list(self):
        self.assertEqual(slots.pins({"device": "CUDA0,CUDA1", "tensor-split": "16,16"}),
                         {"devices": [0, 1], "share": {0: 0.5, 1: 0.5}})

    def test_unreadable_pins_are_unknown_not_unpinned(self):
        # a device we can't map (Vulkan0, a typo) must not be "free to place"
        p = slots.pins({"device": "Vulkan0"})
        self.assertEqual(p, {"devices": None, "share": {}})


class Plan(unittest.TestCase):
    def test_worker_best_fit_keeps_the_big_hole(self):
        v = slots.plan(cand(need=4 * G), gpus(2 * G, 8 * G), [], headroom_mib=1 * G, role="worker")
        self.assertTrue(v["ok"], v["reason"])
        self.assertEqual(v["devices"], [1])          # 7 GiB free beats 13: best fit
        self.assertEqual(v["place"], "CUDA1")

    def test_main_takes_the_fastest_gpu_that_fits(self):
        # CUDA numbers devices fastest-first, so CUDA0 wins even with less free
        # (live: the main landed on a 5060 Ti on PCIe x4 for 70 MiB more room
        # while the 5080 sat idle)
        v = slots.plan(cand(need=4 * G), gpus(8 * G, 2 * G), [], headroom_mib=1 * G, role="main")
        self.assertEqual((v["devices"], v["place"]), ([0], "CUDA0"))
        v = slots.plan(cand(need=4 * G), gpus(0, 0), [], headroom_mib=1 * G, role="main",
                       cmap={0: 1, 1: 0})
        self.assertEqual((v["devices"], v["place"]), ([1], "CUDA0"))
        v = slots.plan(cand(need=4 * G), gpus(13 * G, 0), [], headroom_mib=1 * G, role="main")
        self.assertEqual(v["devices"], [1])           # CUDA0 has no room: the next one

    def test_a_worker_stays_off_the_mains_gpu(self):
        main = [{"model": "chat", "role": "main", "devices": [0], "footprint": {0: 8 * G}}]
        v = slots.plan(cand(need=4 * G), gpus(8 * G, 2 * G), main, headroom_mib=1 * G,
                       role="worker")
        self.assertEqual(v["devices"], [1])           # GPU0 is the tighter fit, but it's chat's
        v = slots.plan(cand(need=4 * G), gpus(8 * G, 14 * G), main, headroom_mib=1 * G,
                       role="worker")
        self.assertEqual(v["devices"], [0])           # nowhere else fits: share it

    def test_workers_still_pack_away_from_the_main(self):
        main = [{"model": "chat", "role": "main", "devices": [0], "footprint": {0: 4 * G}}]
        g = gpus(10 * G, 2 * G, 8 * G)               # free after headroom: 5, 13, 7
        v = slots.plan(cand(need=4 * G), g, main, headroom_mib=1 * G, role="worker")
        self.assertEqual(v["devices"], [2])           # tightest of the GPUs chat isn't on

    def test_unset_context_is_named_in_the_refusal(self):
        c = cand(need=28 * G)
        c["ctx_from_model"] = 262144
        v = slots.plan(c, gpus(0, 0), [], headroom_mib=1 * G, role="worker")
        self.assertFalse(v["ok"])
        self.assertIn("ctx-size", v["reason"])
        self.assertIn("262,144", v["reason"])
        self.assertNotIn("beside other models", v["reason"])     # nothing else is loaded
        self.assertIn("fits your GPUs", v["reason"])

    def test_unset_context_beside_loaded_models_says_beside(self):
        c = cand(need=28 * G)
        c["ctx_from_model"] = 262144
        main = [{"model": "m0", "role": "main", "devices": [0]}]
        v = slots.plan(c, gpus(0, 0), main, headroom_mib=1 * G, role="worker")
        self.assertFalse(v["ok"])
        self.assertIn("beside other models", v["reason"])

    def test_predicted_off_gpu0_pays_a_context_on_gpu0(self):
        # GPU0 is too full for the model (3 GiB free) but not for a CUDA context
        v = slots.plan(cand(need=4 * G), gpus(12 * G, 0), [], headroom_mib=1 * G, role="worker")
        self.assertEqual(v["devices"], [1])
        self.assertEqual(v["footprint"], {1: 4 * G, 0: slots.OTHER_GPU0_MIB})
        self.assertEqual(v["source"], "predicted")

    def test_the_gpu0_context_can_be_what_does_not_fit(self):
        # GPU1 has room, but GPU0 has no room left even for the context
        full0 = 16 * G - 1 * G - 100
        v = slots.plan(cand(need=4 * G), gpus(full0, 0), [], headroom_mib=1 * G, role="worker")
        self.assertFalse(v["ok"])

    def test_measured_beats_predicted(self):
        # predicted 12 GiB fits nowhere; the measured 5 GiB (+231 on GPU0) fits GPU1
        c = cand(need=12 * G, measured={(1,): {1: 5 * G, 0: 231}})
        v = slots.plan(c, gpus(14 * G, 4 * G), [], headroom_mib=1 * G, role="worker")
        self.assertTrue(v["ok"], v["reason"])
        self.assertEqual(v["devices"], [1])
        self.assertEqual(v["source"], "measured")
        self.assertEqual(v["footprint"], {1: 5 * G, 0: 231})

    def test_headroom_is_respected(self):
        v = slots.plan(cand(need=7 * G), gpus(8 * G, 8 * G), [], headroom_mib=2 * G, role="worker")
        self.assertFalse(v["ok"])
        self.assertIn("headroom", v["reason"])

    def test_refusal_names_the_numbers(self):
        v = slots.plan(cand(model="big", need=10 * G), gpus(9 * G, 9 * G), [],
                       headroom_mib=1 * G, role="worker")
        self.assertFalse(v["ok"])
        self.assertIn("10.0 GiB", v["reason"])
        self.assertIn("6.0 GiB", v["reason"])        # what each GPU has after headroom
        self.assertEqual(v["evict"], [])             # workers never evict

    def test_worker_never_spans(self):
        v = slots.plan(cand(need=10 * G), gpus(8 * G, 8 * G), [], headroom_mib=1 * G, role="worker")
        self.assertFalse(v["ok"])

    def test_main_may_span_when_no_single_gpu_fits(self):
        v = slots.plan(cand(need=20 * G), gpus(1 * G, 1 * G), [], headroom_mib=1 * G, role="main")
        self.assertTrue(v["ok"], v["reason"])
        self.assertEqual(v["devices"], [0, 1])
        self.assertEqual(v["place"], "")             # "" = no device key: today's split

    def test_main_evicts_workers_lru_first(self):
        loaded = [{"model": "w-old", "role": "worker", "footprint": {0: 6 * G}, "last_used": 1},
                  {"model": "w-new", "role": "worker", "footprint": {0: 6 * G}, "last_used": 9}]
        v = slots.plan(cand(model="chat", need=8 * G), gpus(13 * G, 15 * G), loaded,
                       headroom_mib=1 * G, role="main")
        self.assertFalse(v["ok"])
        self.assertEqual(v["evict"], ["w-old"])      # one eviction is enough, oldest first
        self.assertEqual(v["devices"], [0])

    def test_main_is_never_in_an_evict_list(self):
        loaded = [{"model": "chat", "role": "main", "footprint": {0: 14 * G}, "last_used": 1}]
        v = slots.plan(cand(model="x", need=8 * G), gpus(15 * G, 15 * G), loaded,
                       headroom_mib=1 * G, role="main")
        self.assertFalse(v["ok"])
        self.assertEqual(v["evict"], [])

    def test_cap(self):
        loaded = [{"model": m, "role": "worker", "footprint": {0: G}, "last_used": i}
                  for i, m in enumerate(("a", "b"))]
        v = slots.plan(cand(need=G), gpus(2 * G, 0), loaded, headroom_mib=G, cap=2, role="worker")
        self.assertFalse(v["ok"])
        self.assertIn("2", v["reason"])

    def test_cap_for_main_evicts_one_worker(self):
        loaded = [{"model": m, "role": "worker", "footprint": {0: G}, "last_used": i}
                  for i, m in enumerate(("a", "b"))]
        v = slots.plan(cand(need=G), gpus(2 * G, 0), loaded, headroom_mib=G, cap=2, role="main")
        self.assertEqual(v["evict"], ["a"])

    def test_already_loaded(self):
        loaded = [{"model": "m", "role": "worker", "footprint": {0: G}, "last_used": 1}]
        v = slots.plan(cand(model="m"), gpus(15 * G, 15 * G), loaded, headroom_mib=G)
        self.assertTrue(v["ok"])
        self.assertTrue(v["already"])

    def test_pinned_is_checked_where_it_is_pinned(self):
        c = cand(need=6 * G, pins={"devices": [1], "share": {1: 1.0}})
        v = slots.plan(c, gpus(0, 12 * G), [], headroom_mib=1 * G, role="worker")
        self.assertFalse(v["ok"])                    # GPU0 is empty, but the user pinned GPU1
        self.assertIsNone(v["place"])                # pins are never rewritten

    def test_pinned_split_shares_the_need(self):
        c = cand(need=20 * G, pins={"devices": [0, 1], "share": {0: 0.5, 1: 0.5}})
        v = slots.plan(c, gpus(2 * G, 2 * G), [], headroom_mib=1 * G, role="main")
        self.assertTrue(v["ok"], v["reason"])
        self.assertEqual(v["devices"], [0, 1])
        self.assertEqual(v["footprint"][0], 10 * G + slots.OVERHEAD_MIB // 2)

    def test_cpu_only_pin_needs_no_vram(self):
        c = cand(need=6 * G, pins={"devices": [], "share": {}}, ram_mib=6 * G)
        v = slots.plan(c, gpus(16 * G - 10, 16 * G - 10), [], headroom_mib=G,
                       ram_free_mib=32 * G, role="worker")
        self.assertTrue(v["ok"], v["reason"])
        self.assertEqual(v["devices"], [])

    def test_ram_side_must_fit_too(self):
        c = cand(need=4 * G, ram_mib=40 * G)         # MoE experts kept on the CPU
        v = slots.plan(c, gpus(0, 0), [], headroom_mib=G, ram_free_mib=32 * G, role="worker")
        self.assertFalse(v["ok"])
        self.assertIn("RAM", v["reason"])

    def test_unknown_pins_refuse(self):
        c = cand(pins={"devices": None, "share": {}})
        v = slots.plan(c, gpus(0, 0), [], headroom_mib=G, role="worker")
        self.assertFalse(v["ok"])
        self.assertIn("device", v["reason"])

    def test_no_gpu_information(self):
        v = slots.plan(cand(), [], [], headroom_mib=G, role="worker")
        self.assertFalse(v["ok"])

    def test_free_mib_beats_total_minus_used(self):
        # nvidia-smi's free already takes out the driver's reservation
        g = gpus(0, 2 * G)                           # total - used would say GPU0
        g[0]["free_mib"] = 8 * G
        g[1]["free_mib"] = 9 * G
        v = slots.plan(cand(need=7 * G + 600), g, [], headroom_mib=1 * G, role="main")
        self.assertTrue(v["ok"], v["reason"])
        self.assertEqual(v["devices"], [1])

    def test_the_projector_is_charged_to_the_first_gpu(self):
        # mtmd puts the mmproj on the first GPU backend whatever `device` says
        # (measured: gemma-4-12b pinned to CUDA1 still took 575 MiB on CUDA0)
        v = slots.plan(cand(need=4 * G, first_gpu_mib=344), gpus(12 * G, 0), [],
                       headroom_mib=1 * G, role="worker")
        self.assertEqual(v["devices"], [1])
        self.assertEqual(v["footprint"], {1: 4 * G, 0: slots.OTHER_GPU0_MIB + 344})
        v = slots.plan(cand(need=4 * G, first_gpu_mib=344), gpus(0, 12 * G), [],
                       headroom_mib=1 * G, role="worker")
        self.assertEqual(v["footprint"], {0: 4 * G + 344})

    def test_the_projector_can_be_what_does_not_fit(self):
        tight0 = 16 * G - 1 * G - slots.OTHER_GPU0_MIB - 100
        v = slots.plan(cand(need=4 * G, first_gpu_mib=344), gpus(tight0, 0), [],
                       headroom_mib=1 * G, role="worker")
        self.assertFalse(v["ok"])

    def test_cuda_order_that_differs_from_nvidia_smi(self):
        # CUDA0 is nvidia-smi GPU1 here: the place written must be in CUDA terms,
        # and the stray context lands on CUDA0, wherever nvidia-smi lists it
        m = {0: 1, 1: 0}
        v = slots.plan(cand(need=4 * G), gpus(12 * G, 0), [], headroom_mib=1 * G,
                       role="worker", cmap=m)
        self.assertEqual((v["devices"], v["place"]), ([1], "CUDA0"))
        self.assertEqual(v["footprint"], {1: 4 * G})
        v = slots.plan(cand(need=4 * G, first_gpu_mib=344), gpus(0, 12 * G), [],
                       headroom_mib=1 * G, role="worker", cmap=m)
        self.assertEqual((v["devices"], v["place"]), ([0], "CUDA1"))
        self.assertEqual(v["footprint"], {0: 4 * G, 1: slots.OTHER_GPU0_MIB + 344})


LIST_DEVICES = """Available devices:
  CUDA0: NVIDIA GeForce RTX 5060 Ti (16283 MiB, 15172 MiB free)
  CUDA1: NVIDIA GeForce RTX 5080 (16275 MiB, 14985 MiB free)
"""
SMI = [{"index": 0, "name": "NVIDIA GeForce RTX 5080", "total_mib": 16303, "used_mib": 0},
       {"index": 1, "name": "NVIDIA GeForce RTX 5060 Ti", "total_mib": 16311, "used_mib": 0}]


class CudaMap(unittest.TestCase):
    def test_maps_by_name_when_the_orders_differ(self):
        self.assertEqual(slots.cuda_map(LIST_DEVICES, SMI), {0: 1, 1: 0})

    def test_identical_cards_keep_their_order(self):
        out = ("  CUDA0: NVIDIA GeForce RTX 3090 (24000 MiB, 1 MiB free)\n"
               "  CUDA1: NVIDIA GeForce RTX 3090 (24000 MiB, 1 MiB free)\n")
        smi = [{"index": i, "name": "NVIDIA GeForce RTX 3090"} for i in (0, 1)]
        self.assertEqual(slots.cuda_map(out, smi), {0: 0, 1: 1})

    def test_unmatched_is_none(self):
        self.assertIsNone(slots.cuda_map("  CUDA0: Something Else (1 MiB, 1 MiB free)\n", SMI))
        self.assertIsNone(slots.cuda_map("", SMI))

    def test_pins_and_places_translate(self):
        m = {0: 1, 1: 0}
        p = slots.to_smi({"devices": [0], "share": {0: 1.0}}, m)
        self.assertEqual(p, {"devices": [1], "share": {1: 1.0}})
        self.assertEqual(slots.to_smi({"devices": [5], "share": {5: 1.0}}, m),
                         {"devices": None, "share": {}})
        self.assertIsNone(slots.to_smi(None, m))
        self.assertEqual(slots.to_smi({"devices": [], "share": {}}, m),
                         {"devices": [], "share": {}})
        self.assertEqual(slots.cuda_name(1, m), "CUDA0")


if __name__ == "__main__":
    unittest.main()
