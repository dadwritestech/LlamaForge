import conftest_paths  # noqa: F401
import os
import tempfile
import unittest
from unittest import mock

import gpuprobe


VULKAN_AMD = """\
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon RX 7900 XTX (AMD proprietary driver) | uma: 0 | fp16: 1
Available devices:
  Vulkan0: AMD Radeon RX 7900 XTX (24560 MiB, 23012 MiB free)
"""

BOTH = """\
Available devices:
  ROCm0: AMD Radeon RX 7800 XT (16368 MiB, 16000 MiB free)
  Vulkan0: AMD Radeon RX 7800 XT (16384 MiB, 15990 MiB free)
  Vulkan1: AMD Radeon(TM) Graphics (2048 MiB, 1900 MiB free)
"""


class ListDevicesTest(unittest.TestCase):
    def test_parses_a_vulkan_card(self):
        self.assertEqual(gpuprobe.parse_list_devices(VULKAN_AMD), [
            {"backend": "Vulkan", "n": 0, "name": "AMD Radeon RX 7900 XTX",
             "total": 24560, "free": 23012}])

    def test_cpu_only_build_lists_nothing(self):
        self.assertEqual(gpuprobe.parse_list_devices("Available devices:\n"), [])
        self.assertEqual(gpuprobe.parse_list_devices(""), [])

    def test_a_card_seen_by_two_backends_counts_once(self):
        tele = gpuprobe.telemetry(gpuprobe.parse_list_devices(BOTH))
        self.assertEqual([g["name"] for g in tele], ["AMD Radeon RX 7800 XT"])
        self.assertEqual(tele[0]["via"], "ROCm")

    def test_telemetry_has_the_nvidia_smi_shape(self):
        tele = gpuprobe.telemetry(gpuprobe.parse_list_devices(VULKAN_AMD))
        self.assertEqual(tele, [{"index": 0, "name": "AMD Radeon RX 7900 XTX",
                                 "used": 1548, "total": 24560, "util": None,
                                 "temp": None, "via": "Vulkan"}])

    def test_an_igpu_beside_real_cards_is_dropped(self):
        # real b12032 Vulkan output on a Ryzen box with two NVIDIA cards: the iGPU
        # reports 16 GB of shared system RAM, which isn't VRAM
        out = ("Available devices:\n"
               "  Vulkan0: AMD Radeon(TM) Graphics (16412 MiB, 15591 MiB free)\n"
               "  Vulkan1: NVIDIA GeForce RTX 5060 Ti (16050 MiB, 15282 MiB free)\n"
               "  Vulkan2: NVIDIA GeForce RTX 5080 (15977 MiB, 15209 MiB free)\n")
        tele = gpuprobe.telemetry(gpuprobe.parse_list_devices(out))
        self.assertEqual([(g["index"], g["name"]) for g in tele],
                         [(0, "NVIDIA GeForce RTX 5060 Ti"), (1, "NVIDIA GeForce RTX 5080")])

    def test_an_igpu_on_its_own_is_kept(self):
        out = "  Vulkan0: AMD Radeon 890M Graphics (16384 MiB, 15000 MiB free)\n"
        self.assertEqual(len(gpuprobe.telemetry(gpuprobe.parse_list_devices(out))), 1)

    def test_integrated_names(self):
        for name in ("AMD Radeon(TM) Graphics", "AMD Radeon 780M Graphics",
                     "AMD Radeon 8060S Graphics", "Intel(R) UHD Graphics 770",
                     "Intel(R) Iris(R) Xe Graphics", "Intel(R) Arc(TM) Graphics"):
            self.assertTrue(gpuprobe.integrated(name), name)
        for name in ("AMD Radeon RX 7900 XTX", "Intel(R) Arc(TM) A770 Graphics",
                     "Intel(R) Arc(TM) B580 Graphics", "AMD Radeon PRO W7900",
                     "NVIDIA GeForce RTX 5080"):
            self.assertFalse(gpuprobe.integrated(name), name)

    def test_software_vulkan_is_never_a_gpu(self):
        out = "  Vulkan0: llvmpipe (LLVM 17.0.6, 256 bits) (32000 MiB, 32000 MiB free)\n"
        self.assertEqual(gpuprobe.telemetry(gpuprobe.parse_list_devices(out)), [])

    def test_names_with_brackets_keep_their_brackets(self):
        out = "  SYCL0: Intel(R) Arc(TM) A770 Graphics (16225 MiB, 15000 MiB free)\n"
        self.assertEqual(gpuprobe.parse_list_devices(out)[0]["name"],
                         "Intel(R) Arc(TM) A770 Graphics")


class AmdSysfsTest(unittest.TestCase):
    def _card(self, root, n, vendor, total=None, used=None, product=None):
        d = os.path.join(root, f"card{n}", "device")
        os.makedirs(d)
        for name, val in (("vendor", vendor), ("mem_info_vram_total", total),
                          ("mem_info_vram_used", used), ("product_name", product)):
            if val is not None:
                with open(os.path.join(d, name), "w") as f:
                    f.write(f"{val}\n")

    def test_reads_amdgpu_vram_and_skips_other_vendors(self):
        with tempfile.TemporaryDirectory() as root:
            self._card(root, 0, "0x8086")                                  # Intel iGPU
            self._card(root, 1, "0x1002", 17163091968, 524288000, "Radeon RX 7800 XT")
            os.makedirs(os.path.join(root, "card1-DP-1"))                  # a connector, not a card
            tele = gpuprobe.amdgpu_sysfs(root)
        self.assertEqual(tele, [{"index": 0, "name": "AMD Radeon RX 7800 XT",
                                 "used": 500, "total": 16368, "util": None,
                                 "temp": None, "via": "amdgpu"}])

    def test_no_drm_dir_is_empty(self):
        self.assertEqual(gpuprobe.amdgpu_sysfs("/nonexistent/drm"), [])


class OtherGpusTest(unittest.TestCase):
    def test_prefers_the_engine_and_falls_back_to_sysfs_on_linux(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            sbin = f.name
        self.addCleanup(os.remove, sbin)
        with mock.patch.object(gpuprobe, "_list_devices", return_value=VULKAN_AMD):
            self.assertEqual(gpuprobe.other_gpus(sbin)[0]["via"], "Vulkan")
        sysfs = [{"index": 0, "name": "AMD GPU", "used": 1, "total": 2,
                  "util": None, "temp": None, "via": "amdgpu"}]
        with mock.patch.object(gpuprobe, "_list_devices", return_value=""), \
             mock.patch.object(gpuprobe, "IS_LINUX", True), \
             mock.patch.object(gpuprobe, "amdgpu_sysfs", return_value=sysfs):
            self.assertEqual(gpuprobe.other_gpus(sbin), sysfs)

    def test_missing_binary_is_not_run(self):
        with mock.patch.object(gpuprobe, "_list_devices") as ld, \
             mock.patch.object(gpuprobe, "IS_LINUX", False):
            self.assertEqual(gpuprobe.other_gpus("/no/such/llama-server"), [])
        ld.assert_not_called()


class RoutesFallbackTest(unittest.TestCase):
    def test_no_nvidia_smi_uses_the_other_probe_and_elastic_ignores_it(self):
        import routes
        amd = gpuprobe.telemetry(gpuprobe.parse_list_devices(VULKAN_AMD))
        with mock.patch.object(routes.osplat, "IS_MAC", False), \
             mock.patch.object(routes.subprocess, "check_output",
                               side_effect=FileNotFoundError("nvidia-smi")), \
             mock.patch.object(routes._OTHER_GPUS, "get", return_value=amd):
            self.assertEqual(routes._gpu_telemetry(), amd)
            self.assertEqual(routes.total_vram_mib(), 24560)
            self.assertEqual(routes._el_gpus(), {})      # Elastic VRAM is NVIDIA-only

    def test_nvidia_answer_skips_the_other_probe(self):
        import routes
        with mock.patch.object(routes.osplat, "IS_MAC", False), \
             mock.patch.object(routes.subprocess, "check_output",
                               return_value="0, RTX, 100, 16000, 1, 40\n"), \
             mock.patch.object(routes._OTHER_GPUS, "get") as other:
            self.assertEqual(routes._gpu_telemetry()[0]["name"], "RTX")
        other.assert_not_called()

    def test_nothing_anywhere_keeps_the_error(self):
        import routes
        with mock.patch.object(routes.osplat, "IS_MAC", False), \
             mock.patch.object(routes.subprocess, "check_output",
                               side_effect=FileNotFoundError("nvidia-smi")), \
             mock.patch.object(routes._OTHER_GPUS, "get", return_value=[]):
            self.assertIn("error", routes._gpu_telemetry()[0])


if __name__ == "__main__":
    unittest.main()
