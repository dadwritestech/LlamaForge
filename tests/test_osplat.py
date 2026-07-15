import conftest_paths  # noqa: F401
import unittest
import osplat

CPUINFO = """\
processor\t: 0
model name\t: AMD Ryzen 9 7950X 16-Core Processor
physical id\t: 0
core id\t\t: 0
flags\t\t: fpu vme avx2 avx512f avx512vnni
processor\t: 1
model name\t: AMD Ryzen 9 7950X 16-Core Processor
physical id\t: 0
core id\t\t: 0
flags\t\t: fpu vme avx2 avx512f avx512vnni
processor\t: 2
model name\t: AMD Ryzen 9 7950X 16-Core Processor
physical id\t: 0
core id\t\t: 1
flags\t\t: fpu vme avx2 avx512f avx512vnni
"""

CPUINFO_NO512 = """\
processor\t: 0
model name\t: Intel(R) Core(TM) i7-9700K
flags\t\t: fpu vme avx2
"""

VMSTAT = """\
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                              123456.
Pages active:                            222222.
Pages inactive:                          100000.
"""


class TestCpuinfo(unittest.TestCase):
    def test_parses_name_cores_threads_avx512(self):
        c = osplat.parse_proc_cpuinfo(CPUINFO)
        self.assertEqual(c["name"], "AMD Ryzen 9 7950X 16-Core Processor")
        self.assertEqual(c["threads"], 3)
        self.assertEqual(c["cores"], 2)          # (0,0) and (0,1)
        self.assertTrue(c["avx512"])

    def test_no_avx512_and_no_topology_falls_back_to_threads(self):
        c = osplat.parse_proc_cpuinfo(CPUINFO_NO512)
        self.assertFalse(c["avx512"])
        self.assertEqual(c["cores"], 1)

    def test_empty(self):
        c = osplat.parse_proc_cpuinfo("")
        self.assertEqual(c["threads"], None)
        self.assertFalse(c["avx512"])


class TestMac(unittest.TestCase):
    def test_vm_stat_free_bytes(self):
        free = osplat.parse_vm_stat(VMSTAT)
        self.assertEqual(free, (123456 + 100000) * 16384)

    def test_apple_gpu_budget(self):
        g = osplat.apple_silicon_gpu(32 * 1024**3, free_bytes=16 * 1024**3)
        self.assertEqual(g["total"], int(32 * 1024 * osplat.METAL_BUDGET))
        self.assertLessEqual(g["used"], g["total"])
        self.assertIn("Apple Silicon", g["name"])

    def test_apple_gpu_used_never_negative(self):
        g = osplat.apple_silicon_gpu(8 * 1024**3, free_bytes=16 * 1024**3)
        self.assertEqual(g["used"], 0)


class TestPosixPort(unittest.TestCase):
    def test_parse_lsof_pids(self):
        self.assertEqual(osplat.parse_lsof_pids("123\n456\n"), [123, 456])
        self.assertEqual(osplat.parse_lsof_pids(""), [])
        self.assertEqual(osplat.parse_lsof_pids("garbage\n"), [])


class TestPkg(unittest.TestCase):
    def test_install_hint(self):
        self.assertIn("apt-get install", osplat.linux_install_hint("apt-get", "cmake"))
        self.assertIn("pacman -S", osplat.linux_install_hint("pacman", "cmake"))
        self.assertEqual(osplat.linux_install_hint("", "cmake"), "")


if __name__ == "__main__":
    unittest.main()
