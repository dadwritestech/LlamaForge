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

    def test_parse_ss_pids(self):
        line = ('LISTEN 0 4096 127.0.0.1:8080 0.0.0.0:* '
                'users:(("llama-server",pid=4321,fd=3))\n')
        self.assertEqual(osplat.parse_ss_pids(line), [4321])
        self.assertEqual(osplat.parse_ss_pids("LISTEN 0 4096 *:8080 *:*\n"), [])

    def test_falls_back_when_lsof_is_missing(self):
        """Arch, minimal Debian/Fedora and containers ship no lsof (05 #3)."""
        from unittest import mock
        seen = []
        def fake(cmd, timeout=10):
            seen.append(cmd[0])
            return {"ss": 'LISTEN 0 1 *:8080 *:* users:(("python3",pid=77,fd=5))',
                    "fuser": " 88"}.get(cmd[0], "")
        with mock.patch.object(osplat, "run_text", side_effect=fake):
            self.assertEqual(osplat.pid_on_port_posix(8080, tool="ss"), 77)
            self.assertEqual(osplat.pid_on_port_posix(8080, tool="fuser"), 88)
            with mock.patch("shutil.which", side_effect=lambda t: t == "ss"):
                self.assertEqual(osplat.port_tool(), "ss")
                self.assertEqual(osplat.pid_on_port_posix(8080), 77)
            with mock.patch("shutil.which", return_value=None):
                self.assertIsNone(osplat.port_tool())
                self.assertIsNone(osplat.pid_on_port_posix(8080))
        self.assertNotIn("lsof", seen)


class TestPkg(unittest.TestCase):
    def test_install_hint(self):
        self.assertIn("apt-get install", osplat.linux_install_hint("apt-get", "cmake"))
        self.assertIn("pacman -S", osplat.linux_install_hint("pacman", "cmake"))
        self.assertEqual(osplat.linux_install_hint("", "cmake"), "")


if __name__ == "__main__":
    unittest.main()
