import conftest_paths  # noqa: F401
import unittest

import telemetry


class GpuTelemetryCacheTest(unittest.TestCase):
    def test_reuses_probe_result_until_ttl_expires(self):
        calls = []
        clock = iter((10.0, 12.0, 21.0)).__next__
        cache = telemetry.TimedCache(
            lambda: calls.append(True) or [{"used": len(calls)}],
            ttl=10,
            clock=clock,
        )

        self.assertEqual(cache.get(), [{"used": 1}])
        self.assertEqual(cache.get(), [{"used": 1}])
        self.assertEqual(cache.get(), [{"used": 2}])
        self.assertEqual(len(calls), 2)

    def test_returns_copies_so_request_code_cannot_poison_cache(self):
        cache = telemetry.TimedCache(lambda: [{"used": 1}], ttl=10,
                                     clock=lambda: 1.0)
        first = cache.get()
        first[0]["used"] = 999
        self.assertEqual(cache.get(), [{"used": 1}])


if __name__ == "__main__":
    unittest.main()
