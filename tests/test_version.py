import conftest_paths  # noqa: F401
import unittest
from unittest import mock

import routes, version


class VersionTest(unittest.TestCase):
    def test_is_semver(self):
        self.assertRegex(version.VERSION, r"^\d+\.\d+\.\d+$")

    def test_state_reports_it(self):
        # No live router or GPU poll: state() and telemetry are stubbed.
        with mock.patch.object(routes.REGISTRY, "state", return_value={"models": []}), \
             mock.patch.object(routes._GPU_TELEMETRY, "get", return_value=[]):
            self.assertEqual(routes.get_state(routes.Req())[1]["version"],
                             version.VERSION)


if __name__ == "__main__":
    unittest.main()
