import conftest_paths  # noqa: F401
import os, unittest
from unittest import mock
import vllm_argspec

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "vllm_help.txt")


class TestParse(unittest.TestCase):
    def setUp(self):
        with open(FIX, encoding="utf-8") as f:
            self.items = vllm_argspec.parse_help(f.read())
        self.by_key = {it["key"]: it for it in self.items}

    def test_reserved_flags_excluded_from_editable(self):
        editable = {it["key"] for it in self.items if not it["reserved"]}
        self.assertNotIn("host", editable)
        self.assertNotIn("port", editable)
        self.assertNotIn("help", editable)

    def test_int_knob(self):
        tp = self.by_key["tensor-parallel-size"]
        self.assertEqual(tp["type"], "int")
        self.assertEqual(tp["default"], "1")

    def test_float_knob(self):
        g = self.by_key["gpu-memory-utilization"]
        self.assertEqual(g["type"], "float")
        self.assertEqual(g["default"], "0.9")

    def test_enum_knob_choices(self):
        d = self.by_key["dtype"]
        self.assertEqual(d["type"], "enum")
        self.assertIn("bfloat16", d["options"])

    def test_quantization_includes_nvfp4(self):
        q = self.by_key["quantization"]
        self.assertEqual(q["type"], "enum")
        self.assertIn("nvfp4", q["options"])

    def test_bool_knob(self):
        e = self.by_key["enforce-eager"]
        self.assertEqual(e["type"], "bool")


class TestBuildSchema(unittest.TestCase):
    def test_build_schema_runs_help_in_wsl_and_groups(self):
        with open(FIX, encoding="utf-8") as f:
            help_text = f.read()
        with mock.patch("wsl.run", return_value=(0, help_text, "")):
            schema = vllm_argspec.build_schema(distro="Ubuntu", venv="~/x")
        self.assertIn("groups", schema)
        self.assertGreater(schema["count"], 0)

    def test_build_schema_error_when_help_fails(self):
        with mock.patch("wsl.run", return_value=(1, "", "not found")):
            schema = vllm_argspec.build_schema(distro="Ubuntu", venv="~/x")
        self.assertIn("error", schema)


if __name__ == "__main__":
    unittest.main()
