"""Test that lint configuration files exist and are valid."""
import json
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RuffConfigTest(unittest.TestCase):
    """Verify ruff.toml exists with E and F rules enabled."""

    def test_ruff_toml_exists(self):
        self.assertTrue(
            os.path.isfile(os.path.join(ROOT, "ruff.toml")),
            "ruff.toml must exist at repo root",
        )

    def test_ruff_toml_has_select_ef(self):
        """ruff.toml must select E (pycodestyle) and F (pyflakes) rules."""
        path = os.path.join(ROOT, "ruff.toml")
        with open(path) as f:
            content = f.read()
        self.assertIn("select", content.lower(), "ruff.toml must have a [lint] section with 'select'")
        # Check that E and F are in the select list
        match = re.search(r"select\s*=\s*\[(.*?)\]", content, re.DOTALL)
        self.assertIsNotNone(match, "ruff.toml must have a select = [...] list")
        rules = match.group(1)
        self.assertIn("E", rules, "E (pycodestyle) rules must be selected")
        self.assertIn("F", rules, "F (pyflakes) rules must be selected")

    def test_ruff_toml_targets_backend(self):
        """ruff.toml should target the backend directory."""
        path = os.path.join(ROOT, "ruff.toml")
        with open(path) as f:
            content = f.read()
        # ruff.toml at root applies to the whole repo; backend is the Python source
        self.assertIn("backend", content.lower())


class ESLintConfigTest(unittest.TestCase):
    """Verify .eslintrc.json exists and bans .innerHTML outside core.js."""

    def test_eslintrc_exists(self):
        self.assertTrue(
            os.path.isfile(os.path.join(ROOT, ".eslintrc.json")),
            ".eslintrc.json must exist at repo root",
        )

    def test_eslintrc_is_valid_json(self):
        path = os.path.join(ROOT, ".eslintrc.json")
        with open(path) as f:
            config = json.load(f)
        self.assertIsInstance(config, dict, ".eslintrc.json must be a JSON object")

    def test_eslintrc_bans_innerhtml(self):
        """The eslint config must ban .innerHTML assignments via no-restricted-syntax."""
        path = os.path.join(ROOT, ".eslintrc.json")
        with open(path) as f:
            config = json.load(f)
        rules = config.get("rules", {})
        restricted = rules.get("no-restricted-syntax", [])
        self.assertIsInstance(restricted, list, "no-restricted-syntax must be a list")
        # Find the innerHTML selector
        found = False
        for item in restricted:
            if isinstance(item, dict) and item.get("selector") == "AssignmentExpression[left.type='MemberExpression'][left.property.name='innerHTML']":
                found = True
                break
        self.assertTrue(
            found,
            ".eslintrc.json must have a no-restricted-syntax rule banning .innerHTML",
        )

    def test_eslintrc_allows_core_js(self):
        """core.js should be exempted from the .innerHTML ban."""
        path = os.path.join(ROOT, ".eslintrc.json")
        with open(path) as f:
            config = json.load(f)
        overrides = config.get("overrides", [])
        # Find the override for core.js
        core_override = None
        for ov in overrides:
            files = ov.get("files", [])
            if any("core.js" in f for f in files):
                core_override = ov
                break
        self.assertIsNotNone(
            core_override,
            ".eslintrc.json must have an override for core.js to allow .innerHTML",
        )
        # The core.js override must disable no-restricted-syntax
        core_rules = core_override.get("rules", {})
        self.assertIn(
            "no-restricted-syntax",
            core_rules,
            "core.js override must disable no-restricted-syntax",
        )
        self.assertEqual(
            core_rules["no-restricted-syntax"], "off",
            "core.js override must set no-restricted-syntax to 'off'",
        )


class CIWorkflowLintJobTest(unittest.TestCase):
    """Verify the CI workflow includes a lint job."""

    def test_ci_yml_exists(self):
        path = os.path.join(ROOT, ".github", "workflows", "ci.yml")
        self.assertTrue(os.path.isfile(path), ".github/workflows/ci.yml must exist")

    def test_ci_has_lint_job(self):
        """The CI workflow must include a lint job."""
        path = os.path.join(ROOT, ".github", "workflows", "ci.yml")
        with open(path) as f:
            content = f.read()
        self.assertIn("lint", content.lower(), "CI workflow must include a 'lint' job")

    def test_ci_lint_runs_ruff(self):
        """The lint job must run ruff check on backend."""
        path = os.path.join(ROOT, ".github", "workflows", "ci.yml")
        with open(path) as f:
            content = f.read()
        self.assertIn("ruff", content, "CI lint job must run ruff check")

    def test_ci_lint_runs_eslint(self):
        """The lint job must run eslint on web/js."""
        path = os.path.join(ROOT, ".github", "workflows", "ci.yml")
        with open(path) as f:
            content = f.read()
        self.assertIn("eslint", content, "CI lint job must run eslint")


if __name__ == "__main__":
    unittest.main()
