"""skills/llamaforge/SKILL.md (the agent skill, also the Claude Code plugin) stays in step
with the MCP server and the HTTP routes it tells agents to use."""
import conftest_paths  # noqa: F401
import json
import os
import re
import unittest

import mcp_server
import routes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILL = os.path.join(ROOT, "skills", "llamaforge", "SKILL.md")


def _read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


class Skill(unittest.TestCase):
    def setUp(self):
        self.text = _read(SKILL)

    def test_frontmatter(self):
        m = re.match(r"^---\n(.*?)\n---\n", self.text, re.S)
        self.assertTrue(m)
        fm = dict(line.split(": ", 1) for line in m.group(1).splitlines())
        self.assertEqual(fm["name"], "llamaforge")
        self.assertLess(len(fm["description"]), 1024)

    def test_every_mcp_tool_is_explained(self):
        for name in mcp_server.setup_info(python="python", script="x.py")["tools"]:
            self.assertTrue(f"`{name}`" in self.text, f"SKILL.md never mentions `{name}`")

    def test_every_endpoint_it_names_exists(self):
        named = set(re.findall(r"`(GET|POST) (/api/[a-z/_-]+)", self.text))
        named |= {("POST", p) for p in re.findall(r"POST (/api/[a-z/_-]+)", self.text)}
        self.assertGreater(len(named), 10)
        for method, path in named:
            table = routes.GET_ROUTES if method == "GET" else routes.POST_ROUTES
            self.assertIn(path, table, f"{method} {path}")

    def test_plugin_manifests(self):
        plugin = json.loads(_read(os.path.join(ROOT, ".claude-plugin", "plugin.json")))
        market = json.loads(_read(os.path.join(ROOT, ".claude-plugin", "marketplace.json")))
        self.assertEqual(plugin["name"], "llamaforge")
        self.assertEqual(plugin["license"], "MIT")
        self.assertEqual(market["plugins"][0]["name"], plugin["name"])


if __name__ == "__main__":
    unittest.main()
