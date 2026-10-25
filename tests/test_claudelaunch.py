import conftest_paths  # noqa: F401
import os
import tempfile
import unittest

import claudelaunch as cl


def which_from(found):
    return lambda name: found.get(name)


class TestLocate(unittest.TestCase):
    def test_path_wins(self):
        p = cl.locate(which=which_from({"claude": "/usr/bin/claude"}), home="/h",
                      is_win=False, isfile=lambda p: True, env={})
        self.assertEqual(p, "/usr/bin/claude")

    def test_native_installer_under_local_bin(self):
        want = os.path.join("/h", ".local", "bin", "claude")
        p = cl.locate(which=which_from({}), home="/h", is_win=False,
                      isfile=lambda p: p == want, env={})
        self.assertEqual(p, want)

    def test_windows_native_exe(self):
        want = os.path.join("C:/u", ".local", "bin", "claude.exe")
        p = cl.locate(which=which_from({}), home="C:/u", is_win=True,
                      isfile=lambda p: p == want, env={})
        self.assertEqual(p, want)

    def test_windows_npm_global(self):
        want = os.path.join("C:/u/AppData/Roaming", "npm", "claude.cmd")
        p = cl.locate(which=which_from({}), home="C:/u", is_win=True,
                      isfile=lambda p: p == want, env={"APPDATA": "C:/u/AppData/Roaming"})
        self.assertEqual(p, want)

    def test_old_local_install(self):
        want = os.path.join("/h", ".claude", "local", "claude")
        p = cl.locate(which=which_from({}), home="/h", is_win=False,
                      isfile=lambda p: p == want, env={})
        self.assertEqual(p, want)

    def test_windows_desktop_app_bundle_newest_version(self):
        appdata = tempfile.mkdtemp()
        root = os.path.join(appdata, "Claude", "claude-code")
        for ver, h in (("2.1.9", "aa"), ("2.1.10", "bb"), ("junk", "cc")):
            os.makedirs(os.path.join(root, ver, h))
            open(os.path.join(root, ver, h, "claude.exe"), "w").close()
        p = cl.locate(which=which_from({}), home=appdata, is_win=True,
                      isfile=os.path.isfile, env={"APPDATA": appdata})
        self.assertEqual(p, os.path.join(root, "2.1.10", "bb", "claude.exe"))

    def test_standalone_install_beats_desktop_bundle(self):
        appdata = tempfile.mkdtemp()
        os.makedirs(os.path.join(appdata, "Claude", "claude-code", "2.1.9", "aa"))
        open(os.path.join(appdata, "Claude", "claude-code", "2.1.9", "aa", "claude.exe"), "w").close()
        want = os.path.join(appdata, "npm", "claude.cmd")
        os.makedirs(os.path.dirname(want))
        open(want, "w").close()
        p = cl.locate(which=which_from({}), home=appdata, is_win=True,
                      isfile=os.path.isfile, env={"APPDATA": appdata})
        self.assertEqual(p, want)

    def test_nothing(self):
        self.assertEqual(cl.locate(which=which_from({}), home="/h", is_win=False,
                                   isfile=lambda p: False, env={}), "")


class TestVersion(unittest.TestCase):
    def test_parses(self):
        self.assertEqual(cl.version("c", run=lambda argv: (0, "2.1.7 (Claude Code)\n")), "2.1.7")

    def test_failure_is_blank(self):
        self.assertEqual(cl.version("c", run=lambda argv: (1, "boom")), "")
        self.assertEqual(cl.version("", run=lambda argv: (0, "2.0.0")), "")


class TestEnv(unittest.TestCase):
    BASE = {"PATH": "/bin", "ANTHROPIC_API_KEY": "sk-real", "ANTHROPIC_BASE_URL": "https://x",
            "CLAUDE_CODE_USE_BEDROCK": "1", "CLAUDE_CODE_USE_VERTEX": "1",
            "ANTHROPIC_CUSTOM_HEADERS": "a: b", "HOME": "/h"}

    def env(self, token="k", small="tiny"):
        return cl.session_env(self.BASE, "http://127.0.0.1:8090", token, "big", small)

    def test_points_at_panel_and_model(self):
        e = self.env()
        self.assertEqual(e["ANTHROPIC_BASE_URL"], "http://127.0.0.1:8090")
        self.assertEqual(e["ANTHROPIC_AUTH_TOKEN"], "k")
        self.assertEqual(e["ANTHROPIC_MODEL"], "big")
        self.assertEqual(e["ANTHROPIC_DEFAULT_OPUS_MODEL"], "big")
        self.assertEqual(e["ANTHROPIC_DEFAULT_SONNET_MODEL"], "big")
        self.assertEqual(e["ANTHROPIC_DEFAULT_HAIKU_MODEL"], "tiny")
        self.assertEqual(e["ANTHROPIC_SMALL_FAST_MODEL"], "tiny")
        self.assertEqual(e["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"], "1")
        self.assertTrue(int(e["API_TIMEOUT_MS"]) >= 600000)

    def test_strips_cloud_settings(self):
        e = self.env()
        for k in ("ANTHROPIC_API_KEY", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX",
                  "ANTHROPIC_CUSTOM_HEADERS"):
            self.assertNotIn(k, e)
        self.assertEqual(e["PATH"], "/bin")
        self.assertEqual(e["HOME"], "/h")

    def test_does_not_mutate_base(self):
        self.env()
        self.assertEqual(self.BASE["ANTHROPIC_API_KEY"], "sk-real")

    def test_defaults(self):
        e = cl.session_env({}, "http://127.0.0.1:8090", "", "big", None)
        self.assertEqual(e["ANTHROPIC_AUTH_TOKEN"], "llamaforge")
        self.assertEqual(e["ANTHROPIC_DEFAULT_HAIKU_MODEL"], "big")

    def test_changes_lists_unsets(self):
        sets, unsets = cl.env_changes(self.BASE, "http://e", "k", "big", "tiny")
        self.assertIn("ANTHROPIC_API_KEY", unsets)
        self.assertNotIn("ANTHROPIC_BASE_URL", unsets)   # it is set, not unset
        self.assertEqual(sets["ANTHROPIC_BASE_URL"], "http://e")


class TestChecks(unittest.TestCase):
    def run_checks(self, **kw):
        info = dict(cli="/c", shim_enabled=True, loaded=True, n_ctx=131072,
                    settings={}, model="big", small="big")
        info.update(kw)
        return cl.checks(**info)

    def level(self, items, cid):
        return next(i["level"] for i in items if i["id"] == cid)

    def test_all_good(self):
        items, ready = self.run_checks()
        self.assertTrue(ready)
        self.assertTrue(all(i["level"] == "ok" for i in items))

    def test_missing_cli_blocks(self):
        items, ready = self.run_checks(cli="")
        self.assertFalse(ready)
        self.assertEqual(self.level(items, "cli"), "error")

    def test_shim_off_blocks(self):
        items, ready = self.run_checks(shim_enabled=False)
        self.assertFalse(ready)
        self.assertEqual(self.level(items, "shim"), "error")

    def test_ctx_thresholds(self):
        self.assertEqual(self.level(self.run_checks(n_ctx=16384)[0], "ctx"), "error")
        self.assertFalse(self.run_checks(n_ctx=16384)[1])
        self.assertEqual(self.level(self.run_checks(n_ctx=40000)[0], "ctx"), "warn")
        self.assertTrue(self.run_checks(n_ctx=40000)[1])
        self.assertEqual(self.level(self.run_checks(n_ctx=65536)[0], "ctx"), "ok")
        self.assertEqual(self.level(self.run_checks(n_ctx=None)[0], "ctx"), "warn")

    def test_no_jinja_blocks(self):
        items, ready = self.run_checks(settings={"no-jinja": "true"})
        self.assertFalse(ready)
        self.assertEqual(self.level(items, "jinja"), "error")

    def test_not_loaded_warns(self):
        items, ready = self.run_checks(loaded=False)
        self.assertTrue(ready)
        self.assertEqual(self.level(items, "loaded"), "warn")

    def test_small_text_names_worker(self):
        items, _ = self.run_checks(small="tiny")
        s = next(i for i in items if i["id"] == "small")
        self.assertIn("tiny", s["text"])


class TestChooseSmall(unittest.TestCase):
    def test_smallest_other_loaded(self):
        rows = [{"id": "big", "status": "loaded", "file_gib": 20},
                {"id": "w2", "status": "loaded", "file_gib": 4},
                {"id": "w1", "status": "loaded", "file_gib": 1},
                {"id": "off", "status": "unloaded", "file_gib": 0.5}]
        self.assertEqual(cl.choose_small(rows, "big"), "w1")

    def test_falls_back_to_main(self):
        self.assertEqual(cl.choose_small([{"id": "big", "status": "loaded"}], "big"), "big")


class TestLaunchPlan(unittest.TestCase):
    ENV = {"PATH": "/bin", "ANTHROPIC_MODEL": "big", "ANTHROPIC_API_KEY": "sk-real"}
    SETS = {"ANTHROPIC_MODEL": "big", "ANTHROPIC_BASE_URL": "http://127.0.0.1:8090"}
    UNSETS = ["ANTHROPIC_API_KEY"]

    def plan(self, os_name, found=None, folder="/work/my repo"):
        return cl.launch_plan(os_name, "/bin/claude", folder, self.ENV, self.SETS, self.UNSETS,
                              script_dir="/tmp/lfx", which=which_from(found or {}),
                              comspec="C:\\Windows\\system32\\cmd.exe")

    def test_windows_new_console_with_env(self):
        p = self.plan("win", folder="D:\\code\\proj")
        self.assertEqual(p["kind"], "spawn")
        self.assertEqual(p["argv"], ["C:\\Windows\\system32\\cmd.exe", "/k", "/bin/claude"])
        self.assertEqual(p["cwd"], "D:\\code\\proj")
        self.assertIs(p["env"], self.ENV)
        self.assertEqual(p["creationflags"], cl.CREATE_NEW_CONSOLE)
        self.assertIsNone(p["script"])
        self.assertIn("$env:ANTHROPIC_MODEL='big'", p["command"])
        self.assertIn("Remove-Item Env:ANTHROPIC_API_KEY", p["command"])
        self.assertIn("Set-Location -LiteralPath 'D:\\code\\proj'", p["command"])

    def test_mac_self_deleting_command_file(self):
        p = self.plan("mac")
        self.assertEqual(p["kind"], "spawn")
        self.assertTrue(p["script_path"].endswith(".command"))
        self.assertEqual(p["argv"], ["open", "-a", "Terminal", p["script_path"]])
        s = p["script"]
        self.assertTrue(s.startswith("#!/bin/sh\n"))
        self.assertIn('rm -f -- "$0"', s.splitlines()[1])
        self.assertIn("unset ANTHROPIC_API_KEY", s)
        self.assertIn("export ANTHROPIC_MODEL=big", s)
        self.assertIn("cd '/work/my repo' || exit 1", s)
        self.assertIn("/bin/claude", s)
        self.assertNotIn("sk-real", s)

    def test_linux_first_terminal(self):
        p = self.plan("linux", found={"gnome-terminal": "/usr/bin/gnome-terminal",
                                      "xterm": "/usr/bin/xterm"})
        self.assertEqual(p["argv"], ["/usr/bin/gnome-terminal", "--", p["script_path"]])
        p = self.plan("linux", found={"xterm": "/usr/bin/xterm"})
        self.assertEqual(p["argv"], ["/usr/bin/xterm", "-e", p["script_path"]])

    def test_linux_no_terminal_is_manual(self):
        p = self.plan("linux")
        self.assertEqual(p["kind"], "manual")
        self.assertIn("cd '/work/my repo'", p["command"])
        self.assertIn("unset ANTHROPIC_API_KEY", p["command"])

    def test_quotes_hostile_values(self):
        sets = {"ANTHROPIC_MODEL": "a'b; rm -rf ~"}
        p = cl.launch_plan("linux", "/bin/claude", "/w", {}, sets, [], script_dir="/t",
                           which=which_from({"xterm": "/x"}), comspec="cmd")
        self.assertIn("export ANTHROPIC_MODEL='a'\"'\"'b; rm -rf ~'", p["script"])
        w = cl.launch_plan("win", "C:\\c.exe", "C:\\w", {}, sets, [], script_dir="",
                           which=which_from({}), comspec="cmd")
        self.assertIn("$env:ANTHROPIC_MODEL='a''b; rm -rf ~'", w["command"])


class TestPickFolder(unittest.TestCase):
    def test_windows_powershell_topmost(self):
        argv = cl.pick_folder_argv("win", which_from({"powershell": "ps.exe"}), "py")
        self.assertEqual(argv[0], "ps.exe")
        self.assertIn("-STA", argv)
        self.assertIn("FolderBrowserDialog", argv[-1])
        self.assertIn("TopMost", argv[-1])

    def test_mac_osascript(self):
        argv = cl.pick_folder_argv("mac", which_from({"osascript": "/usr/bin/osascript"}), "py")
        self.assertEqual(argv[0], "/usr/bin/osascript")
        self.assertIn("choose folder", argv[-1])

    def test_linux_prefers_zenity_then_tk(self):
        self.assertEqual(cl.pick_folder_argv("linux", which_from({"zenity": "/z"}), "py")[0], "/z")
        self.assertEqual(cl.pick_folder_argv("linux", which_from({"kdialog": "/k"}), "py")[0], "/k")
        argv = cl.pick_folder_argv("linux", which_from({}), "/usr/bin/python3")
        self.assertEqual(argv[:2], ["/usr/bin/python3", "-c"])
        self.assertIn("askdirectory", argv[2])

    def test_no_python_no_picker(self):
        self.assertIsNone(cl.pick_folder_argv("linux", which_from({}), ""))

    def test_pick_returns_trimmed_path_or_blank(self):
        self.assertEqual(cl.pick_folder(["x"], run=lambda a: (0, "D:\\code\\proj\r\n")), "D:\\code\\proj")
        self.assertEqual(cl.pick_folder(["x"], run=lambda a: (1, "")), "")


class TestRemember(unittest.TestCase):
    def test_newest_first_deduped_capped(self):
        r = cl.remember(["/a", "/b", "/c"], "/b", cap=3)
        self.assertEqual(r, ["/b", "/a", "/c"])
        r = cl.remember(["/a", "/b", "/c"], "/d", cap=3)
        self.assertEqual(r, ["/d", "/a", "/b"])

    def test_ignores_junk(self):
        self.assertEqual(cl.remember(None, "/a"), ["/a"])
        self.assertEqual(cl.remember(["/a", 5, ""], "/b"), ["/b", "/a"])


if __name__ == "__main__":
    unittest.main()
