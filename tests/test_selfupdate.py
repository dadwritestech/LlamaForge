import json, os, shutil, tempfile, unittest, zipfile

import conftest_paths  # noqa: F401
import selfupdate


def make_release_zip(path, tag, extra=None):
    with zipfile.ZipFile(path, "w") as z:
        root = f"LlamaForge-{tag.lstrip('v')}/"
        z.writestr(root + "backend/server.py", "# new server\n")
        z.writestr(root + "run.ps1", "# runner\n")
        z.writestr(root + "run.sh", "# runner\n")
        for name, data in (extra or {}).items():
            z.writestr(root + name, data)


class SelfUpdate(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        with open(os.path.join(self.root, ".lf-files.json"), "w") as f:
            json.dump({"version": "v0.10.1", "files": ["backend/server.py"]}, f)
        os.makedirs(os.path.join(self.root, "backend"))
        with open(os.path.join(self.root, "backend", "server.py"), "w") as f:
            f.write("# old\n")
        with open(os.path.join(self.root, "config.json"), "w") as f:
            f.write('{"mine": true}')

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def fake_download(self, tag, extra=None):
        def dl(url, dest):
            self.url = url
            make_release_zip(dest, tag, extra)
        return dl

    def test_installs_release_and_keeps_user_files(self):
        job = selfupdate.UpdateJob(self.root, download=self.fake_download("v0.11.0"))
        job.run("v0.11.0")
        self.assertEqual(job.progress()["state"], "done", job.progress())
        self.assertIn("/archive/refs/tags/v0.11.0.zip", self.url)
        with open(os.path.join(self.root, "backend", "server.py")) as f:
            self.assertEqual(f.read(), "# new server\n")
        with open(os.path.join(self.root, "config.json")) as f:
            self.assertEqual(f.read(), '{"mine": true}')
        self.assertEqual(selfupdate.appinstall.installed_version(self.root), "v0.11.0")

    def test_refuses_git_checkouts(self):
        os.remove(os.path.join(self.root, ".lf-files.json"))
        job = selfupdate.UpdateJob(self.root, download=self.fake_download("v0.11.0"))
        job.run("v0.11.0")
        self.assertEqual(job.progress()["state"], "error")
        self.assertIn("git pull", job.progress()["error"])

    def test_rejects_non_release_tags(self):
        job = selfupdate.UpdateJob(self.root, download=self.fake_download("v0.11.0"))
        job.run("master; rm -rf /")
        self.assertEqual(job.progress()["state"], "error")

    def test_bad_archive_is_an_error_not_a_half_install(self):
        def dl(url, dest):
            with open(dest, "wb") as f:
                f.write(b"not a zip")
        job = selfupdate.UpdateJob(self.root, download=dl)
        job.run("v0.11.0")
        self.assertEqual(job.progress()["state"], "error")
        with open(os.path.join(self.root, "backend", "server.py")) as f:
            self.assertEqual(f.read(), "# old\n")


class RestartCommand(unittest.TestCase):
    def test_windows_relaunches_through_run_ps1(self):
        cmd = selfupdate.restart_cmd("C:/lf", "windows")
        self.assertEqual(cmd[0], "powershell")
        self.assertIn("C:/lf" + os.sep + "run.ps1", " ".join(cmd))

    def test_posix_relaunches_through_run_sh(self):
        cmd = selfupdate.restart_cmd("/opt/lf", "linux")
        self.assertEqual(cmd[:2], ["sh", "-c"])
        self.assertIn("run.sh", cmd[2])
        self.assertIn("sleep", cmd[2])


if __name__ == "__main__":
    unittest.main()
