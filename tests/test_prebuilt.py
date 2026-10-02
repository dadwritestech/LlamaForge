import hashlib, io, os, json, pathlib, shutil, tarfile, tempfile, time, unittest, urllib.parse, urllib.request, zipfile

import conftest_paths  # noqa: F401
import prebuilt

TAG = "b11351"
NAMES = [
    f"cudart-llama-{TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz",
    f"cudart-llama-{TAG}-bin-ubuntu-cuda-13.4-x64.tar.gz",
    "cudart-llama-bin-win-cuda-12.4-x64.zip",
    "cudart-llama-bin-win-cuda-13.4-arm64.zip",
    "cudart-llama-bin-win-cuda-13.4-x64.zip",
    f"llama-{TAG}-bin-android-arm64.tar.gz",
    f"llama-{TAG}-bin-linux-arm64-snapdragon.tar.gz",
    f"llama-{TAG}-bin-macos-arm64.tar.gz",
    f"llama-{TAG}-bin-macos-x64.tar.gz",
    f"llama-{TAG}-bin-ubuntu-arm64.tar.gz",
    f"llama-{TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz",
    f"llama-{TAG}-bin-ubuntu-cuda-13.4-x64.tar.gz",
    f"llama-{TAG}-bin-ubuntu-rocm-10.0-x64.tar.gz",
    f"llama-{TAG}-bin-ubuntu-vulkan-x64.tar.gz",
    f"llama-{TAG}-bin-ubuntu-x64.tar.gz",
    f"llama-{TAG}-bin-win-cpu-arm64.zip",
    f"llama-{TAG}-bin-win-cpu-x64.zip",
    f"llama-{TAG}-bin-win-cuda-12.4-x64.zip",
    f"llama-{TAG}-bin-win-cuda-13.4-x64.zip",
    f"llama-{TAG}-bin-win-rocm-10.0-x64.zip",
    f"llama-{TAG}-bin-win-sycl-x64.zip",
    f"llama-{TAG}-bin-win-vulkan-x64.zip",
    f"llama-{TAG}-ui.tar.gz",
    f"llama-{TAG}-xcframework.zip",
]
ASSETS = [{"name": n, "size": 1, "digest": "sha256:" + "ab" * 32,
           "state": "uploaded", "browser_download_url": "https://x/" + n}
          for n in NAMES]

NV_5080 = [{"name": "RTX 5080", "compute_cap": "12.0", "vram_mib": 16303}]
NV_4090 = [{"name": "RTX 4090", "compute_cap": "8.9", "vram_mib": 24564}]


class ParseAssetTest(unittest.TestCase):
    def test_cuda_bin(self):
        a = prebuilt.parse_asset(f"llama-{TAG}-bin-win-cuda-13.4-x64.zip")
        self.assertEqual(a, {"kind": "bin", "os": "win", "arch": "x64",
                             "variant": "cuda-13.4", "cuda": (13, 4)})

    def test_plain_ubuntu_is_cpu_and_macos_arm_is_metal(self):
        self.assertEqual(prebuilt.parse_asset(f"llama-{TAG}-bin-ubuntu-x64.tar.gz")["variant"], "cpu")
        self.assertEqual(prebuilt.parse_asset(f"llama-{TAG}-bin-macos-arm64.tar.gz")["variant"], "metal")
        self.assertEqual(prebuilt.parse_asset(f"llama-{TAG}-bin-macos-x64.tar.gz")["variant"], "cpu")

    def test_cudart_with_and_without_build_number(self):
        a = prebuilt.parse_asset("cudart-llama-bin-win-cuda-12.4-x64.zip")
        self.assertEqual((a["kind"], a["os"], a["cuda"]), ("cudart", "win", (12, 4)))
        b = prebuilt.parse_asset(f"cudart-llama-{TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz")
        self.assertEqual((b["kind"], b["os"], b["cuda"]), ("cudart", "ubuntu", (12, 8)))

    def test_unsupported_assets_are_none(self):
        for n in (f"llama-{TAG}-ui.tar.gz", f"llama-{TAG}-xcframework.zip",
                  f"llama-{TAG}-bin-android-arm64.tar.gz",
                  f"llama-{TAG}-bin-linux-arm64-snapdragon.tar.gz",
                  "nightly-tag.txt"):
            self.assertIsNone(prebuilt.parse_asset(n), n)


class PlatformTest(unittest.TestCase):
    def test_platform_key(self):
        self.assertEqual(prebuilt.platform_key("windows", "AMD64"), ("win", "x64"))
        self.assertEqual(prebuilt.platform_key("linux", "x86_64"), ("ubuntu", "x64"))
        self.assertEqual(prebuilt.platform_key("linux", "aarch64"), ("ubuntu", "arm64"))
        self.assertEqual(prebuilt.platform_key("macos", "arm64"), ("macos", "arm64"))
        self.assertEqual(prebuilt.platform_key("windows", "ARM64"), ("win", "arm64"))

    def test_parse_cuda_driver(self):
        smi = ("| NVIDIA-SMI 581.42   Driver Version: 581.42   CUDA Version: 13.0     |")
        self.assertEqual(prebuilt.parse_cuda_driver(smi), (13, 0))
        self.assertIsNone(prebuilt.parse_cuda_driver("command not found"))

    def test_parse_cuda_driver_r600_header(self):
        # R600+ drivers renamed the field and added a KMD version beside it
        smi = ("| NVIDIA-SMI 616.92                 KMD Version: 616.92        "
               "CUDA UMD Version: 13.4     |")
        self.assertEqual(prebuilt.parse_cuda_driver(smi), (13, 4))


class ChooseTest(unittest.TestCase):
    def pick(self, plat, gpus, driver):
        return prebuilt.choose(ASSETS, plat, gpus, driver)

    def test_new_driver_picks_cuda13_with_cudart(self):
        r = self.pick(("win", "x64"), NV_5080, (13, 0))
        self.assertEqual(r["bin"]["name"], f"llama-{TAG}-bin-win-cuda-13.4-x64.zip")
        self.assertEqual(r["cudart"]["name"], "cudart-llama-bin-win-cuda-13.4-x64.zip")

    def test_cuda12_driver_falls_back_to_cuda12_build(self):
        r = self.pick(("win", "x64"), NV_4090, (12, 6))
        self.assertEqual(r["bin"]["name"], f"llama-{TAG}-bin-win-cuda-12.4-x64.zip")
        self.assertEqual(r["cudart"]["name"], "cudart-llama-bin-win-cuda-12.4-x64.zip")

    def test_blackwell_never_gets_a_build_older_than_12_8(self):
        # cuda-12.4 has no sm_120 kernels; with a 12.x driver use Vulkan instead.
        r = self.pick(("win", "x64"), NV_5080, (12, 6))
        self.assertEqual(r["variant"], "vulkan")
        self.assertIsNone(r["cudart"])
        # ...but on Linux the 12.x build is 12.8, which is fine for Blackwell.
        r = self.pick(("ubuntu", "x64"), NV_5080, (12, 9))
        self.assertEqual(r["variant"], "cuda-12.8")

    def test_no_nvidia_picks_vulkan_then_cpu(self):
        self.assertEqual(self.pick(("win", "x64"), [], None)["variant"], "vulkan")
        self.assertEqual(self.pick(("ubuntu", "x64"), [], None)["variant"], "vulkan")
        self.assertEqual(self.pick(("win", "arm64"), [], None)["variant"], "cpu")

    def test_mac(self):
        self.assertEqual(self.pick(("macos", "arm64"), [], None)["variant"], "metal")

    def test_override_variant(self):
        r = prebuilt.choose(ASSETS, ("win", "x64"), NV_5080, (13, 0), variant="cpu")
        self.assertEqual(r["bin"]["name"], f"llama-{TAG}-bin-win-cpu-x64.zip")
        self.assertIsNone(r["cudart"])

    def test_alternatives_list_every_variant_for_the_platform(self):
        r = self.pick(("win", "x64"), [], None)
        self.assertEqual(r["alternatives"],
                         ["cpu", "cuda-12.4", "cuda-13.4", "rocm-10.0", "sycl", "vulkan"])

    def test_asset_still_uploading_is_not_chosen(self):
        assets = [dict(a, state="new") if "vulkan-x64.zip" in a["name"] else a
                  for a in ASSETS]
        r = prebuilt.choose(assets, ("win", "x64"), [], None)
        self.assertIsNone(r["bin"])


class ExtractTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _zip(self, members):
        p = os.path.join(self.tmp, "a.zip")
        with zipfile.ZipFile(p, "w") as z:
            for name, data in members.items():
                z.writestr(name, data)
        return p

    def _tar(self, members):
        p = os.path.join(self.tmp, "a.tar.gz")
        with tarfile.open(p, "w:gz") as t:
            for name, data in members.items():
                info = tarfile.TarInfo(name); info.size = len(data); info.mode = 0o755
                t.addfile(info, io.BytesIO(data))
        return p

    def test_zip_and_tar_extract_and_find_server(self):
        out = os.path.join(self.tmp, "z")
        prebuilt.safe_extract(self._zip({"llama-server.exe": b"x", "ggml.dll": b"y"}), out)
        self.assertEqual(prebuilt.find_server_bin(out), os.path.join(out, "llama-server.exe"))
        out2 = os.path.join(self.tmp, "t")
        prebuilt.safe_extract(self._tar({"build/bin/llama-server": b"x"}), out2)
        self.assertEqual(prebuilt.find_server_bin(out2),
                         os.path.join(out2, "build", "bin", "llama-server"))

    def test_traversal_is_rejected(self):
        for arc in (self._zip({"../evil.txt": b"x"}), self._tar({"../evil.txt": b"x"}),
                    self._tar({"/abs/evil.txt": b"x"})):
            with self.assertRaises(ValueError):
                prebuilt.safe_extract(arc, os.path.join(self.tmp, "o"))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "evil.txt")))

    def test_tar_symlink_escaping_is_rejected(self):
        p = os.path.join(self.tmp, "l.tar.gz")
        with tarfile.open(p, "w:gz") as t:
            info = tarfile.TarInfo("link"); info.type = tarfile.SYMTYPE
            info.linkname = "../../outside"
            t.addfile(info)
        with self.assertRaises(ValueError):
            prebuilt.safe_extract(p, os.path.join(self.tmp, "o"))

    def test_tar_relative_symlink_inside_is_kept(self):
        # Linux/macOS releases ship libfoo.so -> libfoo.so.0 style links.
        p = os.path.join(self.tmp, "ok.tar.gz")
        with tarfile.open(p, "w:gz") as t:
            info = tarfile.TarInfo("lib/libllama.so.0"); info.size = 1
            t.addfile(info, io.BytesIO(b"x"))
            ln = tarfile.TarInfo("lib/libllama.so"); ln.type = tarfile.SYMTYPE
            ln.linkname = "libllama.so.0"
            t.addfile(ln)
        prebuilt.safe_extract(p, os.path.join(self.tmp, "o"))


class InstallsTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)

    def _install(self, name, at):
        d = os.path.join(self.root, name)
        os.makedirs(d)
        open(os.path.join(d, "llama-server.exe"), "w").close()
        with open(os.path.join(d, prebuilt.MANIFEST), "w") as f:
            json.dump({"tag": name.split("-")[0], "variant": "vulkan",
                       "installed_at": at}, f)
        return d

    def test_list_newest_first_and_marks_active(self):
        a = self._install("b1-vulkan", 1)
        b = self._install("b2-vulkan", 2)
        os.makedirs(os.path.join(self.root, "junk"))     # no manifest -> ignored
        got = prebuilt.list_installs(self.root, active_bin=os.path.join(a, "llama-server.exe"))
        self.assertEqual([i["dir"] for i in got], [b, a])
        self.assertEqual([i["active"] for i in got], [False, True])

    def test_prune_keeps_newest_and_active(self):
        dirs = [self._install(f"b{i}-vulkan", i) for i in range(1, 6)]
        active = os.path.join(dirs[0], "llama-server.exe")
        removed = prebuilt.prune_installs(self.root, keep=2, active_bin=active)
        self.assertEqual(sorted(removed), sorted(dirs[1:3]))
        left = [i["dir"] for i in prebuilt.list_installs(self.root)]
        self.assertEqual(sorted(left), sorted([dirs[0], dirs[3], dirs[4]]))


class ResolveTest(unittest.TestCase):
    def test_stable_follows_nightly_tag_pointer(self):
        calls = []
        def fake_json(url):
            calls.append(url)
            if url.endswith("/releases/latest"):
                return {"tag_name": "v0.5.0", "assets": [
                    {"name": "nightly-tag.txt",
                     "browser_download_url": "https://dl/nightly-tag.txt"}]}
            if url.endswith("/releases/tags/b11146"):
                return {"tag_name": "b11146", "published_at": "t", "assets": ASSETS}
            raise AssertionError(url)
        rel = prebuilt.resolve("stable", get_json=fake_json,
                               get_text=lambda url: "b11146\n")
        self.assertEqual(rel["tag"], "b11146")
        self.assertEqual(rel["label"], "v0.5.0")

    def test_nightly_skips_releases_still_uploading(self):
        fresh = [dict(a, state="new") for a in ASSETS]
        def fake_json(url):
            return [{"tag_name": "b3", "prerelease": True, "assets": fresh},
                    {"tag_name": "b2", "prerelease": True, "assets": ASSETS}]
        rel = prebuilt.resolve("nightly", get_json=fake_json, get_text=None,
                               plat=("win", "x64"), gpus=[], driver=None)
        self.assertEqual(rel["tag"], "b2")


if __name__ == "__main__":
    unittest.main()


class InstallerTest(unittest.TestCase):
    """Full install flow against file:// assets - no network, no real binary."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        src = os.path.join(self.tmp, "src")
        os.makedirs(src)
        self.assets = []
        for name, files in (
                (f"llama-{TAG}-bin-win-cuda-13.4-x64.zip",
                 {"llama-server.exe": b"MZ", "ggml-cuda.dll": b"g"}),
                ("cudart-llama-bin-win-cuda-13.4-x64.zip",
                 {"cudart64_13.dll": b"rt", "ggml-cuda.dll": b"WRONG"})):
            path = os.path.join(src, name)
            with zipfile.ZipFile(path, "w") as z:
                for n, data in files.items():
                    z.writestr(n, data)
            with open(path, "rb") as f:
                blob = f.read()
            self.assets.append({
                "name": name, "size": len(blob), "state": "uploaded",
                "digest": "sha256:" + hashlib.sha256(blob).hexdigest(),
                "browser_download_url": pathlib.Path(path).as_uri()})
        self.release = {"tag": TAG, "label": TAG, "published": "", "channel": "nightly",
                        "assets": self.assets}
        self.activated = []
        self.inst = prebuilt.Installer(
            self.tmp, os.path.join(self.tmp, "logs"), on_installed=self.activated.append,
            detect=lambda: (("win", "x64"), NV_5080, (13, 0)))
        self.inst._smoke = lambda sbin: None
        self._orig = prebuilt.resolve
        prebuilt.resolve = lambda *a, **k: self.release
        self.addCleanup(setattr, prebuilt, "resolve", self._orig)

    def _wait(self):
        for _ in range(200):
            if not self.inst.state["running"]:
                return self.inst.state
            time.sleep(0.02)
        self.fail("installer did not finish")

    def test_installs_verifies_links_cudart_and_activates(self):
        self.assertTrue(self.inst.start("nightly"))
        st = self._wait()
        self.assertEqual(st["phase"], "done", self.inst.tail())
        final = os.path.join(self.tmp, "engines", "llama.cpp", f"{TAG}-cuda-13.4")
        self.assertEqual(self.activated, [os.path.join(final, "llama-server.exe")])
        self.assertTrue(os.path.isfile(os.path.join(final, "cudart64_13.dll")))
        # the build's own libs are never replaced by the runtime bundle
        with open(os.path.join(final, "ggml-cuda.dll"), "rb") as f:
            self.assertEqual(f.read(), b"g")
        man = prebuilt.list_installs(os.path.join(self.tmp, "engines", "llama.cpp"))
        self.assertEqual([m["tag"] for m in man], [TAG])

    def test_checksum_mismatch_fails_and_leaves_nothing(self):
        self.assets[0]["digest"] = "sha256:" + "0" * 64
        self.inst.start("nightly")
        st = self._wait()
        self.assertEqual(st["phase"], "failed")
        self.assertIn("checksum", st["error"])
        self.assertEqual(self.activated, [])
        root = os.path.join(self.tmp, "engines", "llama.cpp")
        self.assertEqual(os.listdir(root) if os.path.isdir(root) else [], [])

    def test_reinstalling_same_build_just_switches(self):
        self.inst.start("nightly"); self._wait()
        # source gone: a second install of the same build must not download
        url = self.assets[0]["browser_download_url"]
        os.remove(urllib.request.url2pathname(urllib.parse.urlparse(url).path))
        self.inst.start("nightly")
        st = self._wait()
        self.assertEqual(st["phase"], "done", self.inst.tail())
        self.assertEqual(len(self.activated), 2)
