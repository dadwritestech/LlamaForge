import conftest_paths  # noqa: F401
import io
import json
import os
import shutil
import struct
import tempfile
import unittest
from unittest import mock

import config
import gguf
import routes
import tmpldoctor


def _s(text):
    b = text.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _kv_str(key, val):
    return _s(key) + struct.pack("<I", 8) + _s(val)


def _kv_str_arr(key, vals):
    return (_s(key) + struct.pack("<I", 9) + struct.pack("<I", 8)
            + struct.pack("<Q", len(vals)) + b"".join(_s(v) for v in vals))


def gguf_bytes(template=None, vocab=("a", "b", "c")):
    kvs = [_kv_str("general.architecture", "llama"), _kv_str_arr("tokenizer.ggml.tokens", list(vocab))]
    if template is not None:
        kvs.append(_kv_str("tokenizer.chat_template", template))
    return (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kvs))
            + b"".join(kvs) + b"\0" * 64)


OLD = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
NEW = OLD + "{% if tools %}<tools>{{ tools | tojson }}</tools>{% endif %}"


class Embedded(unittest.TestCase):
    def test_reads_the_template_from_a_file_and_a_stream(self):
        d = tempfile.mkdtemp()
        try:
            p = os.path.join(d, "m.gguf")
            with open(p, "wb") as f:
                f.write(gguf_bytes(OLD))
            self.assertEqual(gguf.chat_template(p), OLD)
        finally:
            shutil.rmtree(d, ignore_errors=True)
        self.assertEqual(gguf.chat_template_of(tmpldoctor.Forward(io.BytesIO(gguf_bytes(NEW)))), NEW)
        self.assertIsNone(gguf.chat_template_of(io.BytesIO(gguf_bytes(None))))
        self.assertIsNone(gguf.chat_template_of(io.BytesIO(b"nope")))

    def test_forward_reader_refuses_to_read_past_its_cap(self):
        r = tmpldoctor.Forward(io.BytesIO(b"x" * 100), cap=50)
        r.seek(40, 1)
        with self.assertRaises(EOFError):
            r.read(20)


class RepoPath(unittest.TestCase):
    def test_download_folder_and_hf_cache(self):
        self.assertEqual(tmpldoctor.repo_path("D:/m/LlamaForge-downloads/unsloth--Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"),
                         ("unsloth/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf"))
        self.assertEqual(tmpldoctor.repo_path(r"C:\hf\hub\models--bartowski--Foo-GGUF\snapshots\abc123\Q4\Foo-Q4.gguf"),
                         ("bartowski/Foo-GGUF", "Q4/Foo-Q4.gguf"))
        self.assertEqual(tmpldoctor.repo_path("D:/random/model.gguf"), ("", ""))


class Base(unittest.TestCase):
    def test_base_model_comes_from_the_card(self):
        info = {"cardData": {"base_model": ["Qwen/Qwen3-8B"]}}
        with mock.patch.object(tmpldoctor, "_json", return_value=info):
            self.assertEqual(tmpldoctor.base_repo("unsloth/Qwen3-8B-GGUF"), "Qwen/Qwen3-8B")
        with mock.patch.object(tmpldoctor, "_json", return_value={"cardData": {"base_model": "Qwen/Qwen3-8B"}}):
            self.assertEqual(tmpldoctor.base_repo("x/y"), "Qwen/Qwen3-8B")
        with mock.patch.object(tmpldoctor, "_json", side_effect=OSError("offline")):
            self.assertEqual(tmpldoctor.base_repo("x/y"), "")

    def test_base_template_prefers_the_jinja_file_then_tokenizer_config(self):
        def text(url):
            if url.endswith("chat_template.jinja"):
                raise OSError("404")
            return json.dumps({"chat_template": [{"name": "tool_use", "template": "T"},
                                                 {"name": "default", "template": NEW}]})
        with mock.patch.object(tmpldoctor, "_text", side_effect=text):
            self.assertEqual(tmpldoctor.base_template("Qwen/Qwen3-8B"), NEW)
        with mock.patch.object(tmpldoctor, "_text", return_value=NEW):
            self.assertEqual(tmpldoctor.base_template("Qwen/Qwen3-8B"), NEW)


class Check(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        folder = os.path.join(self.d, "unsloth--Qwen3-8B-GGUF")
        os.makedirs(folder)
        self.p = os.path.join(folder, "Q.gguf")
        with open(self.p, "wb") as f:
            f.write(gguf_bytes(OLD))

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def run_check(self, remote, base):
        with mock.patch.object(tmpldoctor, "_open", return_value=io.BytesIO(gguf_bytes(remote))) as op, \
                mock.patch.object(tmpldoctor, "base_repo", return_value="Qwen/Qwen3-8B" if base is not None else ""), \
                mock.patch.object(tmpldoctor, "base_template", return_value=base):
            r = tmpldoctor.check(self.p)
        return r, op

    def test_repo_reuploaded_with_a_new_template(self):
        r, op = self.run_check(NEW, NEW)
        self.assertEqual(r["verdict"], "repo_newer")
        self.assertEqual(r["repo"], "unsloth/Qwen3-8B-GGUF")
        self.assertIn("+", r["repo_diff"])
        self.assertIn("<tools>", r["repo_diff"])
        url, = op.call_args[0][:1]
        self.assertIn("/unsloth/Qwen3-8B-GGUF/resolve/main/Q.gguf", url)
        self.assertIn("Range", op.call_args[1]["headers"])
        self.assertNotIn("repo_template", r)        # the text stays in the backend

    def test_only_the_base_model_moved_on(self):
        r, _ = self.run_check(OLD, NEW)
        self.assertEqual(r["verdict"], "base_differs")
        self.assertEqual(r["base"], "Qwen/Qwen3-8B")
        self.assertIn("<tools>", r["base_diff"])

    def test_everything_matches_even_with_windows_newlines(self):
        r, _ = self.run_check(OLD.replace("\n", "\r\n") + "\n", OLD)
        self.assertEqual(r["verdict"], "same")

    def test_not_from_a_known_repo(self):
        p = os.path.join(self.d, "loose.gguf")
        with open(p, "wb") as f:
            f.write(gguf_bytes(OLD))
        r = tmpldoctor.check(p)
        self.assertEqual(r["verdict"], "unknown")
        self.assertTrue(r["embedded"])

    def test_offline_says_so(self):
        with mock.patch.object(tmpldoctor, "_open", side_effect=OSError("no network")), \
                mock.patch.object(tmpldoctor, "base_repo", return_value=""):
            r = tmpldoctor.check(self.p)
        self.assertEqual(r["verdict"], "unknown")
        self.assertIn("no network", r["error"])

    def test_texts_keeps_what_apply_needs(self):
        r, _ = self.run_check(NEW, None)
        self.assertEqual(tmpldoctor.LAST[os.path.normcase(self.p)]["repo"], NEW)


class Routes(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.ini = os.path.join(self.d, "models.ini")
        self.cfg = os.path.join(self.d, "config.json")
        self._orig, self._dir = config.CONFIG, tmpldoctor.DIR
        config.CONFIG = self.cfg
        tmpldoctor.DIR = os.path.join(self.d, "chat-templates")
        with open(self.cfg, "w", encoding="utf-8") as f:
            json.dump({"models_ini": self.ini, "active_engine": "llamacpp"}, f)
        folder = os.path.join(self.d, "unsloth--Qwen3-8B-GGUF")
        os.makedirs(folder)
        self.p = os.path.join(folder, "Q.gguf")
        with open(self.p, "wb") as f:
            f.write(gguf_bytes(OLD))
        with open(self.ini, "w", encoding="utf-8") as f:
            f.write("[*]\nctx-size = 4096\n\n[m]\nmodel = %s\n" % self.p)
        tmpldoctor.LAST.clear()

    def tearDown(self):
        config.CONFIG, tmpldoctor.DIR = self._orig, self._dir
        tmpldoctor.LAST.clear()
        shutil.rmtree(self.d, ignore_errors=True)

    def test_registered(self):
        self.assertIn("/api/template/check", routes.GET_ROUTES)
        self.assertIn("/api/template/apply", routes.POST_ROUTES)

    def test_check_then_apply_then_revert(self):
        with mock.patch.object(tmpldoctor, "_open", return_value=io.BytesIO(gguf_bytes(NEW))), \
                mock.patch.object(tmpldoctor, "base_repo", return_value=""):
            st, out = routes.GET_ROUTES["/api/template/check"](routes.Req(qs={"model": "m"}))
        self.assertEqual(st, 200)
        self.assertEqual(out["verdict"], "repo_newer")
        self.assertIsNone(out["override"])
        with mock.patch.object(routes, "_reload_preset", return_value=False):
            st, out = routes.POST_ROUTES["/api/template/apply"](routes.Req(body={"model": "m", "source": "repo"}))
        self.assertTrue(out["ok"])
        path = config.read_sections()["m"]["chat-template-file"]
        with open(path, encoding="utf-8") as f:
            self.assertEqual(f.read(), NEW)
        self.assertTrue(path.startswith(tmpldoctor.DIR))
        with mock.patch.object(routes, "_reload_preset", return_value=False):
            st, out = routes.POST_ROUTES["/api/template/apply"](routes.Req(body={"model": "m", "source": ""}))
        self.assertTrue(out["ok"])
        self.assertNotIn("chat-template-file", config.read_sections()["m"])

    def test_apply_without_a_check_is_refused(self):
        with mock.patch.object(routes, "_reload_preset", return_value=False):
            st, out = routes.POST_ROUTES["/api/template/apply"](routes.Req(body={"model": "m", "source": "repo"}))
        self.assertFalse(out["ok"])
        self.assertNotIn("chat-template-file", config.read_sections()["m"])


if __name__ == "__main__":
    unittest.main()
