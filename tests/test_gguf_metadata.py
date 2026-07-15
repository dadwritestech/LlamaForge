import conftest_paths  # noqa: F401
import os, struct, tempfile, unittest
import gguf


def _s(text):
    b = text.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _kv_str(key, val):
    return _s(key) + struct.pack("<I", 8) + _s(val)


def _kv_u32(key, val):
    return _s(key) + struct.pack("<I", 4) + struct.pack("<I", val)


def _kv_f32(key, val):
    return _s(key) + struct.pack("<I", 6) + struct.pack("<f", val)


def _kv_arr_u32(key, vals):
    # array (type 9) of u32 (element type 4) - must be skipped, never captured
    return (_s(key) + struct.pack("<I", 9) + struct.pack("<I", 4)
            + struct.pack("<Q", len(vals)) + b"".join(struct.pack("<I", v) for v in vals))


def _write_gguf(path, kvs):
    body = b"".join(kvs)
    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))          # version
        f.write(struct.pack("<Q", 0))          # tensor count
        f.write(struct.pack("<Q", len(kvs)))   # n_kv
        f.write(body)


class TestMetadata(unittest.TestCase):
    def _make(self, kvs):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "m.gguf")
        _write_gguf(p, kvs)
        return p

    def test_reads_common_header_fields(self):
        p = self._make([
            _kv_str("general.architecture", "llama"),
            _kv_str("general.name", "Test Model"),
            _kv_str("general.size_label", "8B"),
            _kv_u32("general.file_type", 15),          # Q4_K_M
            _kv_u32("llama.context_length", 4096),
            _kv_u32("llama.embedding_length", 4096),
            _kv_u32("llama.block_count", 32),
            _kv_u32("llama.attention.head_count", 32),
            _kv_str("llama.rope.scaling.type", "yarn"),
        ])
        m = gguf.metadata(p)
        self.assertEqual(m["architecture"], "llama")
        self.assertEqual(m["name"], "Test Model")
        self.assertEqual(m["size_label"], "8B")
        self.assertEqual(m["quantization"], "Q4_K_M")
        self.assertEqual(m["context_length"], 4096)
        self.assertEqual(m["embedding_length"], 4096)
        self.assertEqual(m["block_count"], 32)
        self.assertEqual(m["head_count"], 32)
        self.assertEqual(m["rope_scaling"], "yarn")

    def test_skips_arrays_and_unlisted_strings(self):
        # a big token array + chat_template must not break parsing or appear
        p = self._make([
            _kv_str("general.architecture", "qwen2"),
            _kv_str("tokenizer.chat_template", "x" * 5000),
            _kv_arr_u32("tokenizer.ggml.token_type", list(range(2000))),
            _kv_u32("qwen2.context_length", 32768),
        ])
        m = gguf.metadata(p)
        self.assertEqual(m["architecture"], "qwen2")
        self.assertEqual(m["context_length"], 32768)
        self.assertNotIn("tokenizer.chat_template", m)

    def test_omits_missing_fields(self):
        p = self._make([_kv_str("general.architecture", "gemma")])
        m = gguf.metadata(p)
        self.assertEqual(m["architecture"], "gemma")
        self.assertNotIn("quantization", m)
        self.assertNotIn("context_length", m)

    def test_bad_magic_returns_none(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "notgguf.bin")
        with open(p, "wb") as f:
            f.write(b"NOPE" + b"\0" * 32)
        self.assertIsNone(gguf.metadata(p))

    def test_missing_file_returns_none(self):
        self.assertIsNone(gguf.metadata("/no/such/file.gguf"))


if __name__ == "__main__":
    unittest.main()
