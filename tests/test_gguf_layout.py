import conftest_paths  # noqa: F401
import os
import struct
import tempfile
import unittest

import gguf


def _s(text):
    b = text.encode()
    return struct.pack("<Q", len(b)) + b


def kv_u32(key, val):
    return _s(key) + struct.pack("<I", 4) + struct.pack("<I", val)


def kv_str(key, val):
    return _s(key) + struct.pack("<I", 8) + _s(val)


def kv_arr_i32(key, vals):
    return (_s(key) + struct.pack("<I", 9) + struct.pack("<I", 5)
            + struct.pack("<Q", len(vals)) + b"".join(struct.pack("<i", v) for v in vals))


def kv_arr_bool(key, vals):
    return (_s(key) + struct.pack("<I", 9) + struct.pack("<I", 7)
            + struct.pack("<Q", len(vals)) + b"".join(struct.pack("<?", v) for v in vals))


def write_gguf(path, kvs, tensors, align=32):
    """tensors: [(name, type_id, nbytes)] laid out back to back, each padded to `align`."""
    infos, off = [], 0
    for name, typ, nbytes in tensors:
        infos.append(_s(name) + struct.pack("<I", 1) + struct.pack("<Q", nbytes)
                     + struct.pack("<I", typ) + struct.pack("<Q", off))
        off += nbytes + (-nbytes) % align
    head = (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(tensors))
            + struct.pack("<Q", len(kvs)) + b"".join(kvs) + b"".join(infos))
    head += b"\0" * ((-len(head)) % align)
    with open(path, "wb") as f:
        f.write(head)
        f.write(b"\0" * off)


class Layout(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def path(self, name="m.gguf"):
        return os.path.join(self.dir, name)

    def test_tensor_sizes_come_from_offset_gaps(self):
        write_gguf(self.path(), [kv_str("general.architecture", "llama"),
                                 kv_u32("llama.block_count", 2)],
                   [("token_embd.weight", 12, 1000), ("blk.0.attn_q.weight", 14, 64),
                    ("blk.1.ffn_up_exps.weight", 139, 4096)])
        lay = gguf.layout(self.path())
        self.assertEqual(lay["arch"], "llama")
        self.assertEqual(lay["kv"]["block_count"], 2)
        names = [t[0] for t in lay["tensors"]]
        self.assertEqual(names, ["token_embd.weight", "blk.0.attn_q.weight",
                                 "blk.1.ffn_up_exps.weight"])
        sizes = {t[0]: t[2] for t in lay["tensors"]}
        self.assertEqual(sizes["token_embd.weight"], 1024)      # 1000 padded to 32
        self.assertEqual(sizes["blk.0.attn_q.weight"], 64)
        self.assertEqual(sizes["blk.1.ffn_up_exps.weight"], 4096)
        self.assertEqual({t[0]: t[1] for t in lay["tensors"]}["blk.1.ffn_up_exps.weight"], 139)

    def test_keeps_small_numeric_arrays_it_needs(self):
        write_gguf(self.path(), [kv_str("general.architecture", "nemotron_h"),
                                 kv_arr_i32("nemotron_h.attention.head_count_kv", [0, 8, 0, 8]),
                                 kv_arr_bool("nemotron_h.attention.sliding_window_pattern",
                                             [True, False, True, False])],
                   [("blk.0.x.weight", 0, 32)])
        kv = gguf.layout(self.path())["kv"]
        self.assertEqual(kv["head_count_kv"], [0, 8, 0, 8])
        self.assertEqual(kv["sliding_window_pattern"], [True, False, True, False])

    def test_split_gguf_reads_every_shard(self):
        kvs = [kv_str("general.architecture", "llama")]
        write_gguf(self.path("big-00001-of-00002.gguf"), kvs, [("blk.0.a.weight", 0, 64)])
        write_gguf(self.path("big-00002-of-00002.gguf"), kvs, [("blk.1.a.weight", 0, 128)])
        lay = gguf.layout(self.path("big-00001-of-00002.gguf"))
        self.assertEqual([(t[0], t[2]) for t in lay["tensors"]],
                         [("blk.0.a.weight", 64), ("blk.1.a.weight", 128)])

    def test_unreadable_is_none(self):
        with open(self.path(), "wb") as f:
            f.write(b"nope")
        self.assertIsNone(gguf.layout(self.path()))
        self.assertIsNone(gguf.layout(self.path("missing.gguf")))

    def test_header_prefix_plus_true_size_matches_the_file(self):
        """A remote quant is read from its first bytes (an HTTP Range) and the
        size the server reports; tensor sizes must match the whole file's."""
        import io
        write_gguf(self.path(), [kv_str("general.architecture", "llama"),
                                 kv_u32("llama.block_count", 2)],
                   [("token_embd.weight", 12, 1000), ("blk.0.attn_q.weight", 14, 64),
                    ("blk.1.ffn_up_exps.weight", 139, 4096)])
        raw = open(self.path(), "rb").read()
        head = raw[:600]                      # header only, no tensor data
        lay = gguf.layout_streams([(io.BytesIO(head), len(raw))])
        self.assertEqual(lay, gguf.layout(self.path()))

    def test_truncated_header_is_none(self):
        import io
        write_gguf(self.path(), [kv_str("general.architecture", "llama")],
                   [("blk.0.attn_q.weight", 14, 64)])
        raw = open(self.path(), "rb").read()
        self.assertIsNone(gguf.layout_streams([(io.BytesIO(raw[:20]), len(raw))]))


if __name__ == "__main__":
    unittest.main()
