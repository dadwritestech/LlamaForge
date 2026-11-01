import conftest_paths  # noqa: F401
import json, os, struct, tempfile, unittest
from unittest import mock

import clientsetup, config, gguf, routes, scanner


def _s(text):
    b = text.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _kv_str(key, val):
    return _s(key) + struct.pack("<I", 8) + _s(val)


def _kv_u32(key, val):
    return _s(key) + struct.pack("<I", 4) + struct.pack("<I", val)


def _write(path, kvs):
    with open(path, "wb") as f:
        f.write(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
                + struct.pack("<Q", len(kvs)) + b"".join(kvs))


class KindTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def _gguf(self, name, kvs):
        p = os.path.join(self.d, name)
        _write(p, kvs)
        return p

    def test_rank_pooling_is_a_reranker(self):
        # Qwen3-Reranker / bge-reranker GGUFs: pooling_type 4 (LLAMA_POOLING_TYPE_RANK)
        p = self._gguf("r.gguf", [_kv_str("general.architecture", "qwen3"),
                                  _kv_u32("qwen3.pooling_type", 4)])
        self.assertEqual(gguf.retrieval(p), "rerank")

    def test_mean_cls_last_pooling_is_an_embedder(self):
        for pt in (1, 2, 3):
            p = self._gguf(f"e{pt}.gguf", [_kv_str("general.architecture", "qwen3"),
                                           _kv_u32("qwen3.pooling_type", pt)])
            self.assertEqual(gguf.retrieval(p), "embed", pt)

    def test_encoder_architecture_is_an_embedder(self):
        p = self._gguf("b.gguf", [_kv_str("general.architecture", "nomic-bert")])
        self.assertEqual(gguf.retrieval(p), "embed")

    def test_encoder_with_a_classifier_head_is_a_reranker(self):
        # gpustack's jina-reranker-v1-tiny-en GGUF: jina-bert-v2, no pooling_type,
        # but cls.* tensors, which is how llama.cpp tells a reranker apart
        def tensor(name, dims):
            return (_s(name) + struct.pack("<I", len(dims))
                    + b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", 0, 0))
        def write(path, names):
            kvs = [_kv_str("general.architecture", "jina-bert-v2")]
            with open(path, "wb") as f:
                f.write(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(names))
                        + struct.pack("<Q", len(kvs)) + b"".join(kvs)
                        + b"".join(tensor(n, [384, 384]) for n in names))
        rr, em = os.path.join(self.d, "rr.gguf"), os.path.join(self.d, "em.gguf")
        write(rr, ["token_embd.weight", "blk.0.attn_qkv.weight", "cls.weight", "cls.output.weight"])
        write(em, ["token_embd.weight", "blk.0.attn_qkv.weight"])
        self.assertEqual(gguf.retrieval(rr), "rerank")
        self.assertEqual(gguf.retrieval(em), "embed")

    def test_chat_model_is_neither(self):
        p = self._gguf("c.gguf", [_kv_str("general.architecture", "qwen3")])
        self.assertIsNone(gguf.retrieval(p))
        p = self._gguf("n.gguf", [_kv_str("general.architecture", "llama"),
                                  _kv_u32("llama.pooling_type", 0)])
        self.assertIsNone(gguf.retrieval(p))

    def test_unreadable_is_none(self):
        self.assertIsNone(gguf.retrieval(os.path.join(self.d, "missing.gguf")))

    def test_metadata_reports_the_kind(self):
        p = self._gguf("r.gguf", [_kv_str("general.architecture", "qwen3"),
                                  _kv_u32("qwen3.pooling_type", 4)])
        self.assertEqual(gguf.metadata(p)["retrieval"], "rerank")

    def test_scanner_marks_by_header_not_just_name(self):
        rr = self._gguf("bge-reranker-v2-m3-Q4_K_M.gguf",
                        [_kv_str("general.architecture", "bert"), _kv_u32("bert.pooling_type", 4)])
        em = self._gguf("Qwen3-Emb-0.6B-Q8_0.gguf",
                        [_kv_str("general.architecture", "qwen3"), _kv_u32("qwen3.pooling_type", 3)])
        ch = self._gguf("Qwen3-8B-Q4_K_M.gguf", [_kv_str("general.architecture", "qwen3")])
        named = self._gguf("old-embed-model.gguf", [])          # no header facts: the name still counts
        got = {os.path.basename(e["model"]): e for e in scanner.build_entries([rr, em, ch, named])}
        self.assertTrue(got["bge-reranker-v2-m3-Q4_K_M.gguf"].get("reranking"))
        self.assertNotIn("embeddings", got["bge-reranker-v2-m3-Q4_K_M.gguf"])
        self.assertTrue(got["Qwen3-Emb-0.6B-Q8_0.gguf"].get("embeddings"))
        self.assertNotIn("embeddings", got["Qwen3-8B-Q4_K_M.gguf"])
        self.assertNotIn("reranking", got["Qwen3-8B-Q4_K_M.gguf"])
        self.assertTrue(got["old-embed-model.gguf"].get("embeddings"))


class PersistTest(unittest.TestCase):
    def test_reranking_is_written_and_never_cleared_by_a_rescan(self):
        written = []
        with mock.patch.object(config, "read_sections", return_value={}), \
             mock.patch.object(config, "vision_parked", return_value=set()), \
             mock.patch.object(config, "reconcile_mtp_autowire", return_value={}), \
             mock.patch.object(config, "set_keys", side_effect=lambda mid, k: written.append((mid, k))):
            routes._persist_scanned_entries([{"id": "rr", "model": "/m/rr.gguf", "reranking": True},
                                             {"id": "chat", "model": "/m/c.gguf"}])
        keys = dict(written)
        self.assertEqual(keys["rr"]["reranking"], "true")
        self.assertNotIn("reranking", keys["chat"])     # a hand-set flag survives a rescan


class ClientConfigKindTest(unittest.TestCase):
    def test_chat_is_unchanged(self):
        out = clientsetup.generate("http://127.0.0.1:8080", "k", "qwen", "llamacpp", "posix")
        self.assertEqual(out["kind"], "chat")
        self.assertIn("/v1/chat/completions", out["curl"])

    def test_embedding_model_gets_the_embeddings_call(self):
        out = clientsetup.generate("http://127.0.0.1:8080", "k", "nomic", "llamacpp", "posix",
                                   kind="embed")
        self.assertEqual(out["kind"], "embed")
        self.assertIn("/v1/embeddings", out["curl"])
        self.assertEqual(json.loads(out["payload"]), {"model": "nomic", "input": "Hello"})
        names = [a["name"] for a in out["apps"]]
        self.assertIn("Open WebUI (Documents)", names)
        self.assertNotIn("aider", names)                # a chat-only app
        cont = next(a["text"] for a in out["apps"] if a["name"].startswith("Continue"))
        self.assertIn("roles: [embed]", cont)

    def test_reranker_gets_the_rerank_call(self):
        out = clientsetup.generate("http://127.0.0.1:8080", "k", "bge-rr", "llamacpp", "powershell",
                                   kind="rerank")
        self.assertIn("/v1/rerank", out["curl"])
        body = json.loads(out["payload"])
        self.assertEqual(body["model"], "bge-rr")
        self.assertIn("query", body)
        self.assertIsInstance(body["documents"], list)
        self.assertIn("http://127.0.0.1:8080/v1/rerank",
                      next(a["text"] for a in out["apps"] if a["name"].startswith("Open WebUI")))

    def test_kind_from_settings(self):
        self.assertEqual(routes._retrieval_kind({"settings": {"reranking": "true"}}), "rerank")
        self.assertEqual(routes._retrieval_kind({"settings": {"rerank": "true"}}), "rerank")
        self.assertEqual(routes._retrieval_kind({"settings": {"embeddings": "true"}}), "embed")
        self.assertEqual(routes._retrieval_kind({"settings": {"embeddings": "false"}}), "chat")
        self.assertEqual(routes._retrieval_kind({"settings": {}}), "chat")
        self.assertEqual(routes._retrieval_kind({}), "chat")


if __name__ == "__main__":
    unittest.main()
