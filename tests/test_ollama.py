import conftest_paths  # noqa: F401
import json, os, shutil, tempfile, unittest

import ollama

MODEL = "application/vnd.ollama.image.model"
PROJ = "application/vnd.ollama.image.projector"


class OllamaStore(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.root, "blobs"))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def blob(self, hexd, data=b"GGUF" + b"\0" * 60):
        with open(os.path.join(self.root, "blobs", "sha256-" + hexd), "wb") as f:
            f.write(data)
        return "sha256:" + hexd

    def manifest(self, name, tag, layers, host="registry.ollama.ai", ns="library"):
        d = os.path.join(self.root, "manifests", host, ns, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, tag), "w") as f:
            json.dump({"schemaVersion": 2, "layers": [
                {"mediaType": mt, "digest": dg, "size": 64} for mt, dg in layers]}, f)

    def test_model_layer_becomes_an_entry(self):
        self.manifest("llama3.2", "3b", [(MODEL, self.blob("aa" * 32))])
        [e] = ollama.entries(self.root)
        self.assertEqual(e["id"], "llama3.2-3b")
        self.assertEqual(e["model"], os.path.join(self.root, "blobs", "sha256-" + "aa" * 32).replace("\\", "/"))
        self.assertEqual(e["ollama"], "llama3.2:3b")
        self.assertNotIn("mmproj", e)

    def test_latest_tag_is_dropped_from_the_id(self):
        self.manifest("qwen3", "latest", [(MODEL, self.blob("bb" * 32))])
        self.assertEqual(ollama.entries(self.root)[0]["id"], "qwen3")

    def test_projector_layer_is_the_mmproj(self):
        self.manifest("gemma3", "4b", [(MODEL, self.blob("cc" * 32)), (PROJ, self.blob("dd" * 32))])
        [e] = ollama.entries(self.root)
        self.assertTrue(e["mmproj"].endswith("sha256-" + "dd" * 32))

    def test_user_namespace_and_other_hosts_keep_their_prefix(self):
        self.manifest("mymodel", "q4", [(MODEL, self.blob("ee" * 32))], ns="someone")
        self.manifest("Qwen3-GGUF", "Q4_K_M", [(MODEL, self.blob("ff" * 32))],
                      host="hf.co", ns="unsloth")
        ids = sorted(e["id"] for e in ollama.entries(self.root))
        self.assertEqual(ids, ["hf.co-unsloth-qwen3-gguf-q4-k-m", "someone-mymodel-q4"])

    def test_skips_non_gguf_missing_and_broken(self):
        self.manifest("notgguf", "x", [(MODEL, self.blob("11" * 32, b"safetensors"))])
        self.manifest("gone", "x", [(MODEL, "sha256:" + "22" * 32)])
        d = os.path.join(self.root, "manifests", "registry.ollama.ai", "library", "broken")
        os.makedirs(d)
        with open(os.path.join(d, "x"), "w") as f:
            f.write("{not json")
        self.assertEqual(ollama.entries(self.root), [])

    def test_two_tags_of_one_blob_are_listed_once(self):
        dg = self.blob("33" * 32)
        self.manifest("phi4", "latest", [(MODEL, dg)])
        self.manifest("phi4", "14b", [(MODEL, dg)])
        self.assertEqual(len(ollama.entries(self.root)), 1)

    def test_embedding_models_are_marked(self):
        self.manifest("nomic-embed-text", "latest", [(MODEL, self.blob("44" * 32))])
        self.assertTrue(ollama.entries(self.root)[0]["embeddings"])

    def test_no_store_is_empty(self):
        self.assertEqual(ollama.entries(os.path.join(self.root, "nope")), [])


class StoreLocation(unittest.TestCase):
    def test_env_var_wins(self):
        self.assertEqual(ollama.store_dirs({"OLLAMA_MODELS": "/x/m"}, "/home/u")[0], "/x/m")

    def test_default_is_home(self):
        self.assertEqual(ollama.store_dirs({}, "/home/u")[0],
                         os.path.join("/home/u", ".ollama", "models"))


class ScannerMerge(unittest.TestCase):
    def test_ollama_entry_whose_id_clashes_gets_a_prefix(self):
        import scanner
        merged = scanner.merge([{"id": "qwen3", "model": "D:/m/qwen3.gguf"}],
                               [{"id": "qwen3", "model": "C:/o/blobs/sha256-ab"}])
        self.assertEqual([e["id"] for e in merged], ["qwen3", "ollama--qwen3"])


if __name__ == "__main__":
    unittest.main()
