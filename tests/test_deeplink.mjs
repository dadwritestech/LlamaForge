import assert from "node:assert/strict";
import test from "node:test";
import { parseHash, discoverLink } from "../web/js/deeplink.js";

test("a plain tab hash is just the tab", () => {
  assert.deepEqual(parseHash("#models"), {tab: "models", repo: null, file: null});
  assert.deepEqual(parseHash(""), {tab: "", repo: null, file: null});
  assert.deepEqual(parseHash("#discover"), {tab: "discover", repo: null, file: null});
});

test("discover takes a repo and a file", () => {
  assert.deepEqual(parseHash("#discover?repo=unsloth/Qwen3-8B-GGUF&file=Qwen3-8B-Q4_K_M.gguf"),
                   {tab: "discover", repo: "unsloth/Qwen3-8B-GGUF", file: "Qwen3-8B-Q4_K_M.gguf"});
  // a quant in a subfolder, URL-encoded the way huggingface.co builds links
  assert.deepEqual(parseHash("#discover?repo=a/b&file=Q8_0%2Fb-Q8_0-00001-of-00002.gguf"),
                   {tab: "discover", repo: "a/b", file: "Q8_0/b-Q8_0-00001-of-00002.gguf"});
  assert.deepEqual(parseHash("#discover?repo=a.b/c_d-1"), {tab: "discover", repo: "a.b/c_d-1", file: null});
});

test("a pasted huggingface.co URL works as the repo", () => {
  assert.equal(parseHash("#discover?repo=https://huggingface.co/a/b").repo, "a/b");
  assert.equal(parseHash("#discover?repo=hf.co/a/b").repo, "a/b");
});

test("anything that is not a plain owner/name repo or a .gguf path is dropped", () => {
  for (const bad of ["../x", "a/b/c", "a", "<img src=x>/b", "a/b?c", "/a/b", "a/.."])
    assert.equal(parseHash("#discover?repo=" + encodeURIComponent(bad)).repo, null, bad);
  for (const bad of ["x.bin", "../x.gguf", "/x.gguf", "a/../x.gguf", "a\\x.gguf"])
    assert.equal(parseHash("#discover?repo=a/b&file=" + encodeURIComponent(bad)).file, null, bad);
  assert.equal(parseHash("#discover?file=x.gguf").file, null);   // a file needs its repo
});

test("discoverLink round-trips", () => {
  const h = discoverLink("unsloth/Qwen3-8B-GGUF", "Q4_K_M/x.gguf");
  assert.equal(h, "#discover?repo=unsloth/Qwen3-8B-GGUF&file=Q4_K_M%2Fx.gguf");
  assert.deepEqual(parseHash(h), {tab: "discover", repo: "unsloth/Qwen3-8B-GGUF", file: "Q4_K_M/x.gguf"});
});
