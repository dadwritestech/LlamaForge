import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const inline = async (name, deps = {}) => {
  let src = await readFile(new URL(`../web/js/${name}`, import.meta.url), "utf8");
  for (const [from, to] of Object.entries(deps)) src = src.replace(`"./${from}"`, `"${to}"`);
  return `data:text/javascript;base64,${Buffer.from(src).toString("base64")}`;
};
const core = await inline("core.js");
const wav = await inline("wav.js");
const { listenMarkup, dlLine, curlSnippet } =
  await import(await inline("listen.js", {"core.js": core, "wav.js": wav}));

const OFFER = {label: "Gemma 4 E4B", repo: "unsloth/gemma-4-E4B-it-GGUF",
               path: "gemma-4-E4B-it-Q4_K_M.gguf", mmproj: "mmproj-F16.gguf",
               bytes: 5967544256, license: "Apache-2.0", note: "Hears and answers on one model."};
const IDLE = {running: false, phase: "idle", repo: ""};
const ST = (o) => ({models: [], offer: OFFER, download: IDLE, ...o});

test("with no model that can hear, the card offers Gemma 4 E4B", () => {
  const out = listenMarkup(ST(), "");
  assert.match(out, /data-stt-get/);
  assert.match(out, /Gemma 4 E4B/);
  assert.match(out, /5\.6 GiB/);
  assert.doesNotMatch(out, /id="stt-rec"/);
});

test("while the model downloads the card shows progress, not the button", () => {
  const d = {running: true, phase: "downloading", repo: OFFER.repo, file: OFFER.path,
             done_files: 0, total_files: 2, downloaded: 50, total: 200};
  const out = listenMarkup(ST({download: d}), "");
  assert.match(out, /id="stt-dl"/);
  assert.doesNotMatch(out, /data-stt-get/);
  assert.equal(dlLine(d), "Downloading gemma-4-E4B-it-Q4_K_M.gguf (1 of 2) 25%");
  assert.equal(dlLine({phase: "registering"}), "Adding it to your models...");
});

test("another download in progress disables Get", () => {
  const out = listenMarkup(ST({download: {running: true, phase: "downloading", repo: "x/y"}}), "");
  assert.match(out, /data-stt-get="1" disabled/);
  assert.match(out, /Another download/);
});

test("a failed download says why", () => {
  const out = listenMarkup(ST({download: {running: false, phase: "failed", repo: OFFER.repo, error: "<disk full>"}}), "");
  assert.match(out, /&lt;disk full&gt;/);
});

test("with a model, the mic is there and names it", () => {
  const out = listenMarkup(ST({models: ["voxtral-mini"]}), "http://127.0.0.1:8090");
  assert.match(out, /id="stt-rec"/);
  assert.match(out, /voxtral-mini/);
  assert.doesNotMatch(out, /data-stt-get/);
  assert.match(out, /audio\/transcriptions/);
});

test("model names are escaped", () => {
  const out = listenMarkup(ST({models: ["<img src=x>"]}), "");
  assert.doesNotMatch(out, /<img src=x>/);
});

test("the curl snippet points at this panel", () => {
  assert.match(curlSnippet("http://127.0.0.1:8090"),
               /curl http:\/\/127\.0\.0\.1:8090\/v1\/audio\/transcriptions/);
  assert.match(curlSnippet(""), /-F file=@speech\.wav/);
  assert.equal(listenMarkup(null, ""), "");
});
