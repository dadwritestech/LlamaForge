import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const inline = async (name, deps = {}) => {
  let src = await readFile(new URL(`../web/js/${name}`, import.meta.url), "utf8");
  for (const [from, to] of Object.entries(deps)) src = src.replace(`"./${from}"`, `"${to}"`);
  return `data:text/javascript;base64,${Buffer.from(src).toString("base64")}`;
};
const stub = `data:text/javascript;base64,${Buffer.from("export const showModal = () => {};").toString("base64")}`;
const core = await inline("core.js");
const X = await import(await inline("xray.js", {"core.js": core, "models.js": stub}));

const M = 1024 ** 2;

test("a row splits into kinds, GPU part first, RAM part hatched", () => {
  const row = {parts: {attention: 100 * M, experts: 900 * M}, ram_parts: {experts: 900 * M}};
  assert.deepEqual(X.segments(row, 2000 * M), [
    {kind: "attention", pct: 5, ram: false},
    {kind: "experts", pct: 45, ram: true},
  ]);
});

test("widths are relative to the widest row and slivers drop", () => {
  const segs = X.segments({parts: {norm: 1, ffn: 1000 * M}, ram_parts: {}}, 1000 * M);
  assert.deepEqual(segs, [{kind: "ffn", pct: 100, ram: false}]);
  assert.deepEqual(X.segments({parts: {ffn: 5}}, 0), []);
});

test("the slider clamps 99 / 999 to every block plus the output", () => {
  assert.equal(X.clampNgl(999, 64), 65);
  assert.equal(X.clampNgl(-1, 64), 0);
  assert.equal(X.clampNgl("12", 64), 12);
  assert.equal(X.clampNgl("all", 64), 0);
});

test("the slider says what it means", () => {
  assert.match(X.nglText(0, 40), /system RAM/);
  assert.match(X.nglText(41, 40), /all 40 blocks \+ output/);
  assert.equal(X.nglText(10, 40), "last 10 of 40 blocks");
});

test("sizes read as GB or MB", () => {
  assert.equal(X.gb(13.26 * 1024 ** 3), "13.26 GB");
  assert.equal(X.gb(431 * M), "431 MB");
});

test("the body escapes model data and hatches RAM", () => {
  const v = {total: 3 * M, gpu: 1 * M, ram: 2 * M, confident: false,
             types: [{type: "<b>Q4</b>", bytes: 3 * M, share: 1}],
             rows: [{label: "blk <0>", type: "Q4_K", gpu: 1 * M, ram: 2 * M,
                     parts: {ffn: 3 * M}, ram_parts: {ffn: 2 * M}}]};
  const html = X.body(v);
  assert.doesNotMatch(html, /<b>Q4<\/b>/);
  assert.match(html, /blk &lt;0&gt;/);
  assert.match(html, /class="xr-k-ffn ram"/);
  assert.match(html, /is a guess/);
  assert.doesNotMatch(html, /tied output/);
  assert.match(X.body({...v, tied_copy: 748 * M}), /748 MB GPU copy/);
});

test("a quant's short name comes from its file name", () => {
  assert.equal(X.quantName("gemma-4-26b-a4b-it-Q4_K_M.gguf"), "Q4_K_M");
  assert.equal(X.quantName("Q8_0/gemma-4-26b-Q8_0-00001-of-00002.gguf"), "Q8_0");
  assert.equal(X.quantName("Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"), "UD-IQ4_XS");
  assert.equal(X.quantName("gemma-4-26b-a4b-it-MXFP4_MOE.gguf"), "MXFP4_MOE");
  assert.equal(X.quantName("model-bf16.gguf"), "bf16");
  assert.equal(X.quantName("Ternary-Bonsai-2-27B-PTQ1_0.gguf"), "PTQ1_0");
  assert.equal(X.quantName("weird.gguf"), "weird");
});

const G = 1024 ** 3;
const half = (file, total, gpu, label = "blk 0") => ({
  file, total, gpu, ram: total - gpu, confident: true, types: [{type: "Q4_K", bytes: total, share: 1}],
  rows: [{label, type: "Q4_K", gpu, ram: total - gpu, parts: {ffn: total}, ram_parts: total > gpu ? {ffn: total - gpu} : {}}],
});

test("the compare line says what the other quant costs", () => {
  const a = half("m-Q4_K_M.gguf", 9 * G, 9 * G), b = half("m-Q8_0.gguf", 15 * G, 10 * G);
  assert.equal(X.compareText(a, b),
    "Q8_0 vs Q4_K_M: +6.00 GB of weights, +1.00 GB on GPU, +5.00 GB in system RAM");
  assert.match(X.compareText(b, a), /^Q4_K_M vs Q8_0: -6.00 GB of weights/);
});

test("both stacks share one scale and model data is escaped", () => {
  const a = half("m-Q4_K_M.gguf", 5 * G, 5 * G), b = half("<x>-Q8_0.gguf", 10 * G, 10 * G, "blk <1>");
  const html = X.compareBody({...a, vs: b});
  assert.match(html, /style="width:50%"/);     // the smaller file's bar is half as long
  assert.match(html, /style="width:100%"/);
  assert.match(html, /blk &lt;1&gt;/);
  assert.doesNotMatch(html, /<x>/);
  assert.match(html, /none of its weights were downloaded/);
});

test("a file you don't have says whether it fits your VRAM", () => {
  const v = {n_layer: 30, ngl: 31, gpu: 10 * G, tied_copy: 0, ram: 1 * G, vram: 16 * G, budget: 14.4 * G};
  assert.equal(X.fitText(v), "Fits your 16.00 GB of VRAM: 10.00 GB of weights on the GPU, 6.00 GB left for the KV cache and buffers.");
  assert.equal(X.fitText({...v, gpu: 15 * G, tied_copy: 0.5 * G}),
    "Too much for your 16.00 GB of VRAM: 15.50 GB of weights on the GPU, 1.10 GB over what leaves room for the KV cache.");
  assert.match(X.fitText({...v, vram: 0}), /No GPU detected/);
  assert.equal(X.fitText({n_layer: 30, gpu: 1}), "");      // a model you have: no VRAM line
  assert.match(X.body({...v, total: 11 * G, types: [], rows: [], confident: true}), /Fits your 16.00 GB/);
});

test("moeText names the blocks whose experts run on the CPU", () => {
  assert.equal(X.moeText(0, 40), "none: every expert on the GPU");
  assert.equal(X.moeText(12, 40), "first 12 of 40 blocks");
  assert.equal(X.moeText(40, 40), "all 40 blocks");
});

test("a model you have with no GPU says nothing about fitting", () => {
  assert.equal(X.fitText({model: "m", vram: 0, budget: 0, gpu: 1, rows: []}), "");
});

test("Apply asks before writing and says it unloads a loaded model", () => {
  const q = X.applyAsk("qwen", "-ngl 41 --n-cpu-moe 7");
  assert.match(q, /Save -ngl 41 --n-cpu-moe 7 as the GPU layers and experts on CPU of qwen\?/);
  assert.match(q, /unloaded/);
  assert.doesNotMatch(X.applyAsk("gem", "-ngl 20"), /and experts on CPU/);
});
