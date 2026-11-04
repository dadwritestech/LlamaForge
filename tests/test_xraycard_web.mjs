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
const xray = await inline("xray.js", {"core.js": core, "models.js": stub});
const forge = await inline("forgecard.js");
const X = await import(await inline("xraycard.js", {"xray.js": xray, "forgecard.js": forge}));

const M = 1024 ** 2, G = 1024 ** 3;
const view = {
  arch: "gemma4", n_layer: 2, ngl: 3, total: 3 * G, gpu: 2 * G, ram: 1 * G, tied_copy: 0,
  types: [{type: "MXFP4", bytes: 2 * G, share: 0.6667}, {type: "Q8_0", bytes: 1 * G, share: 0.3333},
          {type: "F32", bytes: 1, share: 0}],
  rows: [
    {label: "embedding", gpu: 0, ram: 512 * M, parts: {embedding: 512 * M}, ram_parts: {embedding: 512 * M}},
    {label: "blk 0", gpu: 512 * M, ram: 512 * M, parts: {attention: 512 * M, experts: 512 * M}, ram_parts: {experts: 512 * M}},
    {label: "blk 1", gpu: 1024 * M, ram: 0, parts: {attention: 512 * M, experts: 512 * M}, ram_parts: {}},
    {label: "output", gpu: 512 * M, ram: 0, parts: {output: 512 * M}, ram_parts: {}},
  ],
};

test("the card says what the model is and where it lives", () => {
  const t = X.xrayCardText(view, "gemma-4-26b");
  assert.equal(t.kicker, "MODEL X-RAY");
  assert.equal(t.title, "gemma-4-26b");
  assert.equal(t.sub, "gemma4  ·  2 blocks  ·  3.00 GB of weights");
  assert.equal(t.gpu, "2.00 GB on GPU");
  assert.equal(t.ram, "1.00 GB in system RAM");
  assert.equal(t.layers, "GPU layers: all 2 blocks + output");
  assert.equal(t.types, "MXFP4 67%  ·  Q8_0 33%");
});

test("one column per layer, height relative to the biggest layer", () => {
  const t = X.xrayCardText(view, "m");
  assert.equal(t.cols.length, 4);
  assert.deepEqual(t.cols.map(c => c.h), [0.5, 1, 1, 0.5]);
  assert.deepEqual(t.cols[1].segs, [
    {kind: "attention", frac: 0.5, ram: false},
    {kind: "experts", frac: 0.5, ram: true},
  ]);
});

test("the legend names only the kinds the model has, and RAM only when something is there", () => {
  const t = X.xrayCardText(view, "m");
  assert.deepEqual(t.legend.map(l => l.kind), ["embedding", "attention", "experts", "output", "ram"]);
  const allGpu = {...view, ram: 0, rows: view.rows.map(r => ({...r, ram_parts: {}}))};
  assert.ok(!X.xrayCardText(allGpu, "m").legend.some(l => l.kind === "ram"));
});

test("kinds too small to see on the card stay out of the legend", () => {
  const rows = view.rows.map((r, i) => i === 1 ? {...r, parts: {...r.parts, norm: 1 * M}} : r);
  assert.ok(!X.xrayCardText({...view, rows}, "m").legend.some(l => l.kind === "norm"));
});

test("a tied output's GPU copy is said, and a guess is marked", () => {
  assert.match(X.xrayCardText({...view, tied_copy: 748 * M}, "m").gpu, /\+ 748 MB tied copy/);
  assert.match(X.xrayCardText({...view, confident: false}, "m").note, /guess/);
  assert.match(X.xrayCardText({...view, confident: true}, "m").note, /GGUF header/);
});

const other = {
  ...view, file: "m-Q8_0.gguf", total: 6 * G, gpu: 4 * G, ram: 2 * G,
  types: [{type: "Q8_0", bytes: 6 * G, share: 1}],
  rows: view.rows.map(r => ({...r, gpu: r.gpu * 2, ram: r.ram * 2,
    parts: Object.fromEntries(Object.entries(r.parts).map(([k, b]) => [k, b * 2])),
    ram_parts: Object.fromEntries(Object.entries(r.ram_parts).map(([k, b]) => [k, b * 2]))})),
};
const pair = {...view, file: "m-Q4_K_M.gguf", vs: other};

test("the two-up card leads with what the other quant costs", () => {
  const t = X.compareCardText(pair, "gemma-4-26b");
  assert.equal(t.kicker, "WHICH QUANT FITS?");
  assert.equal(t.delta, "Q8_0 vs Q4_K_M: +3.00 GB of weights, +2.00 GB on GPU, +1.00 GB in system RAM");
  assert.equal(t.sub, "gemma4  ·  2 blocks  ·  GPU layers: all 2 blocks + output");
  assert.deepEqual(t.halves.map(h => h.name), ["Q4_K_M", "Q8_0"]);
  assert.equal(t.halves[1].line, "6.00 GB  ·  4.00 GB GPU  ·  2.00 GB RAM");
  assert.equal(t.halves[1].types, "Q8_0 100%");
});

test("both halves share one scale, so the bigger quant stands taller", () => {
  const [a, b] = X.compareCardText(pair, "m").halves;
  assert.deepEqual(a.cols.map(c => c.h), [0.25, 0.5, 0.5, 0.25]);
  assert.deepEqual(b.cols.map(c => c.h), [0.5, 1, 1, 0.5]);
});

test("the two-up card marks a guess from either side and keeps no paths", () => {
  assert.match(X.compareCardText({...pair, vs: {...other, confident: false}}, "m").note, /guess/);
  assert.match(X.compareCardText(pair, "m").note, /Headers only, nothing downloaded/);
  const t = X.compareCardText({...pair, model: "D:/secret/m.gguf", vs: {...other, file: "sub/m-Q8_0.gguf"}}, "m");
  assert.doesNotMatch(JSON.stringify(t), /secret|sub\//);
});

test("every kind has a colour and the card keeps no paths", () => {
  for (const [k] of X.KIND_ORDER) assert.match(X.KIND_COLORS[k], /^#[0-9a-f]{6}$/i, k);
  const t = X.xrayCardText({...view, model: "D:/secret/m.gguf"}, "m");
  assert.doesNotMatch(JSON.stringify(t), /secret/);
});

test("a view with a flag line puts the flags on the card", () => {
  const t = X.xrayCardText({...view, flags: "-ngl 3 --n-cpu-moe 1"}, "m");
  assert.equal(t.layers, "Load with -ngl 3 --n-cpu-moe 1");
  const c = X.compareCardText({...pair, flags: "-ngl 3 --n-cpu-moe 1"}, "m");
  assert.equal(c.sub, "gemma4  ·  2 blocks  ·  Load with -ngl 3 --n-cpu-moe 1");
});
