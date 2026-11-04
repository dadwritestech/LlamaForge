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
