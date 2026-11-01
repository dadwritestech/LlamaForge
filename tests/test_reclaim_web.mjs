import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const core = await readFile(new URL("../web/js/core.js", import.meta.url), "utf8");
const coreUrl = `data:text/javascript;base64,${Buffer.from(core).toString("base64")}`;
const src = (await readFile(new URL("../web/js/reclaim.js", import.meta.url), "utf8"))
  .replace('"./core.js"', `"${coreUrl}"`);
const { gb, reclaimMarkup } = await import(`data:text/javascript;base64,${Buffer.from(src).toString("base64")}`);

const G = 1024 ** 3;

test("sizes read as GB with one decimal", () => {
  assert.equal(gb(21.46 * G), "21.5 GB");
  assert.equal(gb(0), "0.0 GB");
});

test("unused models and duplicates get a row and a delete button each, names escaped", () => {
  const html = reclaimMarkup({
    days: 90, tracked_long_enough: true, since: 0,
    stale: [{id: "big<b>", path: "D:/m/big.gguf", bytes: 20 * G, last_used: 0}],
    stale_bytes: 20 * G,
    dups: [{bytes: 5 * G, copies: 2, ids: ["a", "b"], files: [{path: "C:/x/a.gguf", ids: ["a"]}, {path: "D:/y/a.gguf", ids: ["b"]}]}],
    dup_bytes: 5 * G,
  });
  assert.match(html, /1 model not used in 90 days: 20\.0 GB/);
  assert.match(html, /never/);
  assert.ok(!html.includes("big<b>"));
  assert.match(html, /big&lt;b&gt;/);
  assert.match(html, /data-del="big&lt;b&gt;"/);
  assert.match(html, /2 copies of the same file: 5\.0 GB to free/);
  assert.match(html, /data-del="a"/);
  assert.match(html, /data-del="b"/);
});

test("young stats are named, and nothing found says so", () => {
  const young = reclaimMarkup({days: 90, tracked_long_enough: false, since: Date.UTC(2026, 9, 20) / 1000,
    stale: [{id: "m", path: "m.gguf", bytes: G, last_used: 0}], stale_bytes: G, dups: [], dup_bytes: 0});
  assert.match(young, /not used since usage tracking began \(2026-10-20\)/);
  const none = reclaimMarkup({days: 90, tracked_long_enough: true, since: 0, stale: [], stale_bytes: 0, dups: [], dup_bytes: 0});
  assert.match(none, /Every model was used in the last 90 days, and there are no duplicate files/);
  assert.equal(reclaimMarkup(null), "");
});
