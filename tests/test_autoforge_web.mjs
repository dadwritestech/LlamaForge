import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const core = await readFile(new URL("../web/js/core.js", import.meta.url), "utf8");
const coreUrl = `data:text/javascript;base64,${Buffer.from(core).toString("base64")}`;
const source = (await readFile(new URL("../web/js/autoforge.js", import.meta.url), "utf8"))
  .replace('"./core.js"', `"${coreUrl}"`);
const af = await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);

const done = {
  stage: "done", running: false, model: "q<27b>", progress: {done: 5, total: 5},
  leaderboard: [
    {id: "w", label: "5080 alone + <DFlash2>", code: 131.6, prose: 70.2, prefill: 2100, vram_mib: 15360,
     vram_per_gpu: [15360, 0], finalist: true, sanity: {ok: true}},
    {id: "current", label: "your current setup", code: 52, prose: 30, prefill: 1500, vram_mib: 28000},
  ],
  failed: [{id: "f", label: "split 50/50, 128k", reason: "out of memory on GPU1 <kv>"}],
  rejected: [{id: "r", label: "5060 alone", reason: "needs 20 GB"}],
  result: {winner: {id: "w", label: "5080 alone + <DFlash2>", code: 131.6, prose: 70.2, spec: "dflash"},
           current: {code: 52, prose: 30}, vs_current: {code: 153.1, prose: 134, prefill: 40},
           unverified_quality: false},
  applied: null, req: {intent: "coding", ctx: 32768, time: "quick"},
};

test("percent labels", () => {
  assert.equal(af.pct(153.1), "+153%");
  assert.equal(af.pct(-4.6), "-5%");
  assert.equal(af.pct(0.2), "same");
  assert.equal(af.pct(null), "");
});

test("result card compares against the current setup and offers Apply", () => {
  const h = af.resultMarkup("q<27b>", done);
  assert.match(h, /131\.6 code<\/b> \(\+153%\)/);
  assert.match(h, /Same answers/);
  assert.match(h, /drafter checked/);
  assert.match(h, /data-af-apply="q&lt;27b&gt;"/);
  assert.doesNotMatch(h, /<DFlash2>/);           // escaped
});

test("applied result offers Undo instead", () => {
  const h = af.resultMarkup("m", {...done, applied: {undo: {}}});
  assert.match(h, /data-af-undo/);
  assert.doesNotMatch(h, /data-af-apply/);
});

test("lossy winner says answers can differ", () => {
  const h = af.resultMarkup("m", {...done, result: {...done.result, unverified_quality: true}});
  assert.match(h, /lossy KV cache/);
});

test("current setup winning has nothing to apply", () => {
  const h = af.resultMarkup("m", {...done, result: {...done.result, winner: {id: "current", label: "your current setup"}}});
  assert.doesNotMatch(h, /data-af-apply/);
  assert.match(h, /already the best/);
});

test("no winner explains why", () => {
  const h = af.resultMarkup("m", {...done, result: {winner: null, reason: "no candidate loaded"}});
  assert.match(h, /No candidate ran/);
});

test("leaderboard escapes labels and collapses misses with reasons", () => {
  const h = af.boardMarkup(done);
  assert.match(h, /&lt;DFlash2&gt;/);
  assert.match(h, /2 tried or skipped/);
  assert.match(h, /out of memory on GPU1 &lt;kv&gt;/);
  assert.match(h, /needs 20 GB/);
  assert.match(h, /class="af-cur"/);
  assert.match(h, /15\.0/);                       // GB
});

test("live view while running: stage, progress, cancel, no setup", () => {
  const st = {...done, running: true, stage: "fit", loading: "split <a>", progress: {done: 2, total: 8}, result: null};
  const h = af.liveMarkup("m", st);
  assert.match(h, /trying every candidate/);
  assert.match(h, /loading split &lt;a&gt;/);
  assert.match(h, /aria-valuenow="25"/);
  assert.match(h, /data-af-cancel="m"/);
});

test("setup sheet keeps the last request", () => {
  const h = af.setupMarkup("m", {intent: "long", ctx: 65536, time: "thorough", allow_loss: true});
  assert.match(h, /value="long" selected/);
  assert.match(h, /value="65536" selected/);
  assert.match(h, /value="thorough" selected/);
  assert.match(h, /data-af-loss checked/);
});

test("no-vision: setup checkbox, row tag and result warning", () => {
  assert.match(af.setupMarkup("m", {drop_vision: true}), /data-af-novision checked/);
  assert.doesNotMatch(af.setupMarkup("m", {}), /data-af-novision checked/);
  const st = structuredClone(done);
  st.leaderboard[0].novision = true;
  st.result.drops_vision = true;
  const html = af.liveMarkup("m", st);
  assert.match(html, />no vision</);
  assert.match(html, /Turns vision off/);
  assert.doesNotMatch(af.liveMarkup("m", done), /no vision|Turns vision off/);
});

test("apply and undo tell the editor its settings changed; cancel doesn't", async () => {
  globalThis.document = {querySelector: () => null};
  globalThis.fetch = async (path) => ({json: async () => path.includes("/status") ? {stage: "done"} : {ok: true}});
  const changed = [];
  af.initAutoForge(() => {}, id => changed.push(id));
  await af.act("/api/forge/apply", "m", "");
  await af.act("/api/forge/undo", "m", "");
  await af.act("/api/forge/cancel", "m", "");
  assert.deepEqual(changed, ["m", "m"]);
  globalThis.fetch = async () => ({json: async () => ({error: "nope"})});
  await af.act("/api/forge/apply", "m", "");
  assert.deepEqual(changed, ["m", "m"]);           // a failed apply changed nothing
});
