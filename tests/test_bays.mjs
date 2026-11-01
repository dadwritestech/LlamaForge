import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const core = await readFile(new URL("../web/js/core.js", import.meta.url), "utf8");
const coreUrl = `data:text/javascript;base64,${Buffer.from(core).toString("base64")}`;
const source = (await readFile(new URL("../web/js/bays.js", import.meta.url), "utf8"))
  .replace('"./core.js"', `"${coreUrl}"`);
const bays = await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);

const GiB = 1024;
const gpu = (index, used, total) => ({index, name: `GPU <${index}>`, used: used * GiB, total: total * GiB, util: 40, temp: 55});
const ledger = html => Object.fromEntries([...html.matchAll(/<dt[^>]*>(\w+)<\/dt><dd[^>]*>([\d.]+)<\/dd>/g)].map(m => [m[1], +m[2]]));

test("bayInputs reads footprints, the main and the loaded llama.cpp models from /api/state", () => {
  const st = {
    slots: {main: "big", footprints: {big: {"0": 20 * GiB}}},
    models: [
      {id: "big", status: "loaded"},
      {id: "nap", status: "sleeping"},
      {id: "cold", status: "unloaded"},
      {id: "vl", status: "loaded", backend: "vllm"},
    ],
  };
  assert.deepEqual(bays.bayInputs(st), {fps: {big: {"0": 20 * GiB}}, main: "big", up: ["big", "nap"]});
  assert.deepEqual(bays.bayInputs(null), {fps: {}, main: "", up: []});
});

test("without a booking, used + free = total and every model is stowed at its footprint", () => {
  const html = bays.bayPlans([gpu(0, 24, 32)], {
    fps: {big: {"0": 20 * GiB}}, main: "big", up: ["big"], seen: new Map(),
  });
  assert.match(html, /class="box main/);
  assert.match(html, /System \+ other apps/);
  assert.doesNotMatch(html, /booked/);
  const l = ledger(html);
  assert.equal(l.Used + l.Free, l.Total);
  assert.equal(l.Total, 32);
  assert.match(html, /GPU &lt;0&gt;/);      // names are escaped
});

test("a booking that does not fit hangs past the hold as Short deck cargo", () => {
  const html = bays.bayPlans([gpu(0, 24, 32)], {
    book: {footprint: {"0": 12 * GiB}}, bookId: "next", seen: new Map(),
  });
  assert.match(html, /class="box booked over/);
  assert.match(html, /class="deck"/);
  const l = ledger(html);
  assert.equal(l.Needs, 12);
  assert.equal(l.Short, 4);
});

test("a box drops in once per view: each caller's seen map is its own", () => {
  const opts = seen => ({fps: {big: {"0": 20 * GiB}}, main: "big", up: ["big"], seen});
  const models = new Map(), stats = new Map();
  assert.match(bays.bayPlans([gpu(0, 24, 32)], opts(models)), /stow/);
  // pretend Models drew it long ago: it stays still there, but Stats has never shown it
  for (const v of models.values()) v.t -= 10_000;
  assert.doesNotMatch(bays.bayPlans([gpu(0, 24, 32)], opts(models)), /box main[^"]*stow/);
  assert.match(bays.bayPlans([gpu(0, 24, 32)], opts(stats)), /box main[^"]*stow/);
});

test("no telemetry says so instead of drawing an empty hold", () => {
  assert.match(bays.bayPlans([], {}), /GPU telemetry unavailable/);
  assert.match(bays.bayPlans([{error: "nvidia-smi"}], {}), /GPU telemetry unavailable/);
});

test("cargoHue matches the bay plan: main orange, workers alternate by name, unmeasured gets none", () => {
  const fps = {zeta: {"1": GiB}, big: {"0": 8 * GiB}, alpha: {"1": GiB}};
  const hue = bays.cargoHue(fps, "big");
  assert.deepEqual(["big", "alpha", "zeta", "ghost"].map(hue), ["main", "w1", "w2", ""]);
  const html = bays.bayPlans([gpu(0, 10, 16), gpu(1, 4, 16)], {fps, main: "big", up: ["big", "alpha", "zeta"]});
  assert.match(html, /class="box w1[^"]*"[^>]*title="alpha/);
  assert.match(html, /class="box w2[^"]*"[^>]*title="zeta/);
});

test("an AMD/Intel card read through llama.cpp shows its source, not null util/temp", () => {
  const amd = {index: 0, name: "AMD Radeon RX 7900 XTX", used: 2 * GiB, total: 24 * GiB, util: null, temp: null, via: "Vulkan"};
  const html = bays.bayPlans([amd]);
  assert.match(html, /VIA <b>Vulkan<\/b>/);
  assert.doesNotMatch(html, /null/);
  assert.match(bays.gpuTele(gpu(0, 1, 16)), /UTIL <b>40%<\/b>/);
});
