import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = await readFile(new URL("../web/js/stats-facts.js", import.meta.url), "utf8");
const facts = await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);
let moduleSerial = 0;

async function loadStatsModule() {
  const coreSource = await readFile(new URL("../web/js/core.js", import.meta.url), "utf8");
  const coreUrl = `data:text/javascript;base64,${Buffer.from(coreSource).toString("base64")}`;
  const factsUrl = `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`;
  const statsSource = (await readFile(new URL("../web/js/stats.js", import.meta.url), "utf8"))
    .replace('"./core.js"', `"${coreUrl}"`)
    .replace('"./stats-facts.js"', `"${factsUrl}"`);
  return import(`data:text/javascript;base64,${Buffer.from(statsSource).toString("base64")}#${++moduleSerial}`);
}

test("named comparisons use the generated-token English estimate", () => {
  const catalog = facts.buildTokenFacts(24_800_000);
  const harry = catalog.find(f => f.id === "book:harry-potter-1");
  const munich = catalog.find(f => f.id === "route:car:munich-frankfurt");

  assert.equal(harry.amount, 241.7);
  assert.match(harry.text, /copies of Harry Potter and the Philosopher's Stone/);
  assert.equal(munich.amount, 423.5);
  assert.match(munich.text, /Munich–Frankfurt trips by car/);
  assert.match(munich.assumption, /400 km/);
});

test("catalog provides a deep pool of concrete references", () => {
  const catalog = facts.buildTokenFacts(1_000_000);
  assert.ok(catalog.length >= 100);
  assert.ok(catalog.some(f => /complete Harry Potter series/.test(f.text)));
  assert.ok(catalog.some(f => /back-to-back screenings of Titanic/.test(f.text)));
  assert.ok(catalog.some(f => /Tokyo–Osaka/.test(f.text)));
  assert.ok(catalog.every(f => f.assumption.includes("0.75 words/token")));
});

test("fact rotator stays stable between polls and changes after fifteen minutes", () => {
  const picks = [0, 0, 0.5];
  const rotate = facts.createFactRotator({ random: () => picks.shift(), intervalMs: 900_000 });

  const first = rotate(1_000_000, 10_000);
  const polled = rotate(2_000_000, 14_000);
  const expired = rotate(2_000_000, 910_001);

  assert.equal(polled.id, first.id);
  assert.notEqual(polled.text, first.text);
  assert.notEqual(expired.id, first.id);
});

test("zero generated tokens has a useful empty state", () => {
  const rotate = facts.createFactRotator({ random: () => 0 });
  const fact = rotate(0, 0);

  assert.equal(fact.id, "empty");
  assert.match(fact.text, /Generate some tokens/);
});

test("small positive totals never rotate to a zero comparison", () => {
  for (const tokens of [1, 100, 1_000, 10_000]) {
    const rotate = facts.createFactRotator({ random: () => 0.999 });
    const fact = rotate(tokens, 0);
    assert.ok(fact.amount > 0, `${tokens} tokens produced ${fact.text}`);
    assert.doesNotMatch(fact.text, /^0(?:\.0+)?\s/);
  }
});

test("VRAM telemetry is normalized without trusting malformed counters", () => {
  assert.deepEqual(facts.normalizeVram({gpus: [
    {index: 0, name: "RTX <5090>", used: 8192, total: 32768, util: 75, temp: 61},
    {index: 1, name: "broken", used: -2, total: 0},
  ]}), [{
    index: 0,
    name: "RTX <5090>",
    used: 8192,
    total: 32768,
    free: 24576,
    util: 75,
    temp: 61,
  }]);
  assert.deepEqual(facts.normalizeVram({error: "offline"}), []);
});

test("compact VRAM cards retain text counters and escape GPU names", async () => {
  const stats = await loadStatsModule();
  const markup = stats.renderStatsVram({gpus: [
    {index: 0, name: "RTX <5090>", used: 8192, total: 32768, util: 75, temp: 61},
  ]});

  assert.match(markup, /RTX &lt;5090&gt;/);
  assert.match(markup, /8\.0<\/b>\/32\.0 GB/);
  assert.match(markup, /FREE <b>24\.0<\/b> GB/);
  assert.match(markup, /role="progressbar"/);
  assert.match(markup, /aria-valuenow="8192"/);
  assert.match(markup, /aria-valuemax="32768"/);
  assert.match(markup, /aria-valuetext="8\.0 of 32\.0 GB used"/);
  assert.equal((markup.match(/class="seg/g) || []).length, 28);
});

test("token scale exposes its full comparison and assumptions to touch and keyboard users", async () => {
  const stats = await loadStatsModule();
  const markup = stats.renderTokenScale(24_800_000, {
    text: "241.7 copies of Harry <Potter>",
    assumption: "English estimate: 0.75 words/token",
  });

  assert.match(markup, /<details class="token-scale-details">/);
  assert.match(markup, /<summary[^>]*>EST\.<\/summary>/);
  assert.match(markup, /241\.7 copies of Harry &lt;Potter&gt;/);
  assert.match(markup, /English estimate: 0\.75 words\/token/);
  assert.match(stats.renderTokenScale(24_800_000, {text: "fact", assumption: "basis"}, true), /<details class="token-scale-details" open>/);
});

test("a pending GPU probe does not hold back the core Stats view", async () => {
  const stats = await loadStatsModule();
  const view = {innerHTML: "", querySelector: () => null};
  globalThis.document = {querySelector: selector => selector === "#view-stats" ? view : null};
  globalThis.fetch = path => path === "/api/stats"
    ? Promise.resolve({json: () => Promise.resolve(statsPayload("FAST"))})
    : new Promise(() => {});

  await Promise.race([
    stats.loadStats(false),
    new Promise((_, reject) => setTimeout(() => reject(new Error("Stats waited for GPU telemetry")), 100)),
  ]);
  assert.match(view.innerHTML, /FAST/);
});

test("an older Stats response cannot overwrite a newer poll", async () => {
  const stats = await loadStatsModule();
  const view = {innerHTML: "", querySelector: () => null};
  globalThis.document = {querySelector: selector => selector === "#view-stats" ? view : null};
  let resolveOld, resolveNew, calls = 0;
  globalThis.fetch = path => {
    if (path === "/api/gpus") return Promise.resolve({json: () => Promise.resolve({gpus: []})});
    calls += 1;
    return new Promise(resolve => {
      const done = label => resolve({json: () => Promise.resolve(statsPayload(label))});
      if (calls === 1) resolveOld = done; else resolveNew = done;
    });
  };

  const oldPoll = stats.loadStats(true);
  const newPoll = stats.loadStats(true);
  resolveNew("NEWER");
  await newPoll;
  resolveOld("OLDER");
  await oldPoll;
  assert.match(view.innerHTML, /NEWER/);
  assert.doesNotMatch(view.innerHTML, /OLDER/);
});

test("a polling render restores keyboard focus to the estimate disclosure", async () => {
  const stats = await loadStatsModule();
  const oldSummary = {};
  let rendered = false, focusOptions = null;
  const newSummary = {focus: options => { focusOptions = options; }};
  const view = {
    _html: "",
    get innerHTML() { return this._html; },
    set innerHTML(value) { this._html = value; rendered = true; },
    querySelector: selector => {
      if (selector === ".token-scale-details") return {open: true};
      if (selector === ".token-scale-details summary") return rendered ? newSummary : oldSummary;
      return null;
    },
  };
  globalThis.document = {
    activeElement: oldSummary,
    querySelector: selector => selector === "#view-stats" ? view : null,
  };
  globalThis.fetch = path => Promise.resolve({json: () => Promise.resolve(
    path === "/api/stats" ? statsPayload("FOCUSED") : {gpus: []},
  )});

  await stats.loadStats(true);
  assert.deepEqual(focusOptions, {preventScroll: true});
});

function statsPayload(mostUsed) {
  return {
    totals: {tokens: 2_000, generated: 1_000, loaded_hours: 1, models_used: 1, total_runs: 2, most_used: mostUsed},
    live: {router_up: true, loaded_model: "model", gen_per_sec: 1, prompt_per_sec: 2, requests_processing: 0},
    per_model: [],
    daily: [],
  };
}
