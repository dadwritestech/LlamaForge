import assert from "node:assert/strict";
import test from "node:test";
import { rigText, human } from "../web/js/rigcard.js";

const d = {gpus: ["RTX 5080 16 GB", "RTX 5060 Ti 16 GB"], ram_gb: 64, os: "Windows", version: "0.18.11",
  models: [{name: "qwen3.6-35b-a3b", tps: 131.6, generated: 30_000_000},
           {name: "gemma-4-12b", tps: 0, generated: 11_000_000}],
  generated: 41_234_567, hours: 312, link: "github.com/dadwritestech/LlamaForge"};

test("counts read like the stats tab", () => {
  assert.equal(human(41_234_567), "41.2M");
  assert.equal(human(1_500_000_000), "1.5B");
  assert.equal(human(12_345), "12.3k");
  assert.equal(human(7), "7");
});

test("every string on the card", () => {
  const t = rigText(d);
  assert.equal(t.kicker, "MY LOCAL AI RIG");
  assert.equal(t.brand, "LLAMAFORGE 0.18.11");
  assert.equal(t.hw, "RTX 5080 16 GB + RTX 5060 Ti 16 GB · 64 GB RAM · Windows");
  assert.equal(t.big, "41.2M");
  assert.equal(t.sub, "over 312 hours of loaded models");
  assert.deepEqual(t.rows[0], {name: "qwen3.6-35b-a3b", tps: "131.6", gen: "30.0M"});
  assert.equal(t.rows[1].tps, "–");          // no speed measured yet
  assert.equal(t.foot, "github.com/dadwritestech/LlamaForge");
});

test("a fresh install still makes a card", () => {
  const t = rigText({gpus: [], ram_gb: null, os: "Linux", version: "0.18.11", models: [], generated: 0, hours: 0});
  assert.equal(t.hw, "CPU only · Linux");
  assert.equal(t.big, "0");
  assert.deepEqual(t.rows, []);
  assert.match(t.empty, /No model usage/);
});
