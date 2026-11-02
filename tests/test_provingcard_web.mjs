import assert from "node:assert/strict";
import test from "node:test";
import { provingText } from "../web/js/provingcard.js";

const cats = ["extract", "math", "json", "follow", "reason"];
const tasks = cats.flatMap(c => [1, 2, 3].map(i => ({id: `${c}-${i}`, cat: c})));
const res = miss => Object.fromEntries(tasks.map((t, i) => [t.id, {pass: !miss.includes(i)}]));
const sc = (passed, tok_s) => ({passed, total: 15, tok_s, cats: {}});
const d = {gpus: ["RTX 5080", "RTX 5060 Ti"], categories: cats, tasks, driver: "qwen",
  scores: {qwen: sc(15, 115), tiny: sc(7, 230.4), e4b: sc(14, 155)},
  results: {qwen: res([]), tiny: res([0, 1, 4, 6, 9, 10, 12, 13]), e4b: res([10])}};

test("every string on the card, best first", () => {
  const t = provingText(d);
  assert.equal(t.kicker, "MY CHECKED TASKS");
  assert.equal(t.sub, "RTX 5080 + RTX 5060 Ti  ·  15 tasks, one try each  ·  scored by a script");
  assert.deepEqual(t.rows.map(r => r.name), ["qwen", "e4b", "tiny"]);
  assert.deepEqual(t.rows[2], {rank: "3", name: "tiny", passed: "7/15", speed: "230 tok/s",
    pips: tasks.map((_, i) => ![0, 1, 4, 6, 9, 10, 12, 13].includes(i))});
  assert.deepEqual(t.contrast, ["Fastest: tiny, 230 tok/s, 7 of 15 right.",
                                "Most right: qwen, 15 of 15 at 115 tok/s."]);
  assert.equal(t.groups, 5);                       // a gap between the five categories
  assert.equal(t.foot, "github.com/dadwritestech/LlamaForge");
  assert.match(t.note, /smoke test/);
});

test("top six only, and an empty board still draws", () => {
  const many = Object.fromEntries("abcdefg".split("").map(m => [m, sc(10, 50)]));
  const t = provingText({...d, scores: many, results: {}});
  assert.equal(t.rows.length, 6);
  assert.equal(t.more, "+1 more");
  assert.deepEqual(t.rows[0].pips, []);            // no stored answers, no pips
  const e = provingText({});
  assert.deepEqual(e.rows, []);
  assert.equal(e.more, "");
  assert.match(e.empty, /No checked runs yet/);
  assert.equal(e.sub, "15 tasks, one try each  ·  scored by a script");
});

test("no contrast when the fastest is also the most right, or speed is unknown", () => {
  assert.deepEqual(provingText({...d, scores: {qwen: sc(15, 300), tiny: sc(7, 230)}}).contrast, []);
  assert.deepEqual(provingText({...d, scores: {qwen: sc(15, null), tiny: sc(7, null)}}).contrast, []);
  assert.deepEqual(provingText({...d, scores: {qwen: sc(15, 100)}}).contrast, []);
});

test("unknown speed reads as a dash", () => {
  assert.equal(provingText({...d, scores: {qwen: sc(15, null)}}).rows[0].speed, "–");
});
