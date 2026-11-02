// Proving ground view helpers: board order, progress line, what a model got wrong.
import { test } from "node:test";
import assert from "node:assert/strict";
import { boardRows, progressLine, failures, CAT_LABELS } from "../web/js/provingview.js";

const cats = ["extract", "math", "json", "follow", "reason"];
const d = {
  categories: cats, driver: "qwen",
  tasks: [{id: "a", cat: "math"}, {id: "b", cat: "json"}, {id: "c", cat: "reason"}],
  scores: {
    qwen: {passed: 3, total: 3, tok_s: 50, at: 1, cats: {extract: [0, 0], math: [1, 1], json: [1, 1], follow: [0, 0], reason: [1, 1]}},
    gemma: {passed: 1, total: 3, tok_s: 25, at: 1, cats: {extract: [0, 0], math: [1, 1], json: [0, 1], follow: [0, 0], reason: [0, 1]}},
    tiny: {passed: 3, total: 3, tok_s: 200, at: 1, cats: {extract: [0, 0], math: [1, 1], json: [1, 1], follow: [0, 0], reason: [1, 1]}},
  },
  verdicts: {gemma: "gemma beats qwen on 0 of 5 categories (2 ties), 50% slower."},
  results: {gemma: {a: {pass: true}, b: {pass: false, cut: false, answer: "{\"name\": 1}"}, c: {pass: false, cut: true, answer: ""}}},
};

test("most passed first, faster breaks a tie, the main model is marked", () => {
  const rows = boardRows(d);
  assert.deepEqual(rows.map(r => r.model), ["tiny", "qwen", "gemma"]);
  assert.equal(rows[1].driver, true);
  assert.equal(rows[0].driver, false);
  assert.deepEqual(rows[2].cats, [[0, 0], [1, 1], [0, 1], [0, 0], [0, 1]]);
  assert.equal(rows[2].verdict, d.verdicts.gemma);
});

test("empty or broken payloads give no rows", () => {
  assert.deepEqual(boardRows({}), []);
  assert.deepEqual(boardRows(null), []);
  assert.deepEqual(boardRows({scores: {x: null}}), []);
});

test("progress says which model and how far, or nothing when idle", () => {
  assert.equal(progressLine({running: true, model: "gemma", done: 4, total: 30}), "Checking gemma: 4 of 30 answers");
  assert.equal(progressLine({running: true, model: null, done: 0, total: 30}), "Starting: 0 of 30 answers");
  assert.equal(progressLine({running: false, cancelled: true}), "Cancelled. Models that finished were kept.");
  assert.equal(progressLine({running: false}), "");
  assert.equal(progressLine(null), "");
});

test("failures list the missed tasks in task order with the answer and cut flag", () => {
  assert.deepEqual(failures(d, "gemma"), [
    {id: "b", cat: "json", answer: "{\"name\": 1}", cut: false},
    {id: "c", cat: "reason", answer: "", cut: true},
  ]);
  assert.deepEqual(failures(d, "qwen"), []);
  assert.deepEqual(failures({}, "x"), []);
});

test("every category has a label", () => {
  for (const c of cats) assert.ok(CAT_LABELS[c]);
});
