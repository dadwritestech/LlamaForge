// "Try a llama.cpp PR" view helpers: the look card, the progress line, the pin list, the A/B.
import { test } from "node:test";
import assert from "node:assert/strict";
import { lookCard, phaseLine, pinChoices, abView } from "../web/js/prview.js";

const SHA = "0123456789abcdef0123456789abcdef01234567";
const info = {number: 1234, title: "CUDA: faster MMQ", author: "someone", sha: SHA,
              state: "open", draft: false, url: "https://github.com/ggml-org/llama.cpp/pull/1234"};

test("an open PR is fine to build, at the commit shown", () => {
  const c = lookCard(info);
  assert.equal(c.heading, "PR #1234: CUDA: faster MMQ");
  assert.equal(c.byline, "by someone at 0123456");
  assert.equal(c.warn, "");
});

test("merged, closed and draft PRs say why you might not bother", () => {
  assert.match(lookCard({...info, state: "merged"}).warn, /already merged.*nightly/);
  assert.match(lookCard({...info, state: "closed"}).warn, /closed without merging/);
  assert.match(lookCard({...info, draft: true}).warn, /draft/);
});

test("the progress line follows the build's phase", () => {
  assert.equal(phaseLine(null), "");
  assert.equal(phaseLine({phase: "idle"}), "");
  assert.match(phaseLine({running: true, phase: "fetch", pr: 1234}), /^Fetching PR #1234/);
  assert.match(phaseLine({running: true, phase: "configure", pr: 1234}), /^Configuring/);
  assert.match(phaseLine({running: true, phase: "build", pr: 1234}), /^Compiling.*minutes/);
  assert.match(phaseLine({phase: "done", pr: 1234, install: "pr-1234-0123456"}), /^Built PR #1234\. Pin a model/);
  assert.equal(phaseLine({phase: "failed", pr: 1234, error: "cmake build failed"}),
               "PR #1234 didn't build: cmake build failed. The log below says why.");
  assert.equal(phaseLine({phase: "cancelled", pr: 1234}), "Cancelled.");
});

test("pin choices: every model, with the ones already on this build marked", () => {
  const models = [{id: "qwen"}, {id: "gemma"}, {id: "*"}, {}];
  const inst = {name: "pr-1234-0123456", models: ["qwen"]};
  assert.deepEqual(pinChoices(models, inst), [{id: "gemma", pinned: false}, {id: "qwen", pinned: true}]);
  assert.deepEqual(pinChoices(null, inst), []);
  assert.deepEqual(pinChoices([{id: "a"}], {}), [{id: "a", pinned: false}]);
});

const AB = {running: false, phase: "done", pr: 1234, install: "pr-1234-0123456", model: "qwen",
  base: {pp: {ts: 5000, sd: 10}, tg: {ts: 100, sd: 0.5}},
  pr_result: {pp: {ts: 5030, sd: 9}, tg: {ts: 110.04, sd: 0.5}},
  delta: {pp: {pct: 0.6, noise: true}, tg: {pct: 10.0, noise: false}}};

test("A/B: nothing yet, running, failed", () => {
  assert.equal(abView(null).phase, "");
  assert.equal(abView({phase: "idle"}).phase, "");
  assert.match(abView({running: true, phase: "base", model: "qwen", pr: 1234}).phase,
               /^Benchmarking qwen on your build/);
  assert.match(abView({running: true, phase: "pr", model: "qwen", pr: 1234}).phase,
               /^Benchmarking qwen on PR #1234/);
  assert.equal(abView({phase: "failed", error: "llama-bench on the PR build failed"}).phase,
               "The comparison failed: llama-bench on the PR build failed.");
  assert.deepEqual(abView({phase: "failed"}).rows, []);
});

test("A/B done: both rows, the change, and noise said out loud", () => {
  const v = abView(AB);
  assert.equal(v.heading, "PR #1234 vs your build, qwen");
  assert.deepEqual(v.rows, [
    {label: "Prompt (pp512)", base: "5000.0 ± 10.0", pr: "5030.0 ± 9.0", change: "+0.6%", noise: true},
    {label: "Generation (tg128)", base: "100.0 ± 0.5", pr: "110.0 ± 0.5", change: "+10.0%", noise: false}]);
  assert.match(v.markdown, /^\*\*llama\.cpp PR #1234 vs my build\*\*, qwen, llama-bench pp512 \/ tg128, 3 runs each/);
  assert.match(v.markdown, /\| tg128 \| 100\.0 ± 0\.5 \| 110\.0 ± 0\.5 \| \+10\.0% \|/);
  assert.match(v.markdown, /\| pp512 \| .* \| \+0\.6% \(within noise\) \|/);
  assert.equal(abView({...AB, delta: {pp: {pct: -3.2, noise: false}, tg: {pct: 0, noise: true}}}).rows[0].change, "-3.2%");
});
