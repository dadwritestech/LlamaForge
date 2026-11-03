// "Try a llama.cpp PR" view helpers: the look card, the progress line, the pin list.
import { test } from "node:test";
import assert from "node:assert/strict";
import { lookCard, phaseLine, pinChoices } from "../web/js/prview.js";

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
