import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const inline = async (name, deps = {}) => {
  let src = await readFile(new URL(`../web/js/${name}`, import.meta.url), "utf8");
  for (const [from, to] of Object.entries(deps)) src = src.replace(`"./${from}"`, `"${to}"`);
  return `data:text/javascript;base64,${Buffer.from(src).toString("base64")}`;
};
const core = await inline("core.js");
const wav = await inline("wav.js");
const listen = await inline("listen.js", {"core.js": core, "wav.js": wav});
const { talkMarkup, turnMarkup, historyOf, pickTalkModel, timings } =
  await import(await inline("talk.js", {"core.js": core, "listen.js": listen}));

const ST = (models) => ({models, offer: {}, download: {}});

test("no model that can hear: no Talk card (Speech to text offers one)", () => {
  assert.equal(talkMarkup(ST([]), true), "");
  assert.equal(talkMarkup(null, true), "");
});

test("the card has the Talk button, hands-free and a fresh log", () => {
  const out = talkMarkup(ST(["gemma-4-12b"]), true);
  assert.match(out, /id="talk-go"/);
  assert.match(out, /id="talk-hands"/);
  assert.match(out, /id="talk-log"/);
  assert.doesNotMatch(out, /id="talk-model"/);        // one model: nothing to pick
  assert.doesNotMatch(out, /Set up Speak/);
});

test("several models give a picker; without speech the answers are text", () => {
  const out = talkMarkup(ST(["a", "<b>"]), false);
  assert.match(out, /id="talk-model"/);
  assert.match(out, /&lt;b&gt;/);
  assert.doesNotMatch(out, /<b>"/);
  assert.match(out, /Set up Speak/);
});

test("the saved pick wins while it is still there", () => {
  assert.equal(pickTalkModel(["a", "b"], "b"), "b");
  assert.equal(pickTalkModel(["a", "b"], "gone"), "a");
  assert.equal(pickTalkModel([], "x"), "");
});

test("a turn shows both sides, escaped, with the timings", () => {
  const t = {heard: "<hi>", reply: "Hello & welcome.", model: "gemma", hear_s: 0.42, think_s: 0.5, speak_s: 2.25};
  const out = turnMarkup(t);
  assert.match(out, /&lt;hi&gt;/);
  assert.match(out, /Hello &amp; welcome\./);
  assert.match(out, /gemma/);
  assert.equal(timings(t), "hear 0.4 s · think 0.5 s · speak 2.3 s");
  assert.equal(timings({...t, speak_s: null}), "hear 0.4 s · think 0.5 s");
});

test("history keeps the last turns as chat messages", () => {
  const turns = Array.from({length: 9}, (_, i) => ({heard: "q" + i, reply: "a" + i}));
  turns.push({heard: "unanswered", reply: ""});
  const h = historyOf(turns, 3);
  assert.equal(h.length, 6);
  assert.deepEqual(h.slice(-2), [{role: "user", content: "q8"}, {role: "assistant", content: "a8"}]);
  assert.deepEqual(historyOf([], 3), []);
});
