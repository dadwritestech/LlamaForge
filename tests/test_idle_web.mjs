import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const core = await readFile(new URL("../web/js/core.js", import.meta.url), "utf8");
const coreUrl = `data:text/javascript;base64,${Buffer.from(core).toString("base64")}`;
const src = (await readFile(new URL("../web/js/idle.js", import.meta.url), "utf8"))
  .replace('"./core.js"', `"${coreUrl}"`);
const { isUp, idleMarkup } = await import(`data:text/javascript;base64,${Buffer.from(src).toString("base64")}`);

test("a sleeping llama.cpp model counts as up, vLLM and unloaded don't", () => {
  assert.equal(isUp({status: "loaded"}), true);
  assert.equal(isUp({status: "sleeping"}), true);
  assert.equal(isUp({status: "sleeping", backend: "vllm"}), false);
  assert.equal(isUp({status: "unloaded"}), false);
  assert.equal(isUp({status: "loading"}), false);
  assert.equal(isUp(null), false);
});

test("the card shows the minutes set, or says the build can't", () => {
  const on = idleMarkup({supported: true, minutes: 15});
  assert.match(on, /id="idle-on"[^>]*checked/);
  assert.match(on, /id="idle-min"[^>]*value="15"/);
  const off = idleMarkup({supported: true, minutes: 0});
  assert.doesNotMatch(off, /id="idle-on"[^>]*checked/);
  assert.match(off, /value="15"/);                    // a sensible default to switch on with
  const no = idleMarkup({supported: false, minutes: 0});
  assert.match(no, /Build/);
  assert.doesNotMatch(no, /idle-save/);
  assert.equal(idleMarkup(null), "");
});
