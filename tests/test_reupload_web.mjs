import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const core = await readFile(new URL("../web/js/core.js", import.meta.url), "utf8");
const coreUrl = `data:text/javascript;base64,${Buffer.from(core).toString("base64")}`;
const src = (await readFile(new URL("../web/js/reupload.js", import.meta.url), "utf8"))
  .replace('"./core.js"', `"${coreUrl}"`);
const { S, chip, chipSig, noteMarkup } = await import(`data:text/javascript;base64,${Buffer.from(src).toString("base64")}`);

const rec = {repo: "unsloth/Q-GGUF", path: "UD/Q.gguf", verdict: "reuploaded", title: "Fix <template>", date: "2026-10-30T12:00:00.000Z", hf_cache: false, error: ""};
const set = (r) => { S.data = {checked_at: Date.now() / 1000 - 3600, models: {m: r}}; };

test("a re-uploaded file gets a row badge and a Re-download button", () => {
  set(rec);
  assert.match(chip({id: "m"}), /re-uploaded/);
  assert.equal(chipSig({id: "m"}), "reuploaded");
  const html = noteMarkup({id: "m"});
  assert.match(html, /replaced this file on 2026-10-30/);
  assert.match(html, /Fix &lt;template&gt;/);
  assert.ok(!html.includes("<template>"));
  assert.match(html, /data-reup-fetch="m"/);
});

test("an HF-cache copy is updated with hf download, not a button", () => {
  set({...rec, hf_cache: true});
  const html = noteMarkup({id: "m"});
  assert.match(html, /hf download unsloth\/Q-GGUF UD\/Q.gguf/);
  assert.doesNotMatch(html, /data-reup-fetch/);
});

test("changed is soft, gone says so, current and unknown are silent", () => {
  set({...rec, verdict: "changed"});
  assert.equal(chip({id: "m"}), "");
  assert.match(noteMarkup({id: "m"}), /same size: often metadata only/);
  set({...rec, verdict: "gone"});
  assert.match(noteMarkup({id: "m"}), /no longer has this file/);
  for (const v of ["current", "unknown"]) { set({...rec, verdict: v}); assert.equal(noteMarkup({id: "m"}), ""); assert.equal(chip({id: "m"}), ""); }
  S.data = null;
  assert.equal(noteMarkup({id: "m"}), "");
  assert.equal(chipSig({id: "m"}), "");
});
