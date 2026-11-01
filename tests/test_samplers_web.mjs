import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const core = await readFile(new URL("../web/js/core.js", import.meta.url), "utf8");
const coreUrl = `data:text/javascript;base64,${Buffer.from(core).toString("base64")}`;
const src = (await readFile(new URL("../web/js/samplers.js", import.meta.url), "utf8"))
  .replace('"./core.js"', `"${coreUrl}"`);
const { samplerBar, samplerMarkup, changed } = await import(`data:text/javascript;base64,${Buffer.from(src).toString("base64")}`);

const r = {source: "Qwen/Q", url: "https://huggingface.co/Qwen/Q/blob/main/generation_config.json",
  values: {"temp": "1", "top-k": "20", "top-p": "0.95"}, current: {"top-k": "20", "temp": "0.7"}, inherited: ["temp"], checked: ["a/b", "Qwen/Q"]};

test("only knobs that differ from what runs now are applied", () => {
  assert.deepEqual(changed(r), ["temp", "top-p"]);
  assert.deepEqual(changed({...r, current: {"temp": "1.0", "top-k": "20", "top-p": "0.950"}}), []);
  assert.deepEqual(changed({...r, current: {"temp": "", "top-k": "x", "top-p": "0.95"}}), ["temp", "top-k"]);
});

test("the note cites the source and marks matching knobs dim", () => {
  const html = samplerMarkup("m<1>", r);
  assert.match(html, /href="https:\/\/huggingface.co\/Qwen\/Q\/blob\/main\/generation_config.json"/);
  assert.match(html, /class="ctxpill same" title="yours: 20">top-k 20/);
  assert.match(html, /title="yours: 0.7 \(from \[\*\]\)">temp 1/);
  assert.match(html, /title="yours: default">top-p 0.95/);
  assert.match(html, /2 differ from yours/);
  assert.match(html, /data-smp-apply="m&lt;1&gt;"/);
  assert.match(samplerMarkup("m", {...r, current: r.values}), /Yours already match/);
});

test("nothing published says where it looked", () => {
  const html = samplerMarkup("m", {values: {}, checked: ["a/b", "<x>/y"], error: ""});
  assert.match(html, /No published samplers/);
  assert.match(html, /checked a\/b, &lt;x&gt;\/y/);
  assert.equal(samplerMarkup("m", null), "");
  assert.match(samplerMarkup("m", {checking: true}), /reading/);
});

test("the bar needs a models.ini section", () => {
  assert.match(samplerBar({id: "m", in_ini: true}), /data-smp-check="m"/);
  assert.equal(samplerBar({id: "m", in_ini: false}), "");
});
