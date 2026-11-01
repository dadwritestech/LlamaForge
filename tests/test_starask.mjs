import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const core = await readFile(new URL("../web/js/core.js", import.meta.url), "utf8");
const coreUrl = `data:text/javascript;base64,${Buffer.from(core).toString("base64")}`;
const src = (await readFile(new URL("../web/js/starask.js", import.meta.url), "utf8"))
  .replace('"./core.js"', `"${coreUrl}"`);
const sa = await import(`data:text/javascript;base64,${Buffer.from(src).toString("base64")}`);

test("asks after an Auto-Forge win that's worth it", () => {
  assert.equal(sa.askFor({kind: "forge", model: "qwen", x: 4.73}, null), "Auto-Forge just made qwen 4.7x faster.");
  assert.equal(sa.askFor({kind: "forge", model: "qwen", x: 1.04}, null), null);   // a rounding-error win isn't a moment
});

test("asks on the 5th load only", () => {
  assert.equal(sa.askFor({kind: "load"}, null, 4), null);
  assert.match(sa.askFor({kind: "load"}, null, 5), /5 models/);
  assert.equal(sa.askFor({kind: "load"}, null, 6), null);
});

test("never asks twice, whatever the answer was", () => {
  for (const stored of ["asked", "starred", "no"]) {
    assert.equal(sa.askFor({kind: "forge", model: "m", x: 3}, stored), null);
    assert.equal(sa.askFor({kind: "load"}, stored, 5), null);
  }
});

test("the ask is a plain link: no request leaves the panel", async () => {
  const text = await readFile(new URL("../web/js/starask.js", import.meta.url), "utf8");
  assert.doesNotMatch(text, /\bfetch\(|\bapi\(|XMLHttpRequest|sendBeacon/);
});
