import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const core = await readFile(new URL("../web/js/core.js", import.meta.url), "utf8");
const coreUrl = `data:text/javascript;base64,${Buffer.from(core).toString("base64")}`;
const src = (await readFile(new URL("../web/js/watchdog.js", import.meta.url), "utf8"))
  .replace('"./core.js"', `"${coreUrl}"`);
const { watchdogNote } = await import(`data:text/javascript;base64,${Buffer.from(src).toString("base64")}`);

const base = {armed: true, restarts: 0, last_restart: null, last_error: null,
              gave_up: false, error: null, suggestion: null};

test("nothing to say while the router never crashed", () => {
  assert.equal(watchdogNote(null), "");
  assert.equal(watchdogNote(base), "");
});

test("a restart shows a count and the time", () => {
  const h = watchdogNote({...base, restarts: 2, last_restart: 1767225600});
  assert.match(h, /restarted it 2 times/);
  assert.match(h, /slotnote dim/);
});

test("a restart that failed to start says why, escaped", () => {
  const h = watchdogNote({...base, restarts: 1, last_restart: 1767225600, last_error: "<b>no bin</b>"});
  assert.match(h, /once/);
  assert.match(h, /&lt;b&gt;no bin&lt;\/b&gt;/);
  assert.match(h, /slotnote warn/);
});

test("giving up shows the verdict, escaped", () => {
  const h = watchdogNote({...base, restarts: 5, gave_up: true, error: "out of <VRAM>", suggestion: "lower -ngl"});
  assert.match(h, /stopped restarting/);
  assert.match(h, /out of &lt;VRAM&gt;/);
  assert.match(h, /lower -ngl/);
});
