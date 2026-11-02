// Arena recent prompts: newest first, no duplicates, capped, labels kept short.
import { test } from "node:test";
import assert from "node:assert/strict";
import { pushRecent, recentLabel, MAX_RECENT } from "../web/js/arenarecent.js";

test("newest prompt goes first and a repeat moves up instead of doubling", () => {
  let l = pushRecent([], "a");
  l = pushRecent(l, "b");
  l = pushRecent(l, "a");
  assert.deepEqual(l, ["a", "b"]);
});

test("blank prompts are ignored and whitespace is trimmed", () => {
  assert.deepEqual(pushRecent(["x"], "   "), ["x"]);
  assert.deepEqual(pushRecent([], "  hi \n"), ["hi"]);
});

test("the list is capped and junk from storage is dropped", () => {
  let l = [];
  for (let i = 0; i < MAX_RECENT + 5; i++) l = pushRecent(l, `p${i}`);
  assert.equal(l.length, MAX_RECENT);
  assert.equal(l[0], `p${MAX_RECENT + 4}`);
  assert.deepEqual(pushRecent([null, 3, "ok"], "new"), ["new", "ok"]);
  assert.deepEqual(pushRecent("not a list", "new"), ["new"]);
});

test("labels are one line and short", () => {
  assert.equal(recentLabel("fix\nthis   bug"), "fix this bug");
  const long = recentLabel("x".repeat(200));
  assert.ok(long.length <= 70 && long.endsWith("…"));
});
