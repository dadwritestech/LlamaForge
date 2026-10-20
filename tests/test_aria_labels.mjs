import assert from "node:assert/strict";
import { readdir, readFile } from "node:fs/promises";
import test from "node:test";

// Seven inputs use a placeholder as their only text label. Placeholders disappear
// when the user types and aren't a reliable accessible name for screen readers.
// Each must have an aria-label that describes the field's purpose.
//
// See GitHub issue #29.

const inputs = [
  { id: "model-search", file: "web/index.html" },
  { id: "hub-q", file: "web/js/discover.js" },
  { id: "hub-mode", file: "web/js/discover.js" },
  { id: "hub-sort", file: "web/js/discover.js" },
  { id: "wr-repo", file: "web/js/willrun.js" },
  { id: "wr-quant", file: "web/js/willrun.js" },
  { id: "wk-text", file: "web/js/context.js" },
];

test("all seven inputs have aria-labels", async () => {
  const base = new URL("../", import.meta.url);
  const missing = [];

  for (const { id, file } of inputs) {
    const content = await readFile(new URL(file, base), "utf8");
    // Match the element by id="..." and check for aria-label on the same tag
    const re = new RegExp(
      `<[a-z]+[^>]*\\bid=["']${id}["'][^>]*aria-label=["'][^"']+["']`,
      "i"
    );
    if (!re.test(content)) {
      missing.push(id);
    }
  }

  assert.deepEqual(missing, []);
});
