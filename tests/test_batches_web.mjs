import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const core = await readFile(new URL("../web/js/core.js", import.meta.url), "utf8");
const coreUrl = `data:text/javascript;base64,${Buffer.from(core).toString("base64")}`;
const src = (await readFile(new URL("../web/js/batches.js", import.meta.url), "utf8"))
  .replace('"./core.js"', `"${coreUrl}"`);
const { progress, barWidths, statusText, pySnippet, batchesMarkup } =
  await import(`data:text/javascript;base64,${Buffer.from(src).toString("base64")}`);

const B = (o) => ({id: "batch_0123abcd", status: "in_progress", model: "qwen", created_at: 0,
                   input_filename: "q.jsonl", request_counts: {total: 10, completed: 3, failed: 1}, ...o});

test("progress counts answered and failed lines as done", () => {
  assert.deepEqual(progress(B()), {done: 4, total: 10, failed: 1});
  assert.deepEqual(progress({}), {done: 0, total: 0, failed: 0});
});

test("the bar splits answered from failed", () => {
  assert.deepEqual(barWidths(B()), [30, 10]);
  assert.deepEqual(barWidths(B({request_counts: {total: 3, completed: 1, failed: 2}})), [33, 67]);
  assert.deepEqual(barWidths({}), [0, 0]);
});

test("status words say what the batch is waiting on", () => {
  assert.equal(statusText(B({status: "validating", when: "idle"}), 20), "waiting for 20 quiet minutes");
  assert.equal(statusText(B({status: "validating", when: "now"})), "starting");
  assert.equal(statusText(B()), "4 of 10, 1 failed");
  assert.equal(statusText(B({status: "completed"})), "done: 9 answered, 1 failed");
  assert.equal(statusText(B({status: "failed", errors: {data: [{message: "model won't load"}]}})),
               "failed: model won't load");
  assert.equal(statusText(B({status: "cancelled"})), "cancelled at 4 of 10");
});

test("open batches can be cancelled, finished ones removed and downloaded", () => {
  const open = batchesMarkup({batches: [B()], quiet_minutes: 20}, "http://127.0.0.1:8090");
  assert.match(open, /data-batchcancel="batch_0123abcd"/);
  assert.doesNotMatch(open, /data-batchremove/);
  const done = batchesMarkup({batches: [B({status: "completed", output_file_id: "file-aa11bb22"})]}, "");
  assert.match(done, /href="\/api\/batch\/file\?id=file-aa11bb22" download="q_results.jsonl"/);
  assert.match(done, /data-batchremove/);
  assert.doesNotMatch(done, /data-batchcancel|Errors</);
});

test("names and errors are escaped", () => {
  const out = batchesMarkup({batches: [B({input_filename: "<img src=x>.jsonl", model: "<b>",
                                          status: "failed", errors: {data: [{message: "<script>"}]}})]}, "");
  assert.doesNotMatch(out, /<img src=x>|<b>&|<script>/);
  assert.match(out, /&lt;img src=x&gt;/);
});

test("the snippet points the OpenAI SDK at this panel", () => {
  assert.match(pySnippet("http://127.0.0.1:8090"), /base_url="http:\/\/127\.0\.0\.1:8090\/v1"/);
  assert.match(pySnippet(""), /client\.batches\.create\(/);
  assert.equal(batchesMarkup(null, ""), "");
});
