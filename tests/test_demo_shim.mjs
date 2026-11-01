// docs/demo/shim.js answers the panel's fetch() from fixtures.json.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const src = readFileSync(new URL("../docs/demo/shim.js", import.meta.url), "utf8");
const FX = {
  "GET /api/state": {models: [1]},
  "GET /api/forge/status?model=a": {stage: "done"},
  "POST /api/hub/search": {results: []},
};

function load() {
  const real = [];
  const window = {
    fetch: (u) => { real.push(u); return Promise.resolve(new Response(JSON.stringify(FX))); },
  };
  const document = {readyState: "complete", getElementById: () => null,
    createElement: () => ({setAttribute() {}, appendChild() {}, classList: {add() {}, remove() {}}}),
    head: {appendChild() {}}, body: {appendChild() {}}};
  vm.runInNewContext(src, {window, document, Response, Promise, JSON, TypeError, Date, String, setTimeout, clearTimeout});
  return {fetch: window.fetch, real};
}

test("reads come from the recording, by exact url then by path", async () => {
  const {fetch} = load();
  assert.deepEqual(await (await fetch("/api/state?x=1")).json(), {models: [1]});
  assert.deepEqual(await (await fetch("/api/forge/status?model=a")).json(), {stage: "done"});
  assert.deepEqual(await (await fetch("/api/forge/status?model=b")).json(), {});
});

test("recorded POSTs answer, every other change is refused with the demo message", async () => {
  const {fetch} = load();
  const s = await (await fetch("/api/hub/search", {method: "POST", body: "{}"})).json();
  assert.deepEqual(s, {results: []});
  const r = await (await fetch("/api/load", {method: "POST", body: "{}"})).json();
  assert.equal(r.ok, false);
  assert.equal(r.success, false);
  assert.equal(r.reason, r.error);
  assert.equal(r.demo, true);
  assert.match(r.error, /demo/);
});

test("the router's web UI is unreachable, other files pass through", async () => {
  const {fetch, real} = load();
  await assert.rejects(fetch("http://localhost:8098/", {mode: "no-cors"}));
  await fetch("web/icons/x.svg");
  assert.ok(real.includes("web/icons/x.svg"));
});
