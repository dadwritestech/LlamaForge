import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const core = await readFile(new URL("../web/js/core.js", import.meta.url), "utf8");
const coreUrl = `data:text/javascript;base64,${Buffer.from(core).toString("base64")}`;
const src = (await readFile(new URL("../web/js/tmpldoctor.js", import.meta.url), "utf8"))
  .replace('"./core.js"', `"${coreUrl}"`);
const { templateBar, templateMarkup } = await import(`data:text/javascript;base64,${Buffer.from(src).toString("base64")}`);

const base = {embedded: true, repo: "unsloth/G-GGUF", file: "G.gguf", base: "google/g", repo_diff: "", base_diff: "", error: "", override: null};

test("a re-uploaded template is the loud case, with its diff escaped and a fix button", () => {
  const html = templateMarkup("m<1>", {...base, verdict: "repo_newer", repo_diff: "-old\n+<tools>"});
  assert.match(html, /unsloth\/G-GGUF has a newer chat template than your file/);
  assert.match(html, /data-tmpl-apply="m&lt;1&gt;" data-tmpl-source="repo"/);
  assert.ok(!html.includes("<tools>"));
  assert.match(html, /\+&lt;tools&gt;/);
});

test("a base-model difference is only a hint", () => {
  const html = templateMarkup("m", {...base, verdict: "base_differs", base_diff: "+x"});
  assert.match(html, /differs from google\/g/);
  assert.match(html, /on purpose/);
  assert.match(html, /data-tmpl-source="base"/);
  assert.doesNotMatch(html, /data-tmpl-source="repo"/);
});

test("same, unknown, missing template and an override say so", () => {
  assert.match(templateMarkup("m", {...base, verdict: "same"}), /Up to date: matches unsloth\/G-GGUF and google\/g/);
  assert.match(templateMarkup("m", {...base, verdict: "unknown", repo: "", error: "not downloaded from a Hugging Face repo"}), /not downloaded/);
  assert.match(templateMarkup("m", {...base, verdict: "same", embedded: false}), /no chat template of its own/);
  assert.equal(templateMarkup("m", null), "");
});

test("the bar shows the override and a way back", () => {
  const html = templateBar({id: "m", in_ini: true, settings: {"chat-template-file": "D:/lf/chat-templates/m.repo.jinja"}});
  assert.match(html, /data-tmpl-check="m"/);
  assert.match(html, /m\.repo\.jinja/);
  assert.match(html, /data-tmpl-source=""/);
  assert.match(templateBar({id: "m", in_ini: true, settings: {}}), /Check/);
  assert.equal(templateBar({id: "m", in_ini: false}), "");
});
