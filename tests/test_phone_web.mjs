import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const inline = async (name, deps = {}) => {
  let src = await readFile(new URL(`../web/js/${name}`, import.meta.url), "utf8");
  for (const [from, to] of Object.entries(deps)) src = src.replace(`"./${from}"`, `"${to}"`);
  return `data:text/javascript;base64,${Buffer.from(src).toString("base64")}`;
};
const stub = `data:text/javascript;base64,${Buffer.from(
  "export const showModal = () => {}; export const qrSVG = () => '';").toString("base64")}`;
const core = await inline("core.js");
const { linkLabel, linkNote } =
  await import(await inline("phone.js", {"core.js": core, "models.js": stub, "qr.js": stub}));

test("each link kind has a tab label; the HTTPS one says it carries the mic", () => {
  assert.equal(linkLabel("lan"), "Wi-Fi / LAN");
  assert.equal(linkLabel("tailnet"), "Tailscale");
  assert.match(linkLabel("https"), /HTTPS/);
});

test("plain-HTTP links say Talk needs the HTTPS one", () => {
  assert.match(linkNote({kind: "lan"}, ""), /same Wi-Fi/);
  assert.match(linkNote({kind: "lan"}, ""), /mic needs HTTPS/);
  assert.match(linkNote({kind: "tailnet"}, "tailscale serve --bg 8092"), /Tailscale HTTPS/);
});

test("the HTTPS link names the one-time tailscale serve command, escaped", () => {
  const out = linkNote({kind: "https"}, "tailscale serve --bg 8092");
  assert.match(out, /<code>tailscale serve --bg 8092<\/code>/);
  assert.match(out, /Talk/);
  assert.doesNotMatch(linkNote({kind: "https"}, "<b>x</b>"), /<b>x<\/b>/);
});
