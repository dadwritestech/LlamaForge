// One-click Claude Code on a folder.
//
// The Claude Code button on a loaded llama model opens this dialog: a readiness
// checklist, a folder (typed, picked with the OS dialog, or a recent one), and
// Launch. The backend opens a new terminal in that folder running `claude`,
// pointed at the panel's Anthropic endpoint with settings scoped to that one
// process; the user's own ~/.claude/settings.json is never touched.
import { $, $$, api, esc, setHTML, toast } from "./core.js";
import { on } from "./bus.js";
import { showModal } from "./models.js";

const LEVEL = {ok: "OK", warn: "Note", error: "Fix"};

let modal = null, info = null;

export function initClaudeCode() {
  on("claude-code", open);
}

async function open(id) {
  modal = showModal("Claude Code - " + id, `<div id="cc-body"><div class="note">Checking…</div></div>`);
  await check(id);
}

async function check(id) {
  const r = await api("/api/claude/check?model=" + encodeURIComponent(id)).catch(e => ({error: String(e)}));
  if (!modal || !modal.isOpen()) return;
  info = r;
  render();
}

function render(result) {
  const body = $("#cc-body");
  if (!body) return;
  const r = info;
  if (r.error) {
    setHTML(body, `<div class="note" role="alert">${esc(r.error)}</div>`);
    return;
  }
  const cliMissing = !r.cli.path;
  setHTML(body, `
    <div class="note">Opens a new terminal in the folder you pick, running Claude Code against
      <b>${esc(r.model)}</b>. The settings apply to that window only; your own Claude Code
      config isn't changed.</div>
    <ul class="cc-checks">${r.checks.map(c =>
      `<li><span class="lv ${esc(c.level)}">${esc(LEVEL[c.level] || c.level)}</span><span>${esc(c.text)}</span></li>`).join("")}
    </ul>
    ${cliMissing ? `<div class="slabel">Install Claude Code</div><div class="snip">
      <button type="button" class="qbtn scopy" data-cc-copy="hint" aria-label="Copy install command">Copy</button>${esc(r.install_hint)}</div>` : ""}
    <label class="f cc-folder-l"><span class="lbl">Folder</span>
      <span class="cc-folder">
        <input id="cc-folder" type="text" spellcheck="false" autocomplete="off"
               placeholder="${esc(r.recent[0] || "Path to a project folder")}" value="${esc(r.recent[0] || "")}">
        <button type="button" class="ghost" data-cc-browse>Browse…</button>
      </span></label>
    ${r.recent.length ? `<div class="cc-recent">${r.recent.map((d, i) =>
      `<button type="button" class="chip" data-cc-recent="${i}" title="${esc(d)}">${esc(shortName(d))}</button>`).join("")}</div>` : ""}
    <div class="actions cc-actions">
      <button type="button" class="ghost" data-cc-recheck>Check again</button>
      <button type="button" class="primary" data-cc-launch>Launch</button>
    </div>
    <div id="cc-result">${result || ""}</div>`);
  wire();
}

function shortName(path) {
  const parts = path.split(/[\\/]+/).filter(Boolean);
  return parts.slice(-2).join("/") || path;
}

function syncLaunch() {
  const btn = $("[data-cc-launch]"), folder = $("#cc-folder");
  if (btn && folder) btn.disabled = !info.ready || !folder.value.trim();
}

function copy(text) {
  navigator.clipboard.writeText(text).then(() => toast("Copied to clipboard", "ok"));
}

function wire() {
  const folder = $("#cc-folder");
  folder.addEventListener("input", syncLaunch);
  // Explorer's "Copy as path" wraps the path in quotes
  folder.addEventListener("paste", e => {
    const text = e.clipboardData?.getData("text") ?? "";
    const m = text.trim().match(/^(["'])(.*)\1$/s);
    if (!m) return;
    e.preventDefault();
    folder.setRangeText(m[2].trim(), folder.selectionStart, folder.selectionEnd, "end");
    syncLaunch();
  });
  folder.addEventListener("keydown", e => {
    if (e.key === "Enter" && !$("[data-cc-launch]").disabled) { e.preventDefault(); launch(); }
  });
  syncLaunch();
  $$("[data-cc-recent]").forEach(b => b.onclick = () => {
    folder.value = info.recent[Number(b.dataset.ccRecent)];
    syncLaunch();
  });
  const hint = $("[data-cc-copy=hint]");
  if (hint) hint.onclick = () => copy(info.install_hint);
  $("[data-cc-recheck]").onclick = () => check(info.model);
  $("[data-cc-launch]").onclick = launch;
  $("[data-cc-browse]").onclick = browse;
}

async function browse(e) {
  const btn = e.currentTarget;
  btn.disabled = true;
  btn.textContent = "Waiting for the folder dialog…";
  toast("Folder dialog opened: if you can't see it, check the taskbar. You can also type or paste a path.", "ok");
  const r = await api("/api/claude/pick-folder", {}).catch(() => ({path: ""}));
  if (!btn.isConnected) return;
  btn.disabled = false;
  btn.textContent = "Browse…";
  if (r.unsupported) { toast("No folder dialog on this system: type the path", "err"); return; }
  if (r.path) { $("#cc-folder").value = r.path; syncLaunch(); }
}

async function launch() {
  const btn = $("[data-cc-launch]");
  const folder = $("#cc-folder").value.trim();
  btn.disabled = true;
  btn.textContent = "Opening…";
  const r = await api("/api/claude/launch", {model: info.model, folder})
    .catch(e => ({error: String(e)}));
  if (!modal || !modal.isOpen()) return;
  let result;
  if (r.error) {
    result = `<div class="note cc-err" role="alert">${esc(r.error)}</div>`;
  } else {
    if (!info.recent.includes(folder)) info.recent = [folder, ...info.recent].slice(0, 8);
    result = r.launched
      ? `<div class="note">Claude Code is opening in a new terminal window. To run it by hand
          instead, <button type="button" class="qbtn" data-cc-copy="cmd">copy the command</button>
          (it includes the panel key).</div>`
      : `<div class="note">No terminal program was found to open. Run this in a terminal
          (it includes the panel key):</div><div class="snip">
          <button type="button" class="qbtn scopy" data-cc-copy="cmd" aria-label="Copy command">Copy</button>${esc(r.command)}</div>`;
    if (r.launched) toast("Claude Code launched", "ok");
  }
  render(result);
  $("#cc-folder").value = folder;
  syncLaunch();
  const cmd = $("[data-cc-copy=cmd]");
  if (cmd) cmd.onclick = () => copy(r.command);
}
