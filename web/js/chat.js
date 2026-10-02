// Chat tab: llama.cpp's own web UI, served by backend/chatproxy.py on its own
// port (a separate origin, so untrusted model output can't reach /api). The
// iframe is created once and only hidden on tab switches, so a conversation
// survives browsing the rest of the panel.
import { $, esc, setHTML } from "./core.js";
import { S } from "./state.js";
import { on } from "./bus.js";
import { switchTab } from "./ui.js";

const LOOPBACK = new Set(["127.0.0.1", "localhost", "[::1]"]);
let pendingModel = null;

function chatBase() {
  const port = (S.STATE && S.STATE.chat_port) || 8091;
  return `${location.protocol}//${location.hostname}:${port}/`;
}

/** Open the Chat tab, optionally with a model preselected. */
export function openChat(model) {
  pendingModel = model || null;
  switchTab("chat");
}
on("chat", openChat);

async function reachable(url) {
  try { await fetch(url, {mode: "no-cors", cache: "no-store"}); return true; }
  catch (e) { return false; }
}

export async function loadChat() {
  const view = $("#view-chat");
  if (!view) return;
  const base = chatBase();
  const model = pendingModel; pendingModel = null;
  const src = model ? `${base}?model=${encodeURIComponent(model)}` : base;
  const frame = $("#chat-frame", view);
  if (frame) {                    // keep the conversation; only re-point on a deep link
    if (model) { frame.src = src; $(".chat-bar a", view).href = src; }
    return;
  }
  if (!LOOPBACK.has(location.hostname)) {
    setHTML(view, `<div class="card"><h3>Chat is local-only</h3>
      <div class="note">The chat listener binds 127.0.0.1 so the router key never leaves this PC.
      Open the panel as <code>http://127.0.0.1:${esc(location.port)}</code> on the machine running LlamaForge.</div></div>`);
    return;
  }
  if (!(await reachable(base))) {
    setHTML(view, `<div class="card"><h3>Chat isn't reachable</h3>
      <div class="note">Nothing answered on <code>${esc(base)}</code>. Another program may hold that port:
      set a free <code>chat_port</code> in config.json and restart LlamaForge.</div>
      <div class="actions"><button class="ghost" id="chat-retry">Retry</button></div></div>`);
    $("#chat-retry", view).onclick = () => { pendingModel = model; loadChat(); };
    return;
  }
  setHTML(view, `<div class="chat-bar">
      <span class="note" style="margin:0">llama.cpp's built-in chat &middot; updates with your engine</span>
      <a href="${esc(src)}" target="_blank" rel="noopener">open in its own window &#8599;</a></div>
    <iframe id="chat-frame" class="chat-frame" title="Chat" allow="clipboard-write" src="${esc(src)}"></iframe>`);
}
