// Router watchdog notice above the model list. The backend (watchdog.py)
// restarts a crashed router on its own; this only says that it happened, or
// that it stopped trying and why.
import { $, esc, setHTML } from "./core.js";

const times = n => n === 1 ? "once" : `${n} times`;
const clock = ts => new Date(ts * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"});

export function watchdogNote(w) {
  if (!w) return "";
  if (w.gave_up)
    return `<div class="slotnote warn"><b>The router keeps crashing, so LlamaForge stopped restarting it.</b> ` +
      `${esc(w.error || "")} ${esc(w.suggestion || "")}</div>`;
  if (!w.restarts) return "";
  const when = w.last_restart ? `, last at ${esc(clock(w.last_restart))}` : "";
  if (w.last_error)
    return `<div class="slotnote warn"><b>The router crashed</b> and LlamaForge tried to restart it ${times(w.restarts)}${when}. ` +
      `The last try failed: ${esc(w.last_error)}</div>`;
  return `<div class="slotnote dim">ROUTER CRASHED &middot; LlamaForge restarted it ${times(w.restarts)}${when}</div>`;
}

let shown = null;
export function renderWatchdog(w) {
  const el = $("#watchdog-banner");
  if (!el) return;
  const h = watchdogNote(w);
  if (h !== shown) { setHTML(el, h); shown = h; }
}
