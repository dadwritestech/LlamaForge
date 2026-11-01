// Sleep idle models: llama.cpp's --sleep-idle-seconds for every model (idle.py).
// A sleeping model has freed its memory but stays in the list; the next request
// wakes it, so it counts as up for Chat and Unload.
import { esc } from "./core.js";

export function isUp(m) {
  if (!m) return false;
  if (m.status === "loaded") return true;
  return m.status === "sleeping" && (m.backend || "llamacpp") === "llamacpp";
}

export function idleMarkup(d) {
  if (!d) return "";
  const head = `<div class="card" id="idle-card"><h3>Sleep idle models</h3>
      <div class="note" style="margin-top:0">A model nobody has used for a while frees its VRAM and RAM but stays in the list as <b>sleeping</b>. The next request wakes it in a second or a few, from the OS file cache.</div>`;
  if (!d.supported)
    return `${head}<div class="note">This llama.cpp build has no <code>--sleep-idle-seconds</code>. Update it in Build to use this.</div></div>`;
  const on = d.minutes > 0, mins = on ? d.minutes : 15;
  return `${head}
      <div class="kv"><label class="k" for="idle-on">sleep models after they sit idle</label><span class="v"><input type="checkbox" id="idle-on" ${on ? "checked" : ""}></span></div>
      <div class="kv"><label class="k" for="idle-min">minutes idle</label><span class="v"><input type="number" id="idle-min" min="1" max="1440" step="1" value="${esc(mins)}" style="width:80px"></span></div>
      <div class="actions"><button id="idle-save">Save</button><span class="msg" id="idle-msg"></span></div>
      <div class="note">Applies to every llama.cpp model from its next load. Game mode and Elastic VRAM still work as before.</div>
    </div>`;
}
