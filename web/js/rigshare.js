// Rig card dialog: the machine, its most-used models and their measured
// speed, as an image or markdown the user posts themselves. /api/rig strips
// paths. main.js hands openRig to the Stats tab's button.
import { esc, api, toast } from "./core.js";
import { showModal } from "./models.js";
import { drawRig } from "./rigcard.js";

export async function openRig() {
  const d = await api("/api/rig").catch(e => ({error: String(e.message || e)}));
  if (!d || d.error) { toast((d && d.error) || "couldn't build the rig card", "err"); return; }
  showModal("Share your rig", `<canvas class="af-card" width="1200" height="630" aria-label="Rig card"></canvas>
    <p class="af-q">Holds your GPU and RAM sizes, OS, model names, their speed and token counts. No paths, keys or hosts.</p>
    <div class="af-act"><button type="button" class="qbtn load" data-rig-png>Download image</button>
      <button type="button" class="qbtn" data-rig-copyimg>Copy image</button>
      <button type="button" class="qbtn" data-rig-md>Copy markdown</button></div>
    <details class="af-miss"><summary>markdown</summary><pre class="af-md">${esc(d.markdown)}</pre></details>`);
  const dlg = document.querySelector("#modal-root dialog");
  const canvas = dlg && dlg.querySelector("canvas.af-card");
  if (!canvas) return;
  await drawRig(canvas, d);
  const blob = () => new Promise(r => canvas.toBlob(r, "image/png"));
  dlg.querySelector("[data-rig-png]").onclick = async () => {
    const url = URL.createObjectURL(await blob());
    const a = Object.assign(document.createElement("a"), {href: url, download: "llamaforge-rig.png"});
    a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  dlg.querySelector("[data-rig-copyimg]").onclick = async () => {
    try { await navigator.clipboard.write([new ClipboardItem({"image/png": blob()})]); toast("Image copied", "ok"); }
    catch { toast("This browser can't copy images; use Download", "err"); }
  };
  dlg.querySelector("[data-rig-md]").onclick = () =>
    navigator.clipboard.writeText(d.markdown).then(() => toast("Markdown copied", "ok"), () => toast("Couldn't copy", "err"));
}
