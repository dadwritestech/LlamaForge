// Phone dialog: turn on the phone remote (backend/phone.py) and pair a phone by
// scanning a QR code. The QR holds the LAN link with the token in its #fragment,
// so it only ever lives on this screen and the phone. main.js wires the button.
import { esc, api, toast, askYes } from "./core.js";
import { showModal } from "./models.js";
import { qrSVG } from "./qr.js";

let pick = 0;           // which link the QR shows: LAN first, then any tailnet

export const linkLabel = kind =>
  ({lan: "Wi-Fi / LAN", tailnet: "Tailscale", https: "Tailscale HTTPS (Talk)"})[kind] || kind;

/** What the chosen link reaches, and whether the phone's mic works on it
 *  (browsers give a page the mic only over HTTPS). */
export function linkNote(u, httpsCmd) {
  if (u.kind === "https") return `over Tailscale with HTTPS, so <b>Talk</b> can use the phone's mic.
    Run <code>${esc(httpsCmd)}</code> once on this PC first (HTTPS has to be on for your tailnet).`;
  const where = u.kind === "tailnet" ? "over Tailscale, from anywhere your phone is on your tailnet."
                                     : "on the same Wi-Fi as this PC.";
  return `${where} Chat works here; Talk's mic needs HTTPS: ${httpsCmd
    ? "use the <b>Tailscale HTTPS</b> link." : "see the Tailscale HTTPS link in the docs."}`;
}

export async function openPhone() {
  show(await api("/api/phone").catch(e => ({error: String(e.message || e)})));
}

async function post(body) {
  const d = await api("/api/phone", body).catch(e => ({error: String(e.message || e)}));
  if (d && d.error) { toast(d.error, "err"); return; }
  show(d);
}

function show(d) {
  if (!d || d.error) { toast((d && d.error) || "couldn't read the phone remote", "err"); return; }
  if (!d.enabled) {
    showModal("Chat from your phone", `
      <p class="af-q">Turns on a small chat page at port ${esc(d.port)} on this PC's network address.
        Scan a QR code and your phone chats with the models on your own GPUs.</p>
      <p class="af-q"><b>Anyone with the link can chat with your models</b>, so use it on networks you trust
        (home Wi-Fi, or <a href="https://tailscale.com/" target="_blank" rel="noopener">Tailscale</a> when you're out).
        The link can do nothing else: no settings, no loading or deleting models.</p>
      <p class="af-q">Windows may ask whether Python can use private networks. Allow it, or the phone can't reach this PC.</p>
      <div class="af-act"><button type="button" class="qbtn load" data-phone-on>Turn on</button></div>`);
    wire({"[data-phone-on]": () => post({enabled: true})});
    return;
  }
  const urls = d.urls || [];
  if (!urls.length) {
    showModal("Chat from your phone", `<p class="af-q">The phone remote is on, but this PC has no network address
      a phone could reach. Connect to Wi-Fi or Ethernet and open this again.</p>
      <div class="af-act"><button type="button" class="qbtn" data-phone-off>Turn off</button></div>`);
    wire({"[data-phone-off]": () => post({enabled: false})});
    return;
  }
  if (pick >= urls.length) pick = 0;
  const u = urls[pick];
  const bare = u.url.split("#")[0];
  const tabs = urls.length > 1 ? `<div class="af-act">${urls.map((x, i) =>
    `<button type="button" class="qbtn${i === pick ? " load" : ""}" data-phone-pick="${i}">${esc(linkLabel(x.kind))}</button>`).join("")}</div>` : "";
  showModal("Chat from your phone", `${tabs}
    <div class="phone-qr">${qrSVG(u.url, 240)}</div>
    <p class="af-q">Scan with your phone's camera. It opens <code>${esc(bare)}</code>
      ${linkNote(u, d.https_cmd || "")}</p>
    <p class="af-q">${d.running ? "" : "<b>The listener isn't running:</b> turn it off and on again. "}Anyone with this link
      can chat with your models. <b>New link</b> unpairs every phone that has the old one.</p>
    <div class="af-act"><button type="button" class="qbtn" data-phone-copy>Copy link</button>
      <button type="button" class="qbtn" data-phone-rotate>New link</button>
      <button type="button" class="qbtn" data-phone-off>Turn off</button></div>`);
  wire({
    "[data-phone-copy]": () => navigator.clipboard.writeText(u.url)
      .then(() => toast("Link copied: it works for anyone who has it", "ok"), () => toast("Couldn't copy", "err")),
    "[data-phone-rotate]": async () => {
      if (await askYes("Phones paired with the current link stop working until they scan the new one.",
                       {title: "Make a new link?", ok: "New link"})) post({rotate: true});
    },
    "[data-phone-off]": () => post({enabled: false}),
  });
  document.querySelectorAll("#modal-root [data-phone-pick]").forEach(b =>
    b.onclick = () => { pick = Number(b.dataset.phonePick); show(d); });
}

function wire(map) {
  const dlg = document.querySelector("#modal-root dialog");
  if (!dlg) return;
  for (const [sel, fn] of Object.entries(map)) {
    const b = dlg.querySelector(sel);
    if (b) b.onclick = fn;
  }
}
