---
title: Phone Remote
section: guides
order: 15
---

# Phone Remote

Chat with the models on your PC from your phone. Click **Phone** in the top bar of the panel, turn it on, and scan the QR code with your phone's camera. The phone opens a small chat page served by LlamaForge: pick a model, ask, and the answer streams back from your own GPUs with its speed in tokens per second. Or press **Talk** and ask out loud: your PC hears you, answers, and the phone plays the answer ([Talk from your phone](#talk-from-your-phone)).

Nothing to install on the phone, no account, no cloud relay.

## What the phone can reach

The panel itself stays on `127.0.0.1` only: it can build llama.cpp and install packages, which no phone link should reach. The phone remote is a separate listener on port `8092` of every network address (`0.0.0.0`), and it serves exactly four things:

| Route | What it does |
|---|---|
| `GET /` | The chat page. It holds no data until it has a token. |
| `GET /api/status` | Model names (and which are loaded) plus GPU memory and load. |
| `POST /api/chat` | A plain chat with one model, streamed. Only the model name and text messages are passed on; tools, sampler settings and anything else are dropped. |
| `POST /api/talk` | A recording (WAV) in; what the PC heard, its answer, and the answer spoken as WAV out. Only the recording, the model name and the text of earlier turns are passed on. |

Everything else answers `404`. The router's API key is added on the PC, so the phone never sees it.

> [!WARNING]
> Anyone who has the link can chat with your models (and load one by picking it). Use it on networks you trust: your home Wi-Fi, or [Tailscale](https://tailscale.com/) when you're away. Don't forward port 8092 on your router to the internet.

## Pairing and unpairing

The QR code holds a link like `http://192.168.1.20:8092/#k=<token>`. The token sits after the `#`, which browsers never send over the network. The page moves it into the phone's storage and removes it from the address bar. Each request then sends it as a `Bearer` token, compared in constant time.

- **New link** makes a new token. Every phone with the old link stops working until it scans the new QR code.
- **Turn off** closes the port. Turning it on again keeps the same token, so paired phones keep working.

## Away from home with Tailscale

If this PC is on a [Tailscale](https://tailscale.com/) network, the dialog shows a second QR code for its tailnet address (`100.x.y.z`). Install Tailscale on the phone, sign in to the same tailnet, and the link works from anywhere, encrypted, with no ports opened on your router.

## Talk from your phone

**Talk** sits next to **Send**. Tap it, ask your question, and pause: the phone stops recording by itself after a second or so of quiet (or tap **Stop**). The recording goes to your PC, where the same model that can hear on the [Voice tab](voice.md#talk) (Gemma 4 E4B, for example) writes down what you said and answers it. If **Speak** is set up on the Voice tab, Qwen3-TTS reads the answer on the PC and the phone plays it; otherwise the answer shows as text. Talk turns and typed messages are one conversation.

Phones give a web page the microphone only over **HTTPS**, and the Wi-Fi link is plain HTTP. On that link Talk says so instead of failing. To get HTTPS without making or installing a certificate, let Tailscale put its own in front of the listener:

1. Turn on **HTTPS certificates** for your tailnet (Tailscale admin console, *DNS* page), and have Tailscale on the phone, signed in to the same tailnet.
2. On the PC, once: `tailscale serve --bg 8092` (use your `phone_port` if you changed it).
3. Open **Phone** in the panel and pick **Tailscale HTTPS (Talk)**. Its QR code opens `https://<your-pc>.<tailnet>.ts.net/`.

LlamaForge finds the name by asking `tailscale status --json`, so the tab only shows up when the Tailscale CLI is on this PC and connected. `tailscale serve` only reaches people on your tailnet; it is not `tailscale funnel`, which would publish the page to the internet.

## Troubleshooting

- **The phone can't open the page.** Windows may ask whether Python can use private networks; allow it. If you said no, allow `python.exe` in *Windows Security > Firewall & network protection > Allow an app through firewall*. Also check that the phone is on the same Wi-Fi as the PC, not mobile data or a guest network.
- **"This link is out of date."** The token was rotated. Scan the QR code again.
- **Talk says the mic needs HTTPS.** You opened the Wi-Fi or plain tailnet link. Use the **Tailscale HTTPS (Talk)** link (above).
- **The HTTPS link doesn't load.** Run `tailscale serve status` on the PC: it should show your `ts.net` name proxying to `http://127.0.0.1:8092`. The first visit can take a few seconds while Tailscale fetches the certificate.
- **Port 8092 is taken.** Set `phone_port` in `config.json` to a free port and turn the remote on again.

## Config keys

| Key | Default | Meaning |
|---|---|---|
| `phone_enabled` | `false` | Start the listener with the panel. |
| `phone_port` | `8092` | The listener's port. |
| `phone_token` | `""` | Made when you first turn it on. Never part of the panel's state. |
