---
title: Phone Remote
section: guides
order: 15
---

# Phone Remote

Chat with the models on your PC from your phone. Click **Phone** in the top bar of the panel, turn it on, and scan the QR code with your phone's camera. The phone opens a small chat page served by LlamaForge: pick a model, ask, and the answer streams back from your own GPUs with its speed in tokens per second.

Nothing to install on the phone, no account, no cloud relay.

## What the phone can reach

The panel itself stays on `127.0.0.1` only: it can build llama.cpp and install packages, which no phone link should reach. The phone remote is a separate listener on port `8092` of every network address (`0.0.0.0`), and it serves exactly three things:

| Route | What it does |
|---|---|
| `GET /` | The chat page. It holds no data until it has a token. |
| `GET /api/status` | Model names (and which are loaded) plus GPU memory and load. |
| `POST /api/chat` | A plain chat with one model, streamed. Only the model name and text messages are passed on; tools, sampler settings and anything else are dropped. |

Everything else answers `404`. The router's API key is added on the PC, so the phone never sees it.

> [!WARNING]
> Anyone who has the link can chat with your models (and load one by picking it). Use it on networks you trust: your home Wi-Fi, or [Tailscale](https://tailscale.com/) when you're away. Don't forward port 8092 on your router to the internet.

## Pairing and unpairing

The QR code holds a link like `http://192.168.1.20:8092/#k=<token>`. The token sits after the `#`, which browsers never send over the network. The page moves it into the phone's storage and removes it from the address bar. Each request then sends it as a `Bearer` token, compared in constant time.

- **New link** makes a new token. Every phone with the old link stops working until it scans the new QR code.
- **Turn off** closes the port. Turning it on again keeps the same token, so paired phones keep working.

## Away from home with Tailscale

If this PC is on a [Tailscale](https://tailscale.com/) network, the dialog shows a second QR code for its tailnet address (`100.x.y.z`). Install Tailscale on the phone, sign in to the same tailnet, and the link works from anywhere, encrypted, with no ports opened on your router.

## Troubleshooting

- **The phone can't open the page.** Windows may ask whether Python can use private networks; allow it. If you said no, allow `python.exe` in *Windows Security > Firewall & network protection > Allow an app through firewall*. Also check that the phone is on the same Wi-Fi as the PC, not mobile data or a guest network.
- **"This link is out of date."** The token was rotated. Scan the QR code again.
- **Port 8092 is taken.** Set `phone_port` in `config.json` to a free port and turn the remote on again.

## Config keys

| Key | Default | Meaning |
|---|---|---|
| `phone_enabled` | `false` | Start the listener with the panel. |
| `phone_port` | `8092` | The listener's port. |
| `phone_token` | `""` | Made when you first turn it on. Never part of the panel's state. |
