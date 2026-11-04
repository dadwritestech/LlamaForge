---
title: Voice (Text to Speech)
section: guides
order: 12
---

# Voice

The **Voice** tab turns text into speech on your own GPU, optionally in a voice you recorded. LlamaForge does not do the speaking: llama.cpp ships a text-to-speech tool, `llama-tts`, and LlamaForge runs it with one of the models it supports, then gives you a UI and an OpenAI-style endpoint for it.

## Models

| Model | Size | Languages | Licence | Good for |
|---|---|---|---|---|
| [Qwen3-TTS 1.7B](https://huggingface.co/ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF) | 2.1 GiB (Q8_0) | 10 | Apache-2.0 | Quality, any language, voice cloning. Best on a GPU. |
| [Pocket TTS](https://huggingface.co/EryriLabs/pocket-tts-GGUF) (English) | 209 MiB | English | CC-BY-4.0 | Speed: faster than real time even on a CPU. Always needs a voice clip. |

Install either or both from the tab. Getting Pocket TTS also saves a starter voice, a clip from Kyutai's [donated voices](https://huggingface.co/kyutai/tts-voices) (CC0), so it can speak right away.

> [!NOTE]
> Most TTS GGUFs on Hugging Face are made for other runtimes and do not load in `llama-tts`: for example the Qwen3-TTS and Pocket TTS repos packaged for qwentts.cpp, pockettts.cpp or CrispASR, and models such as Kokoro, Chatterbox, Dia, Orpheus or F5-TTS. `llama-tts` needs a backbone plus an `mmproj` audio projector of the `qwen3tts` or `pockettts` architecture.

## Set up

Two things are needed, and the tab tells you which one is missing:

1. **`llama-tts`**, next to your `llama-server`. Official llama.cpp builds include it. A source build from **Build / Update** builds it as well; an older source build may only have the server, so rebuild once.
2. **A model.** Each **Get** button downloads a backbone and its audio projector into `<root>/tts/models/` (or `tts_dir` in [config.json](config.md)); extra models go in a subfolder each. Speech models are kept apart from your chat models and never appear in the Models list. To add a compatible model by hand, put the `.gguf` and its `mmproj-*.gguf` in a new subfolder; the tab reads the architecture from the file.

## Speak

Type up to 2000 characters, pick a model (when you have more than one), a voice and, for Qwen3-TTS, a language (en, zh, de, it, pt, es, ja, ko, fr, ru), and press **Speak**. You get a waveform, a player and a WAV download.

Each request starts `llama-tts`, loads the model, speaks and exits, so no VRAM is held between requests. One request runs at a time. On an RTX 5080, a sentence takes about 3 to 4 seconds with Qwen3-TTS including the model load, a little faster than the audio plays back. Pocket TTS takes about a second for the same sentence, and about 1.5 seconds on the CPU alone.

## Voices

A voice is a short reference clip of someone speaking. The model speaks in the voice of the clip. **Record a clip** uses your microphone (up to 30 seconds), **Upload a clip** takes any audio file your browser can play. Either way, the browser converts it to 24 kHz mono WAV, trims the silence at both ends and asks for a name. Clips are stored as `<root>/tts/voices/<name>.wav`; 5 to 20 seconds of clear speech in a quiet room works best.

> [!IMPORTANT]
> Only clone voices you have permission to use: your own, or someone who agreed to it.

Without a voice, Qwen3-TTS picks its own; Pocket TTS uses your first saved voice, and refuses to speak if you have none. Set `tts_default_voice` in [config.json](config.md) to use one of your voices whenever a request doesn't name one.

## Use it from code

The panel serves `POST /v1/audio/speech`, shaped like OpenAI's speech API:

```bash
curl http://127.0.0.1:8090/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"Hello from my own GPU.","voice":"alloy"}' \
  -o speech.wav
```

- `input` (required): the text, up to 2000 characters.
- `voice`: the name of one of your voices. Names you haven't saved, like OpenAI's `alloy`, fall back to the default voice, so OpenAI clients work unchanged.
- `model`: which speech model to use, by the name the tab shows (the file name without `.gguf`, for example `pocket-tts-en`). Anything else, like OpenAI's `tts-1`, uses the first model.
- `language`: one of the codes above, default `en`. Ignored by single-language models.
- `response_format`: `pcm` returns raw 16-bit 24 kHz mono samples; anything else returns WAV. LlamaForge's backend has no audio codecs, so `mp3` or `opus` requests get WAV.
- `speed` is accepted and ignored.

Errors use the OpenAI shape (`{"error": {"message", "type"}}`): 400 for a bad request, 503 when speech isn't set up, 500 with the tail of the `llama-tts` log when it fails. When the router is reachable from your network (`router_host` is not `127.0.0.1`), the endpoint asks for the router's API key, as `Authorization: Bearer <key>` or `x-api-key`, like the other endpoints on the panel.

## Speech to text

The other direction needs no extra tool. llama.cpp's server takes audio in a chat request for models whose projector (`mmproj`) has an audio encoder, such as Voxtral, Gemma 4, Qwen2.5-Omni and Ultravox.

**On the Voice tab**, the **Speech to text** card has a **Record** button: talk, press **Stop**, and the words come back from your own GPU. **Speak it back** drops them into the Speak box. When nothing you have can hear yet, the card offers **Gemma 4 E4B** (unsloth's Q4_K_M GGUF plus its F16 audio mmproj, about 5.6 GiB, Apache-2.0) and adds it to your models with its mmproj in one click.

**Which model to hear with.** Earlier releases offered Voxtral Mini 3B. ggml-org's Voxtral GGUF carries a Devstral chat template, and in testing it *answered* spoken questions instead of writing them down ("What is the capital of France?" came back as a fact about Paris), whatever the prompt said. Statements and dictation still come through word for word, so it stays fine for notes. Gemma 4 models write questions down as questions. On an RTX 5080 with clips spoken by Qwen3-TTS:

| Model | Hearing a clip | Mistakes seen | VRAM at 8k context |
|---|---|---|---|
| Gemma 4 E4B Q4_K_M | about 0.2 s | heard "climb it" as "climate"; dropped the second of two sentences once | about 4.7 GB |
| Gemma 4 12B Q6_K_XL | 1.7 to 2.4 s | none on these clips | more (a 12B model) |
| Voxtral Mini 3B Q4_K_M | about 0.45 s | answers questions instead of writing them down | about 4.3 GB |

For the most accurate hearing pick a bigger Gemma 4 with its audio mmproj; for speed, E4B.

### Talk

The **Talk** card on the Voice tab is a spoken conversation. Press **Talk**, ask something, press **Stop**: the model that hears you writes down what you said, answers it in a sentence or three, and the answer is read out by the Speak engine. The same model hears and answers, so nothing is swapped in between (the router keeps one model loaded). Each turn shows what was heard, the answer, and how long hearing, thinking and speaking took.

Tick **Hands-free** and the mic opens again after each answer and stops by itself when you pause (about 1.2 s of quiet after you have spoken), so you can just keep talking. **New conversation** forgets the earlier turns; up to six earlier exchanges go along as context.

On an RTX 5080, Gemma 4 E4B heard each question in about 0.2 s and answered in 0.2 to 0.45 s, and kept the thread: after "What is the tallest mountain?", "how long does it take to climb it" was about Everest when the 12B heard it. Thinking is switched off for these turns (Gemma 4 and Qwen3 otherwise spend the reply's budget thinking), and markdown is stripped so the speech engine doesn't read out asterisks. Without Speak set up, the answers show as text.

The panel route behind it is `POST /api/stt/talk` with `{"audio": <base64 WAV>, "history": [...], "model": ...}`, returning `{heard, reply, model, hear_s, think_s}`.

From code, the panel answers OpenAI's transcription endpoint:

```bash
curl http://127.0.0.1:8090/v1/audio/transcriptions   -F file=@meeting.wav -F model=whisper-1
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8090/v1", api_key="none")   # or your router key
print(client.audio.transcriptions.create(model="whisper-1", file=open("meeting.wav", "rb")).text)
```

- `file` (required): WAV, MP3 or FLAC, up to 25 MB. llama.cpp decodes these itself; other formats (WebM, M4A) are a 400, so convert them first.
- `model`: the name of a model that can hear. Anything else, like OpenAI's `whisper-1`, uses the first one. With none, the error says so.
- `prompt`: names and terms to spell right, passed to the model as a hint.
- `language`: accepted, but left to the model, which detects the language itself. (Told the language, Voxtral Mini stopped after the first sentence.)
- `response_format`: `json` (the default, `{"text": ...}`) or `text`. `srt`, `vtt` and `verbose_json` are a 400: a chat model gives no timestamps, and LlamaForge won't invent them.
- `temperature`: default 0.

The model is a chat model listening, not a dedicated speech recognizer: it is asked to repeat exactly what was said, and usually does. A chatty lead-in ("Here's the text...") and wrapping quotes are trimmed off. It loads on the router like any other request, so the first call waits for the load. The key rules are the same as for `/v1/audio/speech`.

