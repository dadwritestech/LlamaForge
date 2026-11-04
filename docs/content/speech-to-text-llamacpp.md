---
title: Speech to text with llama.cpp
section: guides
order: 18
---

# Speech to text with llama.cpp (no Whisper)

llama.cpp's `llama-server` can turn speech into text by itself. Any model whose projector (`mmproj`) has an audio encoder, such as Gemma 4, Voxtral, Qwen2.5-Omni and Ultravox, takes audio in an ordinary chat request. You don't need whisper.cpp, Python or a second server. This page shows the request that works, the four things that go wrong, and which model to pick. Everything here was run on an RTX 5080 with clips spoken by Qwen3-TTS (November 2026).

## The request that works

Start the server with the model and its audio mmproj:

```bash
llama-server -m gemma-4-E4B-it-Q4_K_M.gguf --mmproj mmproj-F16.gguf -ngl 99 -c 8192 --port 8080
```

(Both files are in [unsloth/gemma-4-E4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF), Apache-2.0.)

Then send the audio as an `input_audio` part, followed by the instruction:

```bash
B64=$(base64 -w0 question.wav)
curl -s http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": [
    {"type": "input_audio", "input_audio": {"data": "'"$B64"'", "format": "wav"}},
    {"type": "text", "text": "Repeat exactly what was said, word for word, with nothing else."}]}],
  "temperature": 0,
  "max_tokens": 200,
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

A spoken "What is the capital city of France?" came back as exactly that, in 0.40 s for the whole curl call. llama.cpp decodes **WAV, MP3 and FLAC** itself; anything else (WebM from a browser's MediaRecorder, M4A) has to be converted first.

## What goes wrong

**1. Without the instruction, the model answers you.** These are chat models listening, not speech recognizers. The same request without the text part came back as `Paris`. Always say what you want done with the audio.

**2. Some models answer questions whatever you say.** ggml-org's Voxtral Mini 3B GGUF carries a Devstral chat template (its default system prompt starts "You are Devstral"). In our tests it wrote statements down word for word but *answered* every spoken question, with every phrasing we tried, including its raw `[TRANSCRIBE]` mode. It also refused outright when the instruction said "transcribe" or came with a system prompt, and a language hint made it stop after the first sentence. That's fine for dictation, but a bad fit for a voice assistant.

**3. Thinking models spend the budget thinking.** With `enable_thinking` left on, Gemma 4 E4B returned an empty answer (`finish_reason: length`) after 60 tokens of reasoning. Pass `"chat_template_kwargs": {"enable_thinking": false}`; templates without that switch ignore it.

**4. A model can run on.** Once Gemma 4 12B read the question and then our own instruction back, for 13 seconds. Cap `max_tokens` by the length of the clip: people say about 3 words a second, so `64 + 8 × seconds` is generous. Cutting the reply where your instruction's text starts catches the rest.

## Which model

| Model (GGUF) | Hearing one clip | Mistakes seen | VRAM at 8k context |
|---|---|---|---|
| Gemma 4 E4B Q4_K_M + mmproj-F16 | about 0.2 s | heard "climb it" as "climate"; dropped the second of two sentences once | about 4.7 GB |
| Gemma 4 12B Q6_K_XL + mmproj-F16 | 1.7 to 2.4 s | none on our clips | more (a 12B model) |
| Voxtral Mini 3B Q4_K_M + Q8_0 mmproj | about 0.45 s | answers spoken questions instead of writing them down | about 4.3 GB |

Pick E4B for speed, a bigger Gemma 4 for accuracy. Because the same model can also chat, one loaded model can both hear a question and answer it. With llama.cpp's router (`--models-max 1`) that means no model swap between hearing and answering.

## The same thing, without the curl

[LlamaForge](https://github.com/dadwritestech/LlamaForge) is a browser control panel over llama.cpp's own router, and it does all of the above for you (MIT, runs on your machine):

- **Speech to text** on the Voice tab: press Record, talk, get the words back. If nothing you have can hear yet, one click downloads Gemma 4 E4B with its mmproj and registers it.
- **Talk**: a spoken conversation. The model hears you, answers, and Qwen3-TTS reads the answer out. Hands-free mode opens the mic again after each answer and stops it when you pause.
- **`/v1/audio/transcriptions`** on the panel, so OpenAI's SDK works unchanged (`model="whisper-1"` uses the first model that can hear).

See [Voice](voice.md#speech-to-text) for the details.
