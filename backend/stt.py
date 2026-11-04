"""Speech to text through llama.cpp's own audio input.

llama-server takes audio in a chat completion (`{"type": "input_audio",
"input_audio": {"data": <base64>, "format": "wav"}}`) for any model whose
mmproj carries an audio encoder: Voxtral, Qwen2.5-Omni, Ultravox and others.
So no third engine is needed: this module turns an OpenAI-style
POST /v1/audio/transcriptions into one chat request to such a model on the
router and hands the words back the way OpenAI's SDK expects them.

llama.cpp decodes WAV, MP3 and FLAC itself, so audio goes through untouched;
the browser's mic records PCM and wraps it as WAV. A chat model gives no
timestamps, so srt / vtt / verbose_json are refused rather than faked.

Pure Python stdlib; this file does no I/O beyond reading mmproj headers.
"""
import base64, json, re

import batchapi
import gguf

MAX_BYTES = 25 * 1024 * 1024        # OpenAI's limit for one transcription upload
FORMATS = ("json", "text")

# The one-click model on the Voice tab: Google's Gemma 4 E4B with its audio
# mmproj, as unsloth packs it. It goes through the Discover download, which
# registers it in models.ini with the mmproj beside it. One model hears and
# answers, so Talk never swaps. (Voxtral Mini, offered first, answered spoken
# questions instead of writing them down: ggml-org's GGUF carries a Devstral
# chat template. It still works for statements and dictation.)
OFFER = {"label": "Gemma 4 E4B", "repo": "unsloth/gemma-4-E4B-it-GGUF",
         "path": "gemma-4-E4B-it-Q4_K_M.gguf", "mmproj": "mmproj-F16.gguf",
         "bytes": 4977171584 + 990372672, "license": "Apache-2.0",
         "note": "Hears and answers on one model; about 4.7 GB of VRAM at an 8k context."}


class SttError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def audio_models(sections, kinds=gguf.projector_modalities):
    """models.ini section names whose mmproj has an audio encoder, sorted."""
    out = []
    for name, keys in (sections or {}).items():
        if name == "*" or not keys.get("mmproj"):
            continue
        k = kinds(keys["mmproj"])
        if k and k.get("audio"):
            out.append(name)
    return sorted(out)


def pick_model(models, want=None):
    """The model asked for if it can hear; otherwise (OpenAI's `whisper-1`,
    or nothing) the first one that can."""
    if not models:
        raise SttError("no model here can take audio: add one with an audio mmproj, "
                       "for example Gemma 4 E4B (Voice tab)")
    return want if want in models else models[0]


def sniff_format(data):
    """'wav', 'mp3' or 'flac' from the first bytes, or None."""
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if data[:4] == b"fLaC":
        return "flac"
    if data[:3] == b"ID3" or (len(data) > 1 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0):
        return "mp3"
    return None


def parse_request(ctype, raw, max_bytes=MAX_BYTES):
    """An SDK's multipart upload -> {audio, format, model, language, prompt,
    response_format, temperature}. Raises SttError."""
    if not (ctype or "").lower().startswith("multipart/form-data"):
        raise SttError("send the audio as multipart/form-data with a 'file' field")
    try:
        fields = batchapi.multipart_fields(ctype, raw)
    except ValueError as e:
        raise SttError(str(e))

    def text(name):
        v = fields.get(name, (b"", None))[0].decode("utf-8", "replace").strip()
        return v or None

    if "file" not in fields:
        raise SttError("the upload has no 'file' field")
    audio = fields["file"][0]
    if not audio:
        raise SttError("the audio file is empty")
    if len(audio) > max_bytes:
        raise SttError(f"the audio is over {max_bytes // (1024 * 1024)} MB", 413)
    fmt = sniff_format(audio)
    if not fmt:
        raise SttError("send WAV, MP3 or FLAC audio; llama.cpp decodes only those")
    rf = text("response_format") or "json"
    if rf not in FORMATS:
        raise SttError("response_format must be json or text: a chat model gives no timestamps")
    try:
        temp = float(text("temperature") or 0)
    except ValueError:
        raise SttError("temperature must be a number")
    return {"audio": audio, "format": fmt, "model": text("model"), "language": text("language"),
            "prompt": text("prompt") or "", "response_format": rf, "temperature": temp}


ASK = "Repeat exactly what was said, word for word, with nothing else."


def _budget(audio, fmt):
    """Tokens enough for the words in a WAV (~3 a second, generously), so a
    model that runs on (Gemma 4 once read our ask back for 13 s) stops early."""
    if fmt != "wav" or len(audio) < 44:
        return 1024
    rate = int.from_bytes(audio[28:32], "little")            # byte rate
    secs = (len(audio) - 44) // rate if rate else 0
    return 64 + 8 * min(secs, 600)


def chat_body(model, audio, fmt, language=None, prompt="", temperature=0.0):
    """The router chat request that transcribes `audio`.

    The shape was found against Voxtral Mini on a real router: the audio part
    first, then "Repeat exactly what was said". A system prompt or the word
    "transcribe" made it answer "I can't transcribe audio", and a language hint
    made it stop after the first sentence, so `language` is accepted (OpenAI's
    SDK sends it) but left to the model, which detects the language itself."""
    ask = ASK
    if prompt:
        ask += f" Spell names and terms as in: {prompt}"
    return {"model": model, "temperature": temperature, "stream": False,
            "max_tokens": _budget(audio, fmt),
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [
                {"type": "input_audio",
                 "input_audio": {"data": base64.b64encode(audio).decode(), "format": fmt}},
                {"type": "text", "text": ask}]}]}


_THINK = re.compile(r"<think>.*?</think>", re.S)
# "Here's the text you provided, repeated exactly as you said it:" then a blank line
_LEAD_IN = re.compile(r"\A[^\n]*:\s*\n\s*\n")


def transcript(reply):
    """The words from the router's chat reply. Raises SttError (502) with the
    router's own message when there are none. A chatty lead-in line and the
    quotes some models wrap the words in are dropped."""
    try:
        content = reply["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        err = (reply or {}).get("error") if isinstance(reply, dict) else None
        msg = err.get("message") if isinstance(err, dict) else err
        raise SttError(f"the router gave no transcript: {msg or 'empty reply'}", 502)
    text = _LEAD_IN.sub("", _THINK.sub("", content or "").strip()).strip()
    cut = text.find(ASK[:24])                     # a model reading our own ask back
    if cut > 0:
        text = text[:cut].strip()
    if len(text) > 1 and text[0] == text[-1] == '"':
        text = text[1:-1].strip()
    return text


# Talk mode: what was heard goes to the same model as a plain chat turn (one
# model hears and answers, so models-max 1 never swaps), and the reply is
# cleaned for the speech engine to read out.
TALK_TURNS = 6                  # earlier exchanges kept as context
TALK_SYSTEM = ("You are talking with the user out loud. Your reply will be spoken by a speech "
               "engine, so answer in one to three short sentences of plain prose: no lists, "
               "no markdown, no code.")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MD_MARK = re.compile(r"(?m)^\s*(?:[-*+]|\d+[.)]|#+)\s+|[*_`#>]+")


def talk_body(model, history, heard):
    """A chat turn answering `heard`, with the last TALK_TURNS exchanges of a
    client-kept history (anything but plain user/assistant text is dropped)."""
    keep = [{"role": m["role"], "content": m["content"].strip()} for m in (history or [])
            if isinstance(m, dict) and m.get("role") in ("user", "assistant")
            and isinstance(m.get("content"), str) and m["content"].strip()]
    msgs = [{"role": "system", "content": TALK_SYSTEM}] + keep[-2 * TALK_TURNS:]
    return {"model": model, "stream": False, "max_tokens": 300,
            # a thinking model (Gemma 4, Qwen3) otherwise spends the budget on thoughts
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": msgs + [{"role": "user", "content": heard}]}


def spoken(reply):
    """The reply's text, fit to be read out: thoughts and markdown dropped."""
    try:
        content = reply["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        err = (reply or {}).get("error") if isinstance(reply, dict) else None
        msg = err.get("message") if isinstance(err, dict) else err
        raise SttError(f"the router gave no answer: {msg or 'empty reply'}", 502)
    text = _MD_MARK.sub("", _MD_LINK.sub(r"\1", _THINK.sub("", content or "")))
    text = " ".join(text.split())
    if not text:
        raise SttError("the model answered with nothing to say", 502)
    return text


def respond(text, response_format):
    """(body bytes, content type) the way OpenAI's SDK reads them."""
    if response_format == "text":
        return text.encode("utf-8"), "text/plain; charset=utf-8"
    return json.dumps({"text": text}).encode(), "application/json"
