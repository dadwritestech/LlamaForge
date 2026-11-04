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
                       "for example Voxtral Mini (Voice tab)")
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


def chat_body(model, audio, fmt, language=None, prompt="", temperature=0.0):
    """The router chat request that transcribes `audio`."""
    ask = "Transcribe this audio word for word. Reply with the transcript only, no comments."
    if language:
        ask += f" The speech is in language '{language}'; write it in that language."
    if prompt:
        ask += f" Context, for spelling names and terms: {prompt}"
    return {"model": model, "temperature": temperature, "stream": False,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": ask},
                {"type": "input_audio",
                 "input_audio": {"data": base64.b64encode(audio).decode(), "format": fmt}}]}]}


_THINK = re.compile(r"<think>.*?</think>", re.S)


def transcript(reply):
    """The words from the router's chat reply. Raises SttError (502) with the
    router's own message when there are none."""
    try:
        content = reply["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        err = (reply or {}).get("error") if isinstance(reply, dict) else None
        msg = err.get("message") if isinstance(err, dict) else err
        raise SttError(f"the router gave no transcript: {msg or 'empty reply'}", 502)
    return _THINK.sub("", content or "").strip()


def respond(text, response_format):
    """(body bytes, content type) the way OpenAI's SDK reads them."""
    if response_format == "text":
        return text.encode("utf-8"), "text/plain; charset=utf-8"
    return json.dumps({"text": text}).encode(), "application/json"
