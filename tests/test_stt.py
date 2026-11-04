import conftest_paths  # noqa: F401
import base64, json, os, struct, tempfile, unittest

import gguf
import stt
from test_gguf_metadata import _kv_str, _kv_u32, _s, _write_gguf


def _kv_bool(key, val):
    return _s(key) + struct.pack("<I", 7) + struct.pack("<?", val)


WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 24
MP3 = b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\x00" * 16
FLAC = b"fLaC" + b"\x00" * 16


def _multipart(fields, boundary="XyZ"):
    out = b""
    for name, (value, filename) in fields.items():
        disp = f'form-data; name="{name}"' + (f'; filename="{filename}"' if filename else "")
        out += f"--{boundary}\r\nContent-Disposition: {disp}\r\n\r\n".encode() + value + b"\r\n"
    return f"multipart/form-data; boundary={boundary}", out + f"--{boundary}--\r\n".encode()


class Modalities(unittest.TestCase):
    def _make(self, kvs):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "mmproj.gguf")
        _write_gguf(p, kvs)
        return p

    def test_audio_and_vision_encoders(self):
        p = self._make([_kv_str("general.architecture", "clip"),
                        _kv_bool("clip.has_vision_encoder", False),
                        _kv_bool("clip.has_audio_encoder", True)])
        self.assertEqual(gguf.projector_modalities(p), {"vision": False, "audio": True})

    def test_audio_projection_dim_alone_means_audio(self):
        p = self._make([_kv_str("general.architecture", "clip"),
                        _kv_u32("clip.audio.projection_dim", 3072)])
        self.assertEqual(gguf.projector_modalities(p), {"vision": False, "audio": True})

    def test_not_a_projector(self):
        p = self._make([_kv_str("general.architecture", "llama")])
        self.assertIsNone(gguf.projector_modalities(p))
        self.assertIsNone(gguf.projector_modalities(p + ".missing"))


class AudioModels(unittest.TestCase):
    def test_only_models_with_an_audio_projector(self):
        kinds = {"a.gguf": {"audio": True, "vision": False},
                 "v.gguf": {"audio": False, "vision": True}}
        sections = {"*": {"mmproj": "a.gguf"},
                    "voxtral": {"model": "x", "mmproj": "a.gguf"},
                    "gemma": {"model": "y", "mmproj": "v.gguf"},
                    "qwen": {"model": "z"},
                    "broken": {"model": "w", "mmproj": "gone.gguf"}}
        self.assertEqual(stt.audio_models(sections, kinds=kinds.get), ["voxtral"])

    def test_pick_model(self):
        self.assertEqual(stt.pick_model(["a", "b"], "b"), "b")
        self.assertEqual(stt.pick_model(["a", "b"], "whisper-1"), "a")
        self.assertEqual(stt.pick_model(["a", "b"], None), "a")
        with self.assertRaises(stt.SttError) as e:
            stt.pick_model([], "whisper-1")
        self.assertEqual(e.exception.status, 400)
        self.assertIn("audio", str(e.exception))


class Offer(unittest.TestCase):
    def test_voxtral_is_offered_with_its_projector(self):
        o = stt.OFFER
        self.assertEqual(o["repo"], "ggml-org/Voxtral-Mini-3B-2507-GGUF")
        self.assertTrue(o["path"].endswith(".gguf") and o["mmproj"].startswith("mmproj-"))
        self.assertEqual(o["bytes"], 2473001920 + 715714080)
        self.assertEqual(o["license"], "Apache-2.0")


class Formats(unittest.TestCase):
    def test_sniff(self):
        self.assertEqual(stt.sniff_format(WAV), "wav")
        self.assertEqual(stt.sniff_format(MP3), "mp3")
        self.assertEqual(stt.sniff_format(b"\xff\xfb\x90\x00" + b"\x00" * 8), "mp3")
        self.assertEqual(stt.sniff_format(FLAC), "flac")
        self.assertIsNone(stt.sniff_format(b"\x1aE\xdf\xa3webm"))
        self.assertIsNone(stt.sniff_format(b""))


class ParseRequest(unittest.TestCase):
    def test_sdk_upload(self):
        ctype, body = _multipart({"file": (WAV, "clip.wav"), "model": (b"whisper-1", None),
                                  "language": (b"de", None), "prompt": (b"LlamaForge, GGUF", None),
                                  "response_format": (b"text", None), "temperature": (b"0.2", None)})
        r = stt.parse_request(ctype, body)
        self.assertEqual(r, {"audio": WAV, "format": "wav", "model": "whisper-1", "language": "de",
                             "prompt": "LlamaForge, GGUF", "response_format": "text",
                             "temperature": 0.2})

    def test_defaults(self):
        ctype, body = _multipart({"file": (MP3, "a.mp3")})
        r = stt.parse_request(ctype, body)
        self.assertEqual((r["format"], r["model"], r["language"], r["prompt"], r["response_format"],
                          r["temperature"]), ("mp3", None, None, "", "json", 0.0))

    def test_refusals(self):
        def bad(fields, word):
            ctype, body = _multipart(fields)
            with self.assertRaises(stt.SttError) as e:
                stt.parse_request(ctype, body)
            self.assertEqual(e.exception.status, 400)
            self.assertIn(word, str(e.exception))
        bad({"model": (b"m", None)}, "file")
        bad({"file": (b"", "a.wav")}, "empty")
        bad({"file": (b"\x1aE\xdf\xa3....", "a.webm")}, "WAV, MP3 or FLAC")
        bad({"file": (WAV, "a.wav"), "response_format": (b"srt", None)}, "json or text")
        bad({"file": (WAV, "a.wav"), "temperature": (b"hot", None)}, "temperature")
        with self.assertRaises(stt.SttError):
            stt.parse_request("application/json", b"{}")

    def test_size_limit(self):
        ctype, body = _multipart({"file": (WAV + b"\x00" * 64, "a.wav")})
        with self.assertRaises(stt.SttError) as e:
            stt.parse_request(ctype, body, max_bytes=32)
        self.assertEqual(e.exception.status, 413)


class ChatBody(unittest.TestCase):
    # Shapes found against Voxtral Mini on a real router (2026-11-04): audio
    # first, then "Repeat exactly what was said". A system prompt, the word
    # "transcribe", or a language hint made it refuse or stop after a sentence.
    def test_audio_first_then_the_ask(self):
        b = stt.chat_body("voxtral", WAV, "wav", language="de", prompt="GGUF", temperature=0.0)
        self.assertEqual((b["model"], b["temperature"], b["stream"]), ("voxtral", 0.0, False))
        self.assertEqual(len(b["messages"]), 1)
        audio, ask = b["messages"][0]["content"]
        self.assertEqual(audio["input_audio"], {"data": base64.b64encode(WAV).decode(), "format": "wav"})
        self.assertIn("Repeat exactly what was said", ask["text"])
        self.assertNotIn("ranscribe", ask["text"])
        self.assertIn("GGUF", ask["text"])
        self.assertNotIn("'de'", ask["text"])

    def test_no_prompt(self):
        b = stt.chat_body("m", MP3, "mp3")
        self.assertNotIn("Spell", b["messages"][0]["content"][1]["text"])


class Transcript(unittest.TestCase):
    def test_text_from_reply(self):
        r = {"choices": [{"message": {"content": "  <think>hm</think>\nHello there.  "}}]}
        self.assertEqual(stt.transcript(r), "Hello there.")

    def test_lead_in_and_quotes_are_dropped(self):
        said = "The quick brown fox jumps over the lazy dog. LlamaForge runs speech-to-text on your own GPU."
        got = "Here's the text you provided, repeated exactly as you said it:\n\n\"" + said + '"'
        self.assertEqual(stt.transcript({"choices": [{"message": {"content": got}}]}), said)
        self.assertEqual(stt.transcript({"choices": [{"message": {"content": '"Llama Forge runs."'}}]}),
                         "Llama Forge runs.")
        # a colon inside the speech itself stays
        self.assertEqual(stt.transcript({"choices": [{"message": {"content": "Note: buy milk."}}]}),
                         "Note: buy milk.")

    def test_bad_reply(self):
        with self.assertRaises(stt.SttError) as e:
            stt.transcript({"error": {"message": "image input is not supported"}})
        self.assertEqual(e.exception.status, 502)
        self.assertIn("not supported", str(e.exception))

    def test_respond(self):
        self.assertEqual(stt.respond("hi", "json"), (json.dumps({"text": "hi"}).encode(), "application/json"))
        self.assertEqual(stt.respond("hi", "text"), (b"hi", "text/plain; charset=utf-8"))


if __name__ == "__main__":
    unittest.main()
