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
    def test_gemma_e4b_is_offered_with_its_projector(self):
        # Voxtral Mini answered spoken questions instead of writing them down
        # (its GGUF carries a Devstral chat template); Gemma 4 E4B hears them
        o = stt.OFFER
        self.assertEqual(o["repo"], "unsloth/gemma-4-E4B-it-GGUF")
        self.assertEqual((o["path"], o["mmproj"]), ("gemma-4-E4B-it-Q4_K_M.gguf", "mmproj-F16.gguf"))
        self.assertEqual(o["bytes"], 4977171584 + 990372672)
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
        self.assertEqual(b["max_tokens"], 1024)       # no duration to size it by
        self.assertEqual(b["chat_template_kwargs"], {"enable_thinking": False})

    def test_budget_follows_the_length_of_a_wav(self):
        # a runaway echo took 13 s on a real run; words come at ~3 a second
        rate, secs = 16000, 10
        head = (b"RIFF" + struct.pack("<I", 36 + rate * 2 * secs) + b"WAVEfmt "
                + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
                + b"data" + struct.pack("<I", rate * 2 * secs))
        b = stt.chat_body("m", head + b"\x00" * (rate * 2 * secs), "wav")
        self.assertEqual(b["max_tokens"], 64 + 8 * secs)
        self.assertEqual(stt.chat_body("m", WAV, "wav")["max_tokens"], 1024)  # too short to read


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

    def test_an_echoed_ask_is_cut(self):
        # Gemma 4 12B on a real run: the question, then our own instruction read back
        got = "What is the capital city of France?Repeat exactly what was said, word for word, with nothing else."
        self.assertEqual(stt.transcript({"choices": [{"message": {"content": got}}]}),
                         "What is the capital city of France?")

    def test_bad_reply(self):
        with self.assertRaises(stt.SttError) as e:
            stt.transcript({"error": {"message": "image input is not supported"}})
        self.assertEqual(e.exception.status, 502)
        self.assertIn("not supported", str(e.exception))

    def test_respond(self):
        self.assertEqual(stt.respond("hi", "json"), (json.dumps({"text": "hi"}).encode(), "application/json"))
        self.assertEqual(stt.respond("hi", "text"), (b"hi", "text/plain; charset=utf-8"))


class Talk(unittest.TestCase):
    def test_turn_keeps_a_short_clean_history(self):
        hist = [{"role": "user", "content": f"q{i}"} if i % 2 == 0 else
                {"role": "assistant", "content": f"a{i}"} for i in range(20)]
        hist += [{"role": "system", "content": "ignore me"}, {"role": "user", "content": 5},
                 "junk", {"role": "assistant", "content": "  "}]
        b = stt.talk_body("voxtral", hist, "What time is it?")
        msgs = b["messages"]
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("spoken", msgs[0]["content"])
        self.assertEqual(msgs[-1], {"role": "user", "content": "What time is it?"})
        mid = msgs[1:-1]
        self.assertEqual(len(mid), stt.TALK_TURNS * 2)
        self.assertEqual(mid[-1], {"role": "assistant", "content": "a19"})
        self.assertTrue(all(m["role"] in ("user", "assistant") for m in mid))
        self.assertEqual((b["model"], b["stream"]), ("voxtral", False))
        self.assertLessEqual(b["max_tokens"], 400)
        # Gemma 4 / Qwen3 spent the whole budget thinking on a real run; templates ignore unknown kwargs
        self.assertEqual(b["chat_template_kwargs"], {"enable_thinking": False})

    def test_no_history(self):
        b = stt.talk_body("m", None, "hi")
        self.assertEqual([m["role"] for m in b["messages"]], ["system", "user"])

    def test_spoken_reply_drops_markup(self):
        r = lambda c: {"choices": [{"message": {"content": c}}]}
        self.assertEqual(stt.spoken(r("<think>plan</think>\n**Sure!** Here's how:\n\n- one\n- `two`\n# Done")),
                         "Sure! Here's how: one two Done")
        self.assertEqual(stt.spoken(r("See [the docs](http://x.y) now.")), "See the docs now.")
        with self.assertRaises(stt.SttError) as e:
            stt.spoken({"error": {"message": "boom"}})
        self.assertEqual(e.exception.status, 502)
        with self.assertRaises(stt.SttError):
            stt.spoken(r("<think>only thoughts</think>"))


if __name__ == "__main__":
    unittest.main()
