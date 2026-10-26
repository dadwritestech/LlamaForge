import conftest_paths  # noqa: F401
import json, unittest
import anthropic_shim as sh


class TestRequestTranslation(unittest.TestCase):
    def test_system_string_becomes_system_message(self):
        o = sh.to_openai_request({"model": "m", "system": "be brief",
                                  "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(o["messages"][0], {"role": "system", "content": "be brief"})
        self.assertEqual(o["messages"][1], {"role": "user", "content": "hi"})

    def test_system_blocks_join_text(self):
        o = sh.to_openai_request({"model": "m",
            "system": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
            "messages": []})
        self.assertEqual(o["messages"][0], {"role": "system", "content": "ab"})

    def test_text_blocks_collapse_to_string(self):
        o = sh.to_openai_request({"model": "m",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]})
        self.assertEqual(o["messages"][0], {"role": "user", "content": "hello"})

    def test_image_block_becomes_image_url(self):
        o = sh.to_openai_request({"model": "m", "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "XYZ"}}]}]})
        part = o["messages"][0]["content"][0]
        self.assertEqual(part["type"], "image_url")
        self.assertEqual(part["image_url"]["url"], "data:image/png;base64,XYZ")

    def test_assistant_tool_use_becomes_tool_calls(self):
        o = sh.to_openai_request({"model": "m", "messages": [{"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "NYC"}}]}]})
        msg = o["messages"][0]
        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(msg["tool_calls"][0]["id"], "t1")
        self.assertEqual(msg["tool_calls"][0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(msg["tool_calls"][0]["function"]["arguments"]), {"city": "NYC"})

    def test_tool_result_becomes_tool_message(self):
        o = sh.to_openai_request({"model": "m", "messages": [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "sunny"}]}]})
        self.assertEqual(o["messages"][0], {"role": "tool", "tool_call_id": "t1", "content": "sunny"})

    def test_tools_and_tool_choice(self):
        o = sh.to_openai_request({"model": "m", "messages": [],
            "tools": [{"name": "f", "description": "d", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "tool", "name": "f"}})
        self.assertEqual(o["tools"][0], {"type": "function",
            "function": {"name": "f", "description": "d", "parameters": {"type": "object"}}})
        self.assertEqual(o["tool_choice"], {"type": "function", "function": {"name": "f"}})

    def test_sampling_and_stop_mapping(self):
        o = sh.to_openai_request({"model": "m", "messages": [], "max_tokens": 64,
            "temperature": 0.3, "top_p": 0.9, "top_k": 40, "stop_sequences": ["\n\n"]})
        self.assertEqual(o["max_tokens"], 64)
        self.assertEqual(o["temperature"], 0.3)
        self.assertEqual(o["top_p"], 0.9)
        self.assertEqual(o["stop"], ["\n\n"])
        self.assertNotIn("top_k", o)

    def test_tool_choice_auto_and_any(self):
        self.assertEqual(sh.map_tool_choice({"type": "auto"}), "auto")
        self.assertEqual(sh.map_tool_choice({"type": "any"}), "required")


def reminder(text):
    return f"<system-reminder>\n{text}\n</system-reminder>"


class TestMidConversationSystem(unittest.TestCase):
    """Claude Code 2.1 sends role=system messages mid-conversation (hook context).
    Strict chat templates (Qwen) raise unless every system message comes first."""

    def roles(self, o):
        return [m["role"] for m in o["messages"]]

    def test_system_after_user_folds_into_that_user_turn(self):
        o = sh.to_openai_request({"model": "m", "system": "top", "messages": [
            {"role": "user", "content": "hey"},
            {"role": "system", "content": [{"type": "text", "text": "hook ctx"},
                                           {"type": "tool_addition"}]}]})
        self.assertEqual(self.roles(o), ["system", "user"])
        self.assertEqual(o["messages"][1]["content"], "hey\n\n" + reminder("hook ctx"))

    def test_system_after_tool_result_becomes_user_turn(self):
        o = sh.to_openai_request({"model": "m", "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "x", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
            {"role": "system", "content": "note"}]})
        self.assertEqual(self.roles(o), ["user", "assistant", "tool", "user"])
        self.assertEqual(o["messages"][3]["content"], reminder("note"))

    def test_system_before_any_turn_joins_the_system_prompt(self):
        o = sh.to_openai_request({"model": "m", "system": "top", "messages": [
            {"role": "system", "content": "more"}, {"role": "user", "content": "hi"}]})
        self.assertEqual(o["messages"], [{"role": "system", "content": "top\n\nmore"},
                                         {"role": "user", "content": "hi"}])

    def test_folds_into_multipart_user_turn(self):
        o = sh.to_openai_request({"model": "m", "messages": [
            {"role": "user", "content": [{"type": "text", "text": "see"}, {"type": "image",
             "source": {"type": "base64", "media_type": "image/png", "data": "AA"}}]},
            {"role": "system", "content": "note"}]})
        self.assertEqual(self.roles(o), ["user"])
        self.assertEqual(o["messages"][0]["content"][-1],
                         {"type": "text", "text": "\n\n" + reminder("note")})

    def test_empty_system_message_is_dropped(self):
        o = sh.to_openai_request({"model": "m", "messages": [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": [{"type": "tool_addition"}]}]})
        self.assertEqual(o["messages"], [{"role": "user", "content": "hi"}])


if __name__ == "__main__":
    unittest.main()
