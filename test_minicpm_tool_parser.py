import importlib.util
import logging
import sys
import types
import unittest
from pathlib import Path


class _Obj:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.__dict__.update(kwargs)


class _ToolParser:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


class _ToolParserManager:
    @staticmethod
    def register_module(name):
        def deco(cls):
            return cls

        return deco


def _install_stubs():
    def add(name):
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        return mod

    for name in (
        "transformers",
        "vllm",
        "vllm.entrypoints",
        "vllm.entrypoints.openai",
        "vllm.entrypoints.openai.protocol",
        "vllm.entrypoints.openai.tool_parsers",
        "vllm.entrypoints.openai.tool_parsers.abstract_tool_parser",
        "vllm.logger",
    ):
        add(name)
    sys.modules["transformers"].PreTrainedTokenizerBase = object
    protocol = sys.modules["vllm.entrypoints.openai.protocol"]
    protocol.ChatCompletionRequest = _Obj
    protocol.DeltaFunctionCall = _Obj
    protocol.DeltaMessage = _Obj
    protocol.DeltaToolCall = _Obj
    protocol.ExtractedToolCallInformation = _Obj
    protocol.FunctionCall = _Obj
    protocol.ToolCall = _Obj
    abstract = sys.modules["vllm.entrypoints.openai.tool_parsers.abstract_tool_parser"]
    abstract.ToolParser = _ToolParser
    abstract.ToolParserManager = _ToolParserManager
    sys.modules["vllm.logger"].init_logger = logging.getLogger


def _load():
    _install_stubs()
    path = (
        Path(__file__).resolve().parent
        / "demo"
        / "minicpm3"
        / "function_call"
        / "minicpm_tool_parser.py"
    )
    spec = importlib.util.spec_from_file_location("minicpm_tool_parser", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parser_mod = _load()


class TestMiniCPMToolParser(unittest.TestCase):
    def test_plain_text_is_not_a_tool_call(self):
        info = parser_mod.MiniCPMToolParser(None).extract_tool_calls("hello", None)
        self.assertFalse(info.tools_called)
        self.assertEqual(info.tool_calls, [])
        self.assertEqual(info.content, "hello")

    def test_a_parsed_call_is_still_a_tool_call(self):
        text = '<|tool_call_start|>get_weather(city="Paris")<|tool_call_end|>'
        info = parser_mod.MiniCPMToolParser(None).extract_tool_calls(text, None)
        self.assertTrue(info.tools_called)
        self.assertEqual(len(info.tool_calls), 1)
        self.assertEqual(info.tool_calls[0].function.name, "get_weather")
        self.assertIn("Paris", info.tool_calls[0].function.arguments)


if __name__ == "__main__":
    unittest.main()
