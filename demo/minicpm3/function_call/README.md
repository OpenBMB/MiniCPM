# MiniCPM FunctionCall

1. Start VLLM functioncall server

```shell
python -m vllm.entrypoints.openai.api_server \
    --model openbmb/MiniCPM3-4B \
    --dtype auto \
    --api-key token-abc123 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser minicpm \
    --tool-parser-plugin minicpm_tool_parser.py
```


2. Functioncall client example

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="token-abc123")

tools = [
  {
    "type": "function",
    "function": {
      "name": "get_current_weather",
      "description": "Get the current weather in a given location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA",
          },
          "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"],
      },
    }
  }
]
messages = [{"role": "user", "content": "What's the weather like in Boston today?"}]
completion = client.chat.completions.create(
  model="openbmb/MiniCPM3-4B",
  messages=messages,
  tools=tools,
  tool_choice="auto"
)

print(completion)

```


3. Run functioncall inference locally

```shell
python functioncall.py
```


# Thanks

- resolve_ast_call and resolve_ast_by_type from [gorilla](https://github.com/ShishirPatil/gorilla)
- minicpm chat template with tool from @CISCai