# Deploy MiniCPM5-1B with SGLang

[SGLang](https://github.com/sgl-project/sglang) serves MiniCPM5-1B as a standard `LlamaForCausalLM` with **RadixAttention prefix cache**, very high concurrency, and an OpenAI-compatible API.

## Install

```bash
pip install "sglang[srt]>=0.5.12"          # latest, requires CUDA 13.x driver
# pip install "sglang==0.5.6.post3"        # fallback for CUDA 12.x drivers
```

> ⚠️ **Tool calling needs a newer build than any current pip release.** The MiniCPM5 XML parser (`--tool-call-parser minicpm5`) landed in [sgl-project/sglang#25600](https://github.com/sgl-project/sglang/pull/25600), merged into `main` on 2026-05-22. The latest release `v0.5.12.post1` was branched earlier and does **not** ship the parser yet. Until `v0.5.13` is tagged, install from source if you need tool calling:
>
> ```bash
> pip install "git+https://github.com/sgl-project/sglang.git@main#subdirectory=python"
> ```
>
> Plain chat completions (no `tools=`) work fine on the pip release — MiniCPM5-1B is a stock `LlamaForCausalLM` so it loads on every SGLang version that supports Llama.

Recommended runtime env vars:

```bash
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_DISABLE_CUDNN_CHECK=1
```

## OpenAI-compatible server

```bash
python -m sglang.launch_server \
    --model-path openbmb/MiniCPM5-1B \
    --served-model-name MiniCPM5-1B \
    --dtype bfloat16 \
    --context-length 131072 \
    --mem-fraction-static 0.85 \
    --tool-call-parser minicpm5 \
    --host 0.0.0.0 \
    --port 30000
```

### Tuning knobs

| Flag | Default | When to change |
| --- | --- | --- |
| `--context-length` | `131072` (native 128K) | drop for small / shared GPUs |
| `--mem-fraction-static` | `0.85` | drop on shared GPUs |
| `--dtype` | `bfloat16` | use `float16` on Ampere or older |

## Chat completion

```bash
curl http://localhost:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniCPM5-1B",
        "messages": [{"role": "user", "content": "用一句话解释什么是 GQA。"}],
        "temperature": 0.9, "top_p": 0.95, "max_tokens": 1024,
        "chat_template_kwargs": {"enable_thinking": true}
    }'
```

| Mode | `enable_thinking` | `temperature` | `top_p` |
| --- | --- | --- | --- |
| Think | `true` | 0.9 | 0.95 |
| No-think | `false` | 0.7 | 0.95 |

## Tool calling

MiniCPM5-1B emits XML-style tool calls. SGLang ships a built-in `minicpm5` parser that converts them to OpenAI-compatible `tool_calls`.

Start the server with either `--tool-call-parser minicpm5` or `--tool-call-parser auto`, then send a standard OpenAI-style tools request:

```bash
curl http://localhost:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniCPM5-1B",
        "messages": [{"role": "user", "content": "What is the weather in Beijing?"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            }
        }],
        "tool_choice": "auto",
        "temperature": 0.7,
        "max_tokens": 256
    }'
```

## Offline / batched (Engine API)

```python
import sglang as sgl

llm = sgl.Engine(
    model_path="openbmb/MiniCPM5-1B",
    tp_size=1,
    mem_fraction_static=0.8,
    context_length=131072,
)

outputs = llm.generate(
    ["用一句话解释什么是 GQA。"],
    sampling_params={
        "temperature": 0.9, "top_p": 0.95, "max_new_tokens": 1024,
        "skip_special_tokens": False,
    },
)
print(outputs)
```
