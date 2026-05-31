---
name: minicpm5-deploy-sglang
description: Serve MiniCPM5-1B via SGLang as an OpenAI-compatible HTTP server with RadixAttention prefix cache and built-in MiniCPM5 tool-call parsing. Use when the user asks for "SGLang", "RadixAttention", "prefix cache", batch evaluation, tool calling, or wants a high-concurrency NVIDIA-GPU server alternative to vLLM.
---

# Deploy MiniCPM5-1B with SGLang

OpenAI-compatible server with RadixAttention prefix cache. Best fit for **tool calling**, batched eval pipelines, and high-concurrency serving.

## Required input

| Var | Example | Default |
| --- | --- | --- |
| `MODEL_PATH` | `openbmb/MiniCPM5-1B` | required |
| `PORT` | `30000` | `30000` |
| `GPU_ID` | `0` | `0` |
| `CTX_LEN` | `131072` (128 K) | `131072` |
| `MEM_FRAC` | `0.85` | `0.85` |
| `TOOL_PARSER` | `minicpm5` | `minicpm5`; use `auto` if you want template detection |

## Steps

### 1. Install (once)

```bash
pip install "sglang[srt]>=0.5.12"          # latest, requires CUDA 13.x driver
# pip install "sglang==0.5.6.post3"        # fallback for CUDA 12.x driver hosts
```

**For tool calling, install from `main`** — the `minicpm5` parser ([PR #25600](https://github.com/sgl-project/sglang/pull/25600), merged 2026-05-22) is not in any pip release yet (`v0.5.12.post1` was branched earlier). Plain chat works on the pip release; only `--tool-call-parser minicpm5` needs `main`:

```bash
pip install "git+https://github.com/sgl-project/sglang.git@main#subdirectory=python"
```

### 2. Recommended runtime env vars

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_DISABLE_CUDNN_CHECK=1
```

### 3. Launch

```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --served-model-name MiniCPM5-1B \
    --dtype bfloat16 \
    --context-length ${CTX_LEN} \
    --mem-fraction-static ${MEM_FRAC} \
    --tool-call-parser ${TOOL_PARSER} \
    --host 0.0.0.0 \
    --port ${PORT}
```

Wait for `The server is fired up and ready to roll!` .

### 4. Validate

```bash
curl http://localhost:${PORT}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniCPM5-1B",
        "messages": [{"role": "user", "content": "1+1=?"}],
        "temperature": 0.7, "top_p": 0.95, "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": false}
    }'
```

Expected: `choices[0].message.content` contains `"2"`.

## Tool calling

SGLang includes a native MiniCPM5 XML tool-call parser. Keep `TOOL_PARSER=minicpm5` (or `auto`) when launching, then send an OpenAI-style tools request:

```bash
curl http://localhost:${PORT}/v1/chat/completions \
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

llm = sgl.Engine(model_path="${MODEL_PATH}", tp_size=1, mem_fraction_static=0.8, context_length=131072)

outputs = llm.generate(
    ["用一句话解释什么是 GQA。"],
    sampling_params={"temperature": 0.9, "top_p": 0.95, "max_new_tokens": 1024,
                     "skip_special_tokens": False},
)
print(outputs)
```

## Common pitfalls

- **`GLIBCXX_3.4.31 not found`** (only if you hit it — not universal): conda Python ships an older `libstdc++` than the SGLang wheel was built against. Force-load the system one: `LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6 python -m sglang.launch_server ...`

## When NOT to use

- Production serving without batch-eval needs → `minicpm5-deploy-vllm` is simpler
- One-shot Python → `minicpm5-deploy-transformers`
- No NVIDIA GPU → `minicpm5-deploy-llama-cpp` / `minicpm5-deploy-mlx`

## Reference

[`docs/deployment/sglang.md`](../../docs/deployment/sglang.md)
