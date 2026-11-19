# Deploy MiniCPM5-1B and MiniCPM5-2B with SGLang

[SGLang](https://github.com/sgl-project/sglang) serves MiniCPM5-1B and MiniCPM5-2B as standard `LlamaForCausalLM` models with **RadixAttention prefix cache**, very high concurrency, and an OpenAI-compatible API.

For MiniCPM5-15B-A2.5B, see [MiniCPM5-15B-A2.5B](#minicpm5-15b-a25b).

## Install

```bash
pip install "sglang[srt]>=0.5.16"          # latest, requires CUDA 13.x driver
# pip install "sglang==0.5.6.post3"        # fallback for CUDA 12.x drivers
```

> ⚠️ **Tool calling needs a newer build than any current pip release.** The MiniCPM5 XML parser (`--tool-call-parser minicpm5`) landed in [sgl-project/sglang#25600](https://github.com/sgl-project/sglang/pull/25600), merged into `main` on 2026-05-22. The latest release `v0.5.12.post1` was branched earlier and does **not** ship the parser yet. Until `v0.5.13` is tagged, install from source if you need tool calling:
>
> ```bash
> pip install "git+https://github.com/sgl-project/sglang.git@main#subdirectory=python"
> ```
>
> Plain chat completions (no `tools=`) work fine on the pip release — both checkpoints are stock `LlamaForCausalLM` models, so they load on every SGLang version that supports Llama.

Recommended runtime env vars:

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_DISABLE_CUDNN_CHECK=1
```

## OpenAI-compatible server

```bash
python -m sglang.launch_server \
    --model-path openbmb/MiniCPM5-2B \
    --served-model-name MiniCPM5-2B \
    --dtype bfloat16 \
    --context-length 131072 \
    --mem-fraction-static 0.85 \
    --tool-call-parser minicpm5 \
    --host 0.0.0.0 \
    --port 30000
```

**Speculative decoding (DSpark)**: we also release [MiniCPM5-2B-DSpark](https://huggingface.co/openbmb/MiniCPM5-2B-DSpark), a DSpark draft model trained for MiniCPM5-2B. Enable it in SGLang to accelerate decoding while keeping the target model's outputs unchanged:

```bash
python -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --served-model-name MiniCPM5-2B \
    --dtype bfloat16 \
    --context-length ${CTX_LEN} \
    --mem-fraction-static ${MEM_FRAC} \
    --tool-call-parser ${TOOL_PARSER} \
    --host 0.0.0.0 \
    --port 3000 \
    --speculative-algorithm DSPARK \
    --speculative-draft-model-path openbmb/MiniCPM5-2B-DSpark \
    --speculative-dspark-block-size 7
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
        "model": "MiniCPM5-2B",
        "messages": [{"role": "user", "content": "用一句话解释什么是 GQA。"}],
        "temperature": 1.0, "top_p": 0.95, "max_tokens": 1024,
        "chat_template_kwargs": {"enable_thinking": true}
    }'
```

| Mode | `enable_thinking` | `temperature` | `top_p` |
| --- | --- | --- | --- |
| MiniCPM5-2B Think | `true` | 1.0 | 0.95 |
| MiniCPM5-1B Think | `true` | 0.9 | 0.95 |
| MiniCPM5-1B No-think | `false` | 0.7 | 0.95 |

## Tool calling

MiniCPM5-1B and MiniCPM5-2B emit XML-style tool calls. SGLang ships a built-in `minicpm5` parser that converts them to OpenAI-compatible `tool_calls`.

Start the server with either `--tool-call-parser minicpm5` or `--tool-call-parser auto`, then send a standard OpenAI-style tools request:

```bash
curl http://localhost:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniCPM5-2B",
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
        "temperature": 1.0, "top_p": 0.95,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": true}
    }'
```

## Offline / batched (Engine API)

```python
import sglang as sgl

llm = sgl.Engine(
    model_path="openbmb/MiniCPM5-2B",
    tp_size=1,
    mem_fraction_static=0.8,
    context_length=131072,
)

outputs = llm.generate(
    ["用一句话解释什么是 GQA。"],
    sampling_params={
        "temperature": 1.0, "top_p": 0.95, "max_new_tokens": 1024,
        "skip_special_tokens": False,
    },
)
print(outputs)
```

## MiniCPM5-15B-A2.5B

### Install the SGLang preview

The examples use the BF16 checkpoint with `model_type` set to `minicpm5_moe` and `MiniCPM5MoEForCausalLM` in `architectures`. Install the [pinned SGLang preview](https://github.com/cyxu0401/sglang/commit/2df302467560035d8de0558553ace4a4ebb86ed3) in a Python 3.11+ environment on Linux with a CUDA 13-compatible NVIDIA driver. The BF16 inference smoke test for this revision used one H800.

```bash
python -m pip install "git+https://github.com/cyxu0401/sglang.git@2df302467560035d8de0558553ace4a4ebb86ed3#subdirectory=python"
```

### Start the BF16 server

Set `MODEL_PATH` to the Hugging Face repo ID or the local snapshot directory for the BF16 checkpoint. If you use a local snapshot, include the tokenizer and chat template alongside the weights.

```bash
export MODEL_PATH="/path/to/MiniCPM5-15B-A2.5B"
python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --served-model-name MiniCPM5-15B-A2.5B \
    --dtype bfloat16 --tp-size 1 \
    --context-length 4096 --max-total-tokens 8192 \
    --mem-fraction-static 0.72 --max-running-requests 4 \
    --attention-backend fa3 --moe-runner-backend triton \
    --enable-deterministic-inference --disable-decode-cuda-graph --disable-prefill-cuda-graph \
    --host 127.0.0.1 --port 30000
```

These settings give each request a 4,096-token context budget for the prompt and completion combined. When increasing the context length, size `--max-total-tokens` for the new context and available KV-cache memory.

### Chat completion

From another terminal, check that `GET /v1/models` lists `MiniCPM5-15B-A2.5B`, then send a short request.

```bash
curl -fsS http://127.0.0.1:30000/v1/models
curl -fsS http://127.0.0.1:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniCPM5-15B-A2.5B",
        "messages": [{"role": "user", "content": "Introduce yourself in one sentence."}],
        "temperature": 0,
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": false}
    }'
```

### Tool calling for 15B

Add `--tool-call-parser minicpm5` to the launch command to parse the checkpoint's XML tool calls. The example uses a tool for looking up temperatures. The client supplies the tool implementation.

```bash
curl -fsS http://127.0.0.1:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniCPM5-15B-A2.5B",
        "messages": [
            {"role": "system", "content": "Use lookup_temperature to answer weather questions."},
            {"role": "user", "content": "What is the current temperature in Beijing?"}
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "lookup_temperature",
                "description": "Look up the current temperature for a city in Celsius.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            }
        }],
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": 2048,
        "chat_template_kwargs": {"enable_thinking": true, "has_tool_sep": false}
    }'
```

When the model calls the tool, check `choices[0].message.tool_calls` for the function name and its JSON-encoded arguments. To continue the conversation, send the assistant message with its tool calls and a `role: "tool"` message with the result and matching `tool_call_id`.

### Speculative decoding with DSpark

Set `DRAFT_MODEL_PATH` to the Hugging Face repo ID or local snapshot of the DSpark draft trained for this target checkpoint. Stop the first server and launch DSpark in the same terminal so `MODEL_PATH` stays set.

```bash
export DRAFT_MODEL_PATH="/path/to/MiniCPM5-15B-A2.5B-DSpark"
python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --served-model-name MiniCPM5-15B-A2.5B \
    --dtype bfloat16 --tp-size 1 \
    --context-length 4096 --max-total-tokens 8192 \
    --mem-fraction-static 0.72 --max-running-requests 4 \
    --attention-backend fa3 --moe-runner-backend triton \
    --enable-deterministic-inference --disable-decode-cuda-graph --disable-prefill-cuda-graph \
    --speculative-algorithm DSPARK \
    --speculative-draft-model-path "$DRAFT_MODEL_PATH" \
    --speculative-dspark-block-size 7 \
    --host 127.0.0.1 --port 30000
```

Reuse the chat request for a smoke test and check that `GET /server_info` reports `speculative_algorithm` as `DSPARK`. In an offline inference check on one H800 with BF16, FA3 attention and Triton MoE, the baseline and DSpark matched greedy token IDs for four prompts with 64 generated tokens each. See the [SGLang validation notes](https://github.com/sgl-project/sglang/pull/51522) for the tested settings and results across attention backends.
