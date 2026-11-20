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

Use the BF16 preview build from [SGLang PR #51568](https://github.com/sgl-project/sglang/pull/51568). It registers `MiniCPM5ForCausalLM` for `model_type=minicpm5_moe` and maps the legacy `MiniCPM5MoEForCausalLM` architecture to the native implementation.

### Run on one NVIDIA H800

Use the pinned PR source with the official Linux x86-64 CUDA 13 image. This image includes FA3. Set `MODEL_PATH` to an absolute path to a complete BF16 snapshot with its tokenizer and chat template.

```bash
git init sglang-minicpm5
cd sglang-minicpm5
git fetch --depth 1 https://github.com/cyxu0401/sglang.git 3c6688d1cdc01bc492a0a302e9e2e4241360e0a0
git checkout --detach FETCH_HEAD

MODEL_PATH="/absolute/path/to/MiniCPM5-15B-A2.5B"
docker run --rm --gpus device=0 --ipc=host \
    -p 127.0.0.1:30000:30000 \
    -v "$PWD/python:/sglang-src/python:ro" \
    -v "$MODEL_PATH:/model:ro" \
    -e PYTHONPATH=/sglang-src/python \
    lmsysorg/sglang@sha256:d7976d3a5963ea29eb34a3d94dd2304b3b9a8245da78add7ada9cb15bf77a641 \
    sglang serve --model-path /model \
    --served-model-name MiniCPM5-15B-A2.5B \
    --attention-backend fa3 --host 0.0.0.0 --port 30000
```

The checkpoint sets BF16 and SGLang defaults to TP=1. The port mapping makes the server available on the host's loopback address.

### Chat completion

Check readiness at `/health` and confirm that `/v1/models` lists `MiniCPM5-15B-A2.5B`. Then send a chat request.

```bash
curl -fsS http://127.0.0.1:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniCPM5-15B-A2.5B",
        "messages": [{"role": "user", "content": "Explain grouped-query attention in one sentence."}],
        "temperature": 0,
        "max_tokens": 128
    }'
```

For XML tool calling, add `--tool-call-parser minicpm5` to the launch command and use the [tool request example](#tool-calling) with `MiniCPM5-15B-A2.5B` in its `model` field. Check the parsed `message.tool_calls` on your checkpoint. The model PR's launch example covers ordinary BF16 inference.
