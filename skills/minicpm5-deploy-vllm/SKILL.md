---
name: minicpm5-deploy-vllm
description: Serve MiniCPM5-1B or MiniCPM5-2B via vLLM as an OpenAI-compatible HTTP server. Use when the user wants high-throughput production serving on NVIDIA GPU, asks for "vLLM", "OpenAI server", "REST API for MiniCPM5", or "production deployment".
  Also covers MiniCPM5-15B-A2.5B BF16, INT4 Marlin and DSpark with the vLLM preview.
---

# Deploy MiniCPM5-1B and MiniCPM5-2B with vLLM

OpenAI-compatible server for the BF16 / FP16 MiniCPM5-1B or MiniCPM5-2B checkpoint.

For MiniCPM5-15B-A2.5B, start at [MiniCPM5-15B-A2.5B](#minicpm5-15b-a25b).

## Required input

| Var | Example | Default |
| --- | --- | --- |
| `MODEL_PATH` | `openbmb/MiniCPM5-2B` | required; `openbmb/MiniCPM5-1B` also works |
| `PORT` | `8000` | `8000` |
| `GPU_ID` | `0` | `0` |
| `CTX_LEN` | `131072` (128 K) | `131072`; lower if VRAM tight |
| `MEM_FRAC` | `0.85` | `0.85`; lower on shared GPUs |

## Steps

### 1. Install (once)

```bash
pip install "vllm>=0.21"          # latest (CUDA 13.x driver hosts)
# pip install "vllm==0.10.1.1"    # fallback for CUDA 12.x driver hosts
```

### 2. Launch

```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} vllm serve "${MODEL_PATH}" \
    --served-model-name MiniCPM5-2B \
    --dtype bfloat16 \
    --max-model-len ${CTX_LEN} \
    --gpu-memory-utilization ${MEM_FRAC} \
    --port ${PORT}
```

Wait for `Application startup complete` in the log.

### 3. Validate

```bash
curl http://localhost:${PORT}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniCPM5-2B",
        "messages": [{"role": "user", "content": "1+1=?"}],
        "temperature": 1.0, "top_p": 0.95, "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": true}
    }'
```

Expected: `choices[0].message.content` contains a coherent answer.

## Sampling defaults

```json
{"temperature": 1.0, "top_p": 0.95, "chat_template_kwargs": {"enable_thinking": true}}    // MiniCPM5-2B Think
{"temperature": 0.9, "top_p": 0.95, "chat_template_kwargs": {"enable_thinking": true}}    // MiniCPM5-1B Think
{"temperature": 0.7, "top_p": 0.95, "chat_template_kwargs": {"enable_thinking": false}}   // MiniCPM5-1B No-think
```

## Common pitfalls

- **`(free / total) < MEM_FRAC` hard error**: lower `--gpu-memory-utilization` (e.g. 0.5 on a shared GPU).
- **OOM at startup with 128 K**: drop `--max-model-len` to 32768 or 8192.

## Tool calling (plugin)

The vLLM-side MiniCPM5 XML parser ([PR #43175](https://github.com/vllm-project/vllm/pull/43175)) merged to `main` on 2026-05-27 but is **not in any pip release yet** (`v0.22.0` was cut before the merge). Use the bridge plugin shipped at `tool_parsers/minicpm5xml_tool_parser.py` in this repo:

```bash
vllm serve "${MODEL_PATH}" \
    --served-model-name MiniCPM5-2B \
    --dtype bfloat16 --max-model-len ${CTX_LEN} --port ${PORT} \
    --enable-auto-tool-choice \
    --tool-parser-plugin /path/to/MiniCPM/tool_parsers/minicpm5xml_tool_parser.py \
    --tool-call-parser minicpm5
```

Drop `--tool-parser-plugin` once vLLM ships a release containing the parser natively.

## When NOT to use

- One-shot Python script → `minicpm5-deploy-transformers`
- Apple Silicon / no NVIDIA GPU → `minicpm5-deploy-llama-cpp` / `minicpm5-deploy-mlx`
- High-concurrency batch eval w/ prefix cache or tool calling → `minicpm5-deploy-sglang`

## Reference

[`docs/deployment/vllm.md`](../../docs/deployment/vllm.md)

## MiniCPM5-15B-A2.5B

Use the [MiniCPM5-15B-A2.5B deployment commands](../../docs/deployment/vllm.md#minicpm5-15b-a25b). For a standalone skill installation, open the [GitHub cookbook](https://github.com/OpenBMB/MiniCPM/blob/main/docs/deployment/vllm.md#minicpm5-15b-a25b).

1. Use the pinned source from [vLLM PR #68748](https://github.com/vllm-project/vllm/pull/68748) with the cookbook's official CUDA nightly and Python overlay. Set `MODEL_DIR` and `TOKENIZER_DIR` to the target weights and compatible fast tokenizer with its chat template.
2. Launch with the cookbook's single-H800 settings. For the compressed-tensors INT4 checkpoint, select Marlin for both Linear and MoE. Check `/v1/models` for `MiniCPM5-15B-A2.5B` and send the chat sample.
3. For DSpark, set `DSPARK_DIR` to the draft checkpoint trained for the selected MiniCPM5-15B-A2.5B target and set `num_speculative_tokens` to `7`. Check the logs or `/metrics` for draft activity. The earlier INT4 check matched greedy tokens on two of four prompts across draft modes.
