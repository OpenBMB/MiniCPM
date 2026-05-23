---
name: minicpm5-deploy-gptq
description: Serve MiniCPM5-1B GPTQ-Marlin Int4 via vLLM. Use when the user wants Int4 / low-VRAM serving on NVIDIA GPU and prefers GPTQ over AWQ, asks for "GPTQ", "gptq_marlin", or points at the `openbmb/MiniCPM5-1B-GPTQ-Marlin-Int4` checkpoint.
---

# Deploy MiniCPM5-1B GPTQ-Marlin Int4 (vLLM)

Symmetric GPTQ-Marlin Int4. ~1.1 GB ckpt, full 128 K context, runs on Hopper / Ada / Ampere.

## Required input

| Var | Example | Default |
| --- | --- | --- |
| `MODEL_PATH` | `openbmb/MiniCPM5-1B-GPTQ-Marlin-Int4` | required (must be the GPTQ ckpt) |
| `PORT` | `8000` | `8000` |
| `GPU_ID` | `0` | `0` |
| `MEM_FRAC` | `0.5` | `0.5` |

## Steps

### 1. Install

```bash
pip install "vllm>=0.21"          # latest (CUDA 13.x driver hosts)
# pip install "vllm==0.10.1.1"    # fallback for CUDA 12.x driver hosts
```

### 2. Launch

```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name MiniCPM5-1B-GPTQ \
    --dtype float16 \
    --quantization gptq_marlin \
    --max-model-len 131072 \
    --gpu-memory-utilization ${MEM_FRAC} \
    --port ${PORT}
```

> ⚠️ `--dtype float16` (not bf16).
> ⚠️ Always pass `--quantization gptq_marlin` explicitly.

### 3. Validate

```bash
curl http://localhost:${PORT}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniCPM5-1B-GPTQ",
        "messages": [{"role": "user", "content": "1+1=?"}],
        "temperature": 0.7, "top_p": 0.95, "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": false}
    }'
```

Expected: `"2"` in the reply. If you see `<think>...`, you forgot `enable_thinking: false`.

## Think vs no-think

Same as fp16 / AWQ — the quantization layer doesn't touch the chat template. Use `chat_template_kwargs`:

| Mode | `enable_thinking` | `temperature` | `top_p` |
| --- | --- | ---: | ---: |
| Think (default) | `true` (or omit) | 0.9 | 0.95 |
| No-think | `false` | 0.7 | 0.95 |

## AWQ vs GPTQ — which to pick

Both share the Marlin kernel and have very similar runtime characteristics. In practice the gap is < 1 point on most public benchmarks. AWQ tends to do slightly better on instruction-following / chat; GPTQ tends to do slightly better on highly structured tasks (math / code). Pick by your target domain.

## When NOT to use

- The base bf16 ckpt → `minicpm5-deploy-vllm` (block A)
- AWQ alternative → `minicpm5-deploy-awq`
- Non-NVIDIA → use GGUF (`minicpm5-deploy-llama-cpp`)

## Reference

[`docs/deployment/gptq.md`](../../docs/deployment/gptq.md)
