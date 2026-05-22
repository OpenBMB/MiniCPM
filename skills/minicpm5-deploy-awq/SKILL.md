---
name: minicpm5-deploy-awq
description: Serve MiniCPM5-1B AWQ-Sym-Marlin Int4 via vLLM. Use when the user wants Int4 / low-VRAM serving on NVIDIA GPU, asks for "AWQ", "AWQ-Marlin", or points at the `openbmb/MiniCPM5-1B-AWQ-Sym-Marlin-Int4` checkpoint.
---

# Deploy MiniCPM5-1B AWQ-Marlin Int4 (vLLM)

Symmetric AWQ-Marlin Int4. ~1.1 GB ckpt, full 128 K context, runs on Hopper / Ada / Ampere.

## Required input

| Var | Example | Default |
| --- | --- | --- |
| `MODEL_PATH` | `openbmb/MiniCPM5-1B-AWQ-Sym-Marlin-Int4` | required (must be the AWQ ckpt) |
| `PORT` | `8000` | `8000` |
| `GPU_ID` | `0` | `0` |
| `MEM_FRAC` | `0.5` | `0.5` (fits 128 K KV cache on a 24 GB consumer card; raise on H200) |

## Steps

### 1. Install (once)

```bash
pip install "vllm>=0.6.0"
```

### 2. Launch

```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name MiniCPM5-1B-AWQ \
    --dtype float16 \
    --quantization awq_marlin \
    --max-model-len 131072 \
    --gpu-memory-utilization ${MEM_FRAC} \
    --port ${PORT}
```

> ⚠️ Use `--dtype float16`, not bf16 — the checkpoint is fp16-based.
> ⚠️ Always pass `--quantization awq_marlin` explicitly; vLLM's auto-detection has been brittle across versions.

### 3. Validate

```bash
curl http://localhost:${PORT}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniCPM5-1B-AWQ",
        "messages": [{"role": "user", "content": "1+1=?"}],
        "temperature": 0.7, "top_p": 0.95, "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": false}
    }'
```

Expected: `"2"` in the reply. If you see `<think>...`, you forgot `enable_thinking: false`.

## When NOT to use

- The base bf16 ckpt → `minicpm5-deploy-vllm` (block A)
- GPTQ alternative → `minicpm5-deploy-gptq`
- Non-NVIDIA → none of the AWQ paths apply; use GGUF (`minicpm5-deploy-llama-cpp`)

## Reference

[`docs/deployment/awq.md`](../../docs/deployment/awq.md)
