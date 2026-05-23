---
name: minicpm5-deploy-vllm
description: Serve MiniCPM5-1B via vLLM as an OpenAI-compatible HTTP server. Use when the user wants high-throughput production serving on NVIDIA GPU, asks for "vLLM", "OpenAI server", "REST API for MiniCPM5", or "production deployment". Also covers the AWQ-Marlin Int4 and GPTQ-Marlin Int4 quantized variants.
---

# Deploy MiniCPM5-1B with vLLM

OpenAI-compatible server. Handles bf16 / fp16 / AWQ-Marlin Int4 / GPTQ-Marlin Int4.

## Required input

| Var | Example | Default |
| --- | --- | --- |
| `MODEL_PATH` | `openbmb/MiniCPM5-1B` (bf16) / `openbmb/MiniCPM5-1B-AWQ-Sym-Marlin-Int4` / `openbmb/MiniCPM5-1B-GPTQ-Marlin-Int4` | required |
| `PORT` | `8000` | `8000` |
| `GPU_ID` | `0` | `0` |
| `CTX_LEN` | `131072` (128 K) | `131072`; lower if VRAM tight |
| `MEM_FRAC` | `0.85` | `0.85`; lower on shared GPUs |

If `MODEL_PATH` ends with `-AWQ-...` use AWQ block; if `-GPTQ-...` use GPTQ block; otherwise use FP16/bf16 block.

## Steps

### 1. Install (once)

```bash
pip install "vllm>=0.21"          # latest (CUDA 13.x driver hosts)
# pip install "vllm==0.10.1.1"    # fallback for CUDA 12.x driver hosts
```

### 2. Launch — pick ONE of the three based on `MODEL_PATH`

#### A. bf16 (full precision, the canonical `openbmb/MiniCPM5-1B`)

```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name MiniCPM5-1B \
    --dtype bfloat16 \
    --max-model-len ${CTX_LEN} \
    --gpu-memory-utilization ${MEM_FRAC} \
    --port ${PORT}
```

#### B. AWQ-Marlin Int4

```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name MiniCPM5-1B \
    --dtype float16 \
    --quantization awq_marlin \
    --max-model-len ${CTX_LEN} \
    --gpu-memory-utilization ${MEM_FRAC} \
    --port ${PORT}
```

#### C. GPTQ-Marlin Int4

```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name MiniCPM5-1B \
    --dtype float16 \
    --quantization gptq_marlin \
    --max-model-len ${CTX_LEN} \
    --gpu-memory-utilization ${MEM_FRAC} \
    --port ${PORT}
```

Wait for `Application startup complete` in the log .

### 3. Validate

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

Expected: `choices[0].message.content` contains `"2"`. If you see `<think>...</think>`, you forgot `chat_template_kwargs.enable_thinking=false`.

## Sampling defaults

```json
{"temperature": 0.9, "top_p": 0.95, "chat_template_kwargs": {"enable_thinking": true}}    // think
{"temperature": 0.7, "top_p": 0.95, "chat_template_kwargs": {"enable_thinking": false}}   // nothink
```

## Common pitfalls

- **`(free / total) < MEM_FRAC` hard error**: lower `--gpu-memory-utilization` (e.g. 0.5 on a shared GPU).
- **OOM at startup with 128 K**: drop `--max-model-len` to 32768 or 8192.
- **AWQ outputs garbled `awq_marlin` for non-AWQ ckpt**: only pass `--quantization awq_marlin` for the AWQ checkpoint, NOT for plain bf16.

## When NOT to use

- One-shot Python script → `minicpm5-deploy-transformers`
- Apple Silicon / no NVIDIA GPU → `minicpm5-deploy-llama-cpp` / `minicpm5-deploy-mlx`
- High-concurrency batch eval w/ prefix cache → `minicpm5-deploy-sglang`

## Reference

[`docs/deployment/vllm.md`](../../docs/deployment/vllm.md), [`docs/deployment/awq.md`](../../docs/deployment/awq.md), [`docs/deployment/gptq.md`](../../docs/deployment/gptq.md)
