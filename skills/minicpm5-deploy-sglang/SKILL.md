---
name: minicpm5-deploy-sglang
description: Serve MiniCPM5-1B via SGLang as an OpenAI-compatible HTTP server with RadixAttention prefix cache. Use when the user asks for "SGLang", "RadixAttention", "prefix cache", batch evaluation, or wants a high-concurrency NVIDIA-GPU server alternative to vLLM.
---

# Deploy MiniCPM5-1B with SGLang

OpenAI-compatible server with RadixAttention prefix cache. Best fit for **batched eval pipelines** and high-concurrency serving.

## Required input

| Var | Example | Default |
| --- | --- | --- |
| `MODEL_PATH` | `openbmb/MiniCPM5-1B` | required |
| `PORT` | `30000` | `30000` |
| `GPU_ID` | `0` | `0` |
| `CTX_LEN` | `131072` (128 K) | `131072` |
| `MEM_FRAC` | `0.85` | `0.85` |

## Steps

### 1. Install (once)

```bash
pip install "sglang[srt]>=0.5.12"          # latest, requires CUDA 13.x driver
# pip install "sglang==0.5.6.post3"        # fallback for CUDA 12.x driver hosts
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

- **`GLIBCXX_3.4.31 not found`**: conda Python ships an older `libstdc++`. Force-load system: `LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6 python -m sglang.launch_server ...`

## When NOT to use

- Production serving without batch-eval needs → `minicpm5-deploy-vllm` is simpler
- One-shot Python → `minicpm5-deploy-transformers`
- No NVIDIA GPU → `minicpm5-deploy-llama-cpp` / `minicpm5-deploy-mlx`

## Reference

[`docs/deployment/sglang.md`](../../docs/deployment/sglang.md)
