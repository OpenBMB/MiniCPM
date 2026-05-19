# Deploy MiniCPM5-1B with SGLang

[SGLang](https://github.com/sgl-project/sglang) serves MiniCPM5-1B as a standard `LlamaForCausalLM` with **RadixAttention prefix cache**, very high concurrency, and an OpenAI-compatible API.

## Verified versions

| Component | Version |
| --- | --- |
| SGLang | 0.5.6 + |
| `torch` | 2.7 + (cu126) or 2.9 + (cu128); pick the build matching your driver |
| Python | 3.10 |

## Install

```bash
pip install "sglang==0.5.6.post3"          # recommended for CUDA 12.x drivers
# pip install "sglang[srt]"                # latest, requires CUDA 13.x driver
```

Recommended runtime env vars:

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
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
    --host 0.0.0.0 \
    --port 30000
```

### Tuning knobs

| Flag | Default | When to change |
| --- | --- | --- |
| `--context-length` | `131072` (native 128 K) | drop for small / shared GPUs |
| `--mem-fraction-static` | `0.85` | drop on shared GPUs |
| `--dtype` | `bfloat16` | use `float16` on Ampere or older |

## Chat completion

```bash
curl http://localhost:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniCPM5-1B",
        "messages": [{"role": "user", "content": "用一句话解释什么是 GQA。"}],
        "temperature": 0.6, "top_p": 0.95, "max_tokens": 1024,
        "chat_template_kwargs": {"enable_thinking": true}
    }'
```

| Mode | `enable_thinking` | `temperature` | `top_p` |
| --- | --- | --- | --- |
| Think | `true` | 0.6 | 0.95 |
| No-think | `false` | 0.7 | 0.8 |

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
        "temperature": 0.6, "top_p": 0.95, "max_new_tokens": 1024,
        "skip_special_tokens": False,
    },
)
print(outputs)
```
