# Deploy MiniCPM5-1B with vLLM

vLLM `>=0.21` supports MiniCPM5-1B natively — no custom kernels. For production-grade throughput and OpenAI-compatible chat completions, this is the recommended path.

## Install

```bash
pip install "vllm>=0.21"          # latest (CUDA 13.x driver hosts)
# pip install "vllm==0.10.1.1"    # fallback for CUDA 12.x driver hosts
```

## OpenAI-compatible server

```bash
vllm serve openbmb/MiniCPM5-1B \
    --served-model-name MiniCPM5-1B \
    --dtype bfloat16 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.85 \
    --port 8000
```

### Tuning knobs

| Flag | Default | When to change |
| --- | --- | --- |
| `--max-model-len` | `131072` (native 128K) | drop to `8192` / `32768` to free KV-cache on small GPUs |
| `--gpu-memory-utilization` | `0.85` | drop on **shared** GPUs — vLLM hard-fails if `(free / total) < value` |
| `--dtype` | `bfloat16` | `float16` for older GPUs (newer NVIDIA GPUs prefer bf16) |
| `--enforce-eager` | unset | set if CUDA graphs OOM on tiny VRAM budgets |

## Chat completions

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniCPM5-1B",
        "messages": [{"role": "user", "content": "用一句话解释什么是 GQA。"}],
        "temperature": 0.9,
        "top_p": 0.95,
        "max_tokens": 1024,
        "chat_template_kwargs": {"enable_thinking": true}
    }'
```

| Mode | `enable_thinking` | `temperature` | `top_p` |
| --- | --- | --- | --- |
| Think | `true` | 0.9 | 0.95 |
| No-think | `false` | 0.7 | 0.95 |

## Sample run

```bash
$ curl -sS http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"MiniCPM5-1B","messages":[{"role":"user","content":"1+1=?"}],"max_tokens":64}'
{
  "id": "chatcmpl-...",
  "model": "MiniCPM5-1B",
  "choices": [{"message": {"role": "assistant", "content": "2"}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 14, "completion_tokens": 2, "total_tokens": 16}
}
```

## Offline / batched inference

```python
from vllm import LLM, SamplingParams

llm = LLM(model="openbmb/MiniCPM5-1B", dtype="bfloat16", max_model_len=131072)
out = llm.chat(
    [[{"role": "user", "content": "用一句话解释 GQA。"}]],
    SamplingParams(temperature=0.9, top_p=0.95, max_tokens=512),
    chat_template_kwargs={"enable_thinking": True},
)
print(out[0].outputs[0].text)
```
