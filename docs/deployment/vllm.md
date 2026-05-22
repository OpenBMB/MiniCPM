# Deploy MiniCPM5-1B with vLLM

vLLM ≥ 0.6 supports MiniCPM5-1B natively — no custom kernels. For production-grade throughput and OpenAI-compatible chat completions, this is the recommended path. For the quantized variants on the same vLLM binary see [`awq.md`](./awq.md) and [`gptq.md`](./gptq.md).

## Verified versions

| Component | Version | Result |
| --- | --- | --- |
| vLLM | **0.10.1** (also 0.9.1) | OpenAI server ✅ chat ✅ think + nothink ✅ |
| `torch` | 2.7.1 + cu126 | bfloat16 / float16 |
| Python | 3.10 | |

## Install

```bash
pip install "vllm>=0.6.0"
```

## OpenAI-compatible server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model openbmb/MiniCPM5-1B \
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
| `--dtype` | `bfloat16` | `float16` for older GPUs (A100/H100/H200 bf16 is preferred) |
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

## Verified run

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

Verified on a single H200 with `--max-model-len 8192 --gpu-memory-utilization 0.5`, prompt → reply round trip ≈ 1 s.

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

## Quantized variants

The same vLLM binary can serve the **AWQ-Marlin Int4** and **GPTQ-Marlin Int4** builds — see [`awq.md`](./awq.md) and [`gptq.md`](./gptq.md). Just point `--model` at the quantized checkpoint and add `--quantization awq_marlin` / `--quantization gptq_marlin`.
