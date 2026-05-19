# Deploy MiniCPM5-1B GPTQ-Marlin Int4

A **symmetric GPTQ-Marlin Int4** build of MiniCPM5-1B is published as `openbmb/MiniCPM5-1B-GPTQ-Marlin-Int4`. It is calibrated with 1024 mixed-domain samples and consumes ~1.1 GB on disk while preserving full 128 K context.

## Verified versions

| Component | Version |
| --- | --- |
| vLLM | 0.10 + (`--quantization gptq_marlin`) |
| `torch` | 2.7 + (cu126) |
| GPU | Hopper / Ada / Ampere (tested on H200) |

## Quantization config

```json
{
  "quantization_config": {
    "bits": 4,
    "group_size": 128,
    "damp_percent": 0.1,
    "desc_act": false,
    "quant_method": "gptq",
    "static_groups": false,
    "sym": true,
    "true_sequential": true
  },
  "torch_dtype": "float16"
}
```

vLLM transparently uses the **Marlin kernel** for this layout when you pass `--quantization gptq_marlin`.

## OpenAI-compatible server

```bash
pip install "vllm>=0.6.0"

python -m vllm.entrypoints.openai.api_server \
    --model openbmb/MiniCPM5-1B-GPTQ-Marlin-Int4 \
    --served-model-name MiniCPM5-1B-GPTQ \
    --dtype float16 \
    --quantization gptq_marlin \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.5 \
    --port 8000
```

## Verified run

```bash
$ curl -sS http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniCPM5-1B-GPTQ",
        "messages": [{"role": "user", "content": "1+1=?"}],
        "temperature": 0.6, "top_p": 0.95, "max_tokens": 64
    }'

# reply: "<think>
# We are asked: '1+1=?' This is a simple arithmetic question. 1+1=2 ..."
# usage: prompt_tokens=14, completion_tokens=64
```

The `<think>...</think>` block is the default chat-template behavior. Set `chat_template_kwargs.enable_thinking=false` for a fast direct answer:

| Mode | `enable_thinking` | `temperature` | `top_p` |
| --- | --- | ---: | ---: |
| Think (default) | `true` (or omit) | 0.6 | 0.95 |
| No-think | `false` | 0.7 | 0.8 |

The quantization layer (Marlin Int4) doesn't touch the chat template, so think / no-think behavior is identical to the fp16 build — only the weights are quantized.

## AWQ vs GPTQ — practical guidance

Both builds use the same Marlin kernel and have very similar runtime characteristics. Pick by:

- **AWQ** tends to do slightly better on instruction-following / chat content because it preserves activation outliers.
- **GPTQ** tends to do slightly better on highly structured tasks (math, code) where per-channel rounding matters.

In practice the gap is < 1 point on most public benchmarks; we recommend running both for your target domain and picking the winner.

## See also

- [`awq.md`](./awq.md) — sister AWQ build
- [`vllm.md`](./vllm.md) — full vLLM tuning
