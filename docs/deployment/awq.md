# Deploy MiniCPM5-1B AWQ-Marlin Int4

A **symmetric AWQ-Marlin Int4** build of MiniCPM5-1B is published as `openbmb/MiniCPM5-1B-AWQ-Sym-Marlin-Int4`. It is calibrated with 1024 mixed-domain samples and consumes ~1.1 GB on disk (vs ~2.1 GB for fp16) while preserving full 128K context.

## Verified versions

| Component | Version |
| --- | --- |
| vLLM | 0.10 + (`--quantization awq_marlin`) |
| `torch` | 2.7 + (cu126) |
| GPU | Hopper / Ada / Ampere (tested on H200) |

## Quantization config (already baked into the checkpoint)

```json
{
  "quantization_config": {
    "bits": 4,
    "group_size": 128,
    "quant_method": "awq_marlin",
    "version": "marlin",
    "zero_point": true
  },
  "torch_dtype": "float16"
}
```

The Marlin kernel is enabled by setting `--quantization awq_marlin` at server launch (vLLM also auto-detects it from the config when `--quantization` is omitted, but passing it explicitly is safer across versions).

## Install

```bash
pip install "vllm>=0.6.0"
```

## OpenAI-compatible server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model openbmb/MiniCPM5-1B-AWQ-Sym-Marlin-Int4 \
    --served-model-name MiniCPM5-1B-AWQ \
    --dtype float16 \
    --quantization awq_marlin \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.5 \
    --port 8000
```

Notes:

- The checkpoint stores weights as fp16, so `--dtype float16` (not bf16). vLLM will warn but still work if you pass bfloat16.
- `0.5` GPU memory is enough for 128K context on H200 — drop further on smaller cards.

## Verified run

```bash
$ curl -sS http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniCPM5-1B-AWQ",
        "messages": [{"role": "user", "content": "1+1=?"}],
        "temperature": 0.9, "top_p": 0.95, "max_tokens": 64
    }'

# reply: "<think>
# We are asked: '1+1=?' ... 1+1 = 2."
# usage: prompt_tokens=14, completion_tokens=64
```

The think block (`<think>...</think>`) is the default chat-template behavior. Set `chat_template_kwargs.enable_thinking=false` for a fast direct answer.

## When to pick AWQ vs GPTQ vs FP16

| Build | Disk | KV-cache @128K | Quality vs FP16 | Pick when |
| --- | --- | --- | --- | --- |
| FP16 / BF16 | ~2.1 GB | full | reference | quality is paramount, plenty of VRAM |
| **AWQ-Marlin Int4** | **~1.1 GB** | full | small drop on math, ~unchanged on chat | edge / shared GPU / serving small batches |
| GPTQ-Marlin Int4 | ~1.1 GB | full | similar to AWQ, slightly different per-task | same as AWQ; choose by which calibration matches your domain |

For comparison data on benchmarks and an apples-to-apples report against fp16, see the model card's **Evaluation Results → Quantized variants** section.

## See also

- [`vllm.md`](./vllm.md) — full vLLM tuning (CTX_LEN, MEM_FRAC, throughput knobs)
- [`gptq.md`](./gptq.md) — sister GPTQ build with the same Marlin kernel
