---
name: minicpm5-deploy-mlx
description: Run MiniCPM5-1B natively on Apple Silicon with Apple's MLX framework. Use when the user has an Apple Silicon Mac and asks for "MLX", "mlx_lm", "mlx_lm.convert", "mlx_lm.generate", or wants the fastest path on Apple Silicon.
---

# Deploy MiniCPM5-1B with MLX (Apple Silicon)

Apple's on-device tensor framework. Highest throughput on M-series. Stays inside one Python process — no separate server, no `llama.cpp` build chain.

## Required input

| Var | Example | Default |
| --- | --- | --- |
| `MLX_REPO` | `openbmb/MiniCPM5-1B-MLX-4bit` (pre-converted) | required |
| OR `HF_REPO` + `QUANT` | `openbmb/MiniCPM5-1B`, `4bit` or `bf16` | for local conversion |
| `MAX_TOKENS` | `200` | `200` |

## Steps

### 1. Install (once)

```bash
pip install "mlx-lm>=0.31" "gguf"
```

### 2A. Use a pre-converted MLX repo (recommended when available)

```bash
mlx_lm.generate --model "${MLX_REPO}" \
    --prompt "<|im_start|>user
1+1=?<|im_end|>
<|im_start|>assistant
" \
    --max-tokens ${MAX_TOKENS} --temp 0.7 --top-p 0.95 \
    --extra-eos-token "<|im_end|>"
```

### 2B. Convert from a HF checkpoint locally (advanced)

Use `mlx_lm.convert` only if you have a self-trained HF fp16 checkpoint:

```bash
HF=/path/to/your-fp16-hf

# Convert: bf16
mlx_lm.convert --hf-path "$HF" --mlx-path ./minicpm5-mlx-bf16

# Convert: 4-bit (smaller / faster)
mlx_lm.convert --hf-path "$HF" --mlx-path ./minicpm5-mlx-q4 -q --q-bits 4
```

Then run as in 2A.

### 3. Validate

The reply should contain `"2"` for `1+1=?`.

## OpenAI-compatible server (mlx-lm)

```bash
mlx_lm.server --model "${MLX_REPO}" --host 127.0.0.1 --port 8000

curl http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "default",
        "messages": [{"role":"user","content":"1+1=?"}],
        "temperature": 0.7, "top_p": 0.95, "max_tokens": 64
    }'
```

## Common pitfalls

- **Slow first generate**: MLX JIT-compiles kernels on first call (~5-10 s); subsequent calls hit the warm cache.
- **Model never stops generating**: pass `--extra-eos-token "<|im_end|>"` (CLI) or add `<|im_end|>` to the Python wrapper's stop list — `</s>` alone doesn't terminate ChatML turns.

## When NOT to use

- Not on Apple Silicon → `minicpm5-deploy-llama-cpp` (CPU/CUDA) or `minicpm5-deploy-vllm` (CUDA)
- Want a desktop GUI → `minicpm5-deploy-lmstudio` (LM Studio bundles an MLX runtime)
- Want one-line CLI run → `minicpm5-deploy-ollama`

## Reference

[`docs/deployment/mlx.md`](../../docs/deployment/mlx.md)
