---
name: minicpm5-deploy-mlx
description: Run MiniCPM5-1B natively on Apple Silicon with Apple's MLX framework (highest on-device throughput). Use when the user has an M1–M4 Mac and asks for "MLX", "mlx_lm", "mlx_lm.convert", "mlx_lm.generate", or wants the fastest path on Apple Silicon.
---

# Deploy MiniCPM5-1B with MLX (Apple Silicon)

Apple's on-device tensor framework. Highest throughput on M-series. Stays inside one Python process — no separate server, no `llama.cpp` build chain.

## Required input

| Var | Example | Default |
| --- | --- | --- |
| `MLX_REPO` | `openbmb/MiniCPM5-1B-MLX-4bit` (pre-converted) | required |
| OR `HF_REPO` + `QUANT` | `openbmb/MiniCPM5-1B`, `4bit` or `bf16` | for local conversion |
| `MAX_TOKENS` | `200` | `200` |

Verified on Apple M4 / 16 GB / mlx-lm 0.31.3:

| Quant | Disk | Peak RAM | Prompt | Gen |
| --- | --- | --- | --- | --- |
| bf16 | 2.0 GB | 2.2 GB | 228 tok/s | **50 tok/s** |
| Q4 (4.5 bpw) | 589 MB | 0.7 GB | 168 tok/s | **157 tok/s** |

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
    --max-tokens ${MAX_TOKENS} --temp 0.7 --top-p 0.8 \
    --extra-eos-token "<|im_end|>"
```

### 2B. Convert from a HF checkpoint locally

If a pre-converted MLX repo is not available, convert from the HF base. **`mlx_lm.convert` < 0.31 silently drops `lm_head` if `tie_word_embeddings` is missing**, so a one-time HF metadata patch is required before convert (the released `openbmb/MiniCPM5-1B` ships with the patch already applied; you only need this for self-trained checkpoints):

```bash
# Apply the two metadata fixes from docs/deployment/mlx.md → "Required HF-side patch"
# (sets tie_word_embeddings=false in config.json + tokenizer_class=PreTrainedTokenizerFast
#  in tokenizer_config.json; writes the patched dir at /path/to/hf-fp16-fixed/).

HF=/path/to/your-hf-fp16-fixed

# Convert: bf16
mlx_lm.convert --hf-path "$HF" --mlx-path ./minicpm5-mlx-bf16

# Convert: 4-bit (smaller / faster)
mlx_lm.convert --hf-path "$HF" --mlx-path ./minicpm5-mlx-q4 -q --q-bits 4
```

Then run as in 2A.

### 3. Validate

The reply should contain `"2"` for `1+1=?`. If output is gibberish (`Ċ` / `Ġ` tokens), the byte-level BPE didn't decode — the HF checkpoint's `tokenizer_class` is wrong (`LlamaTokenizerFast` instead of `PreTrainedTokenizerFast`). The pre-released MLX repos already have this fixed; for local conversion, apply the HF-side patch from [`docs/deployment/mlx.md`](../../docs/deployment/mlx.md#required-hf-side-patch).

## OpenAI-compatible server (mlx-lm)

```bash
mlx_lm.server --model "${MLX_REPO}" --host 127.0.0.1 --port 8000

curl http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "default",
        "messages": [{"role":"user","content":"1+1=?"}],
        "temperature": 0.7, "top_p": 0.8, "max_tokens": 64
    }'
```

## Common pitfalls

- **Output is `ĊWeĠareĠgivenĠ"1+1=?"…` (byte-level BPE tags)**: HF metadata not patched. The pre-released MLX repo is fine; for local conversion, ensure `tokenizer_config.json` has `tokenizer_class: PreTrainedTokenizerFast`.
- **Output is uniform random tokens**: MLX dropped `lm_head` because `tie_word_embeddings` was missing. Apply the HF-side patch from [`docs/deployment/mlx.md`](../../docs/deployment/mlx.md#required-hf-side-patch) and re-convert.
- **Slow first generate**: MLX JIT-compiles kernels on first call (~5-10 s); subsequent calls hit the warm cache.

## When NOT to use

- Not on Apple Silicon → `minicpm5-deploy-llama-cpp` (CPU/CUDA) or `minicpm5-deploy-vllm` (CUDA)
- Want a desktop GUI → `minicpm5-deploy-lmstudio` (LM Studio bundles an MLX runtime)
- Want one-line CLI run → `minicpm5-deploy-ollama`

## Reference

[`docs/deployment/mlx.md`](../../docs/deployment/mlx.md)
