# Deploy MiniCPM5-1B with MLX (Apple Silicon)

[MLX](https://github.com/ml-explore/mlx) is Apple's on-device tensor framework. For MiniCPM5-1B it is the recommended path on Apple Silicon (M1–M4) when you want **highest throughput** and want to stay inside one Python process — no separate server, no `llama.cpp` build chain.

## TL;DR

```bash
pip install "mlx-lm>=0.31"

# Run a pre-converted MLX repo directly
mlx_lm.generate --model openbmb/MiniCPM5-1B-MLX-4bit \
    --prompt "<|im_start|>user
1+1=?<|im_end|>
<|im_start|>assistant
" \
    --max-tokens 200 --temp 0.7 --top-p 0.95 \
    --extra-eos-token "<|im_end|>"
```

## Building MLX weights from your own checkpoint (advanced)

If you have a self-trained HF fp16 checkpoint and want to produce MLX weights, use `mlx_lm.convert`:

```bash
HF=/path/to/your-fp16-hf

# bf16 master copy
mlx_lm.convert --hf-path "$HF" --mlx-path ./minicpm5-mlx-bf16

# 4-bit (smaller / faster, ~4.5 bits/weight on average)
mlx_lm.convert --hf-path "$HF" --mlx-path ./minicpm5-mlx-q4 -q --q-bits 4
```

The 4-bit pass logs `[INFO] Quantized model with 4.501 bits per weight.` — the slight overshoot above 4 bits is from keeping `embed_tokens` and `lm_head` in higher precision, which preserves quality on the small, untied vocabulary head.

## Inference

### One-shot generate

```bash
mlx_lm.generate --model ./minicpm5-mlx-q4 \
    --prompt "<|im_start|>user
鸡兔同笼，头共10个，脚共28只，问鸡和兔各几只？<|im_end|>
<|im_start|>assistant
" \
    --max-tokens 200 --temp 0.7 --top-p 0.95 \
    --extra-eos-token "<|im_end|>"
```

```text
首先，理解问题：总共有10个头（鸡和兔都是头），总共有28只脚。我们需要找出鸡和兔各自的数量。

设鸡的数量为x，兔的数量为y。那么：
1. 头数：总头数 = 鸡的数量 + 兔的数量 = x + y = 10。
2. 脚数：鸡有2只脚，兔有4只脚，总脚数 = 2x + 4y = 28。
…
```

### Python API (streaming)

```python
from mlx_lm import load, stream_generate

model, tk = load("./minicpm5-mlx-q4")

prompt = (
    "<|im_start|>user\n"
    "用一句话解释什么是 GQA。<|im_end|>\n"
    "<|im_start|>assistant\n"
)

for resp in stream_generate(
    model, tk, prompt=prompt, max_tokens=512,
    sampler=None,                    # default temp/top_p
):
    print(resp.text, end="", flush=True)
print()
```

> 💡 `--extra-eos-token "<|im_end|>"` (CLI) or adding `<|im_end|>` to the wrapper's stop list (Python) is required: the released metadata only registers `</s>` (id 1) and the secondary EOS id 130073 as model EOS, but the chat template ends turns with the `<|im_end|>` *string*. Without an extra EOS the model will keep generating into the next role's prompt.

## Recommended sampling

| Mode | `--temp` | `--top-p` | When to use |
| --- | --- | --- | --- |
| Think | 0.9 | 0.95 | reasoning, math, code, multi-step (model auto-emits `<think>` block) |
| No-think | 0.7 | 0.95 | fast assistant, latency-bound |

Both modes are activated by sampling parameters only — the released chat template auto-injects `<think>\n` when no `system` message disables it, so you get think-mode behaviour by default.

## Q&A

### Model never stops generating

Add `--extra-eos-token "<|im_end|>"` (CLI) or include `<|im_end|>` in `stop` (Python). See note above.

## See also

- [`transformers.md`](./transformers.md) — same checkpoint, CPU / CUDA path
- [`llama_cpp.md`](./llama_cpp.md) — alternative on-device path (CPU + Metal)
- [`lmstudio.md`](./lmstudio.md) — desktop GUI consumer of the same MLX models
