# Deploy MiniCPM5-1B with MLX (Apple Silicon)

[MLX](https://github.com/ml-explore/mlx) is Apple's on-device tensor framework. For MiniCPM5-1B it is the recommended path on Apple Silicon (M1–M4) when you want **highest throughput** and want to stay inside one Python process — no separate server, no `llama.cpp` build chain.

## TL;DR

```bash
pip install mlx-lm gguf

# 1. Apply the two metadata fixes from "Required HF-side patch" below, writing
#    the patched directory at /path/to/hf-fp16-fixed.

# 2. Convert (bf16 and / or 4-bit)
HF=/path/to/hf-fp16-fixed
mlx_lm.convert --hf-path "$HF" --mlx-path ./minicpm5-mlx-bf16
mlx_lm.convert --hf-path "$HF" --mlx-path ./minicpm5-mlx-q4 -q --q-bits 4

# 3. Generate
mlx_lm.generate --model ./minicpm5-mlx-q4 \
    --prompt "<|im_start|>user
1+1=?<|im_end|>
<|im_start|>assistant
" \
    --max-tokens 200 --temp 0.7 --top-p 0.95 \
    --extra-eos-token "<|im_end|>"
```

## Required HF-side patch

The released `hf-fp16/` checkpoint needs **two metadata fixes** before `mlx_lm.convert` will work — without them the converted model loads but emits raw byte-level BPE labels (`ĠareĠgivenĠ"1+1=?"...`) instead of decoded text.

### 1. `config.json` — add `tie_word_embeddings: false`

MiniCPM5-1B has an **independent `lm_head`** (see model card). The released `config.json` does not include `tie_word_embeddings`, so MLX falls back to the Llama default (which historically tied) and **drops `lm_head` during conversion**, then transparently aliases it to the embedding matrix at load time. The embedding and `lm_head` are not numerically equivalent, so generations look like uniform-random tokens.

```python
import json
c = json.load(open("hf-fp16/config.json"))
c["tie_word_embeddings"] = False         # ← required
json.dump(c, open("hf-fp16/config.json", "w"), indent=2)
```

After the fix, `mlx_lm.convert` produces 219 weights (vs. 218 without the fix); confirm with:

```bash
python -c "import json; m=json.load(open('mlx-bf16/model.safetensors.index.json')); \
    print('lm_head present:', any('lm_head' in k for k in m['weight_map']))"
```

### 2. `tokenizer_config.json` — switch class to `PreTrainedTokenizerFast`

The released `tokenizer_config.json` advertises:

```json
"tokenizer_class": "LlamaTokenizerFast",
"legacy": true,
"sp_model_kwargs": {},
```

…but the actual `tokenizer.json` is **byte-level BPE** (GPT-2 / Llama-3 family — vocab tokens carry the `Ġ` space and `Ċ` newline prefixes). Loading it under `LlamaTokenizerFast` triggers the SentencePiece decode path, which does not reverse the byte-level mapping; every space is rendered as `Ġ` and every newline as `Ċ`.

```python
import json
tc = json.load(open("hf-fp16/tokenizer_config.json"))
tc["tokenizer_class"] = "PreTrainedTokenizerFast"
for k in ("legacy", "sp_model_kwargs", "use_default_system_prompt",
         "add_prefix_space", "spaces_between_special_tokens"):
    tc.pop(k, None)
json.dump(tc, open("hf-fp16/tokenizer_config.json", "w"),
          indent=2, ensure_ascii=False)
```

Sanity check after the patch:

```python
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained("hf-fp16-fixed")
ids = tk.encode("We are given a problem.", add_special_tokens=False)
assert tk.decode(ids) == "We are given a problem.", "byte-level decode still broken"
```

## Conversion

```bash
HF=./MiniCPM5-1B-hf-fixed   # the patched copy from above

# bf16 master copy
mlx_lm.convert --hf-path "$HF" --mlx-path ./minicpm5-mlx-bf16

# 4-bit Q4, 4.5 bits/weight on average
mlx_lm.convert --hf-path "$HF" --mlx-path ./minicpm5-mlx-q4 -q --q-bits 4
```

The 4-bit pass logs `[INFO] Quantized model with 4.501 bits per weight.` — the slight overshoot above 4 bits is from keeping the `embed_tokens` and `lm_head` in higher precision, which preserves quality on the small, untied vocabulary head.

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

> 💡 `--extra-eos-token "<|im_end|>"` (CLI) or adding `<|im_end|>` to the wrapper's stop list (Python) is required: the GGUF / HF metadata only register `</s>` (id 1) and the secondary EOS id 130073 as model EOS, but the chat template ends turns with the `<|im_end|>` *string*. Without an extra EOS the model will keep generating into the next role's prompt.

## Recommended sampling

| Mode | `--temp` | `--top-p` | When to use |
| --- | --- | --- | --- |
| Think | 0.9 | 0.95 | reasoning, math, code, multi-step (model auto-emits `<think>` block) |
| No-think | 0.7 | 0.95 | fast assistant, latency-bound |

Both modes are activated by sampling parameters only — the GGUF-baked Jinja chat template auto-injects `<think>\n` when no `system` message disables it, so you get think-mode behaviour by default.

## Q&A

### `mlx_lm.generate` prints `Ġinland===éķ¿` style garbage

You skipped step 2 of the HF patch — the byte-level BPE decoder is not being applied. Re-patch `tokenizer_config.json`, re-run `mlx_lm.convert`, and re-test.

### Generated tokens look *random* (not byte-level garbage, just nonsense)

You skipped step 1 of the HF patch — `lm_head` was dropped during conversion. Verify with the `lm_head present` check above; if it prints `False`, fix `tie_word_embeddings` in `config.json` and re-convert.

### Model never stops generating

Add `--extra-eos-token "<|im_end|>"` (CLI) or include `<|im_end|>` in `stop` (Python). See note above.

## See also

- [`transformers.md`](./transformers.md) — same checkpoint, CPU / CUDA path
- [`llama_cpp.md`](./llama_cpp.md) — alternative on-device path (CPU + Metal)
- [`lmstudio.md`](./lmstudio.md) — desktop GUI consumer of the same MLX models
