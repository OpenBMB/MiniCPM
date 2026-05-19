---
name: minicpm5-deploy-lmstudio
description: Run MiniCPM5-1B in LM Studio (desktop GUI) using either the GGUF runtime (cross-platform) or the MLX runtime (Apple Silicon, faster). Includes OpenAI-compatible local server. Use when the user mentions "LM Studio", desktop GUI inference, "lms" CLI, or wants a no-code chat UI for MiniCPM5.
---

# Deploy MiniCPM5-1B with LM Studio

Desktop GUI + OpenAI-compatible local server. On Apple Silicon ships **two runtimes**:

| Runtime | Format | When to use |
| --- | --- | --- |
| **GGUF** (llama.cpp engine) | F16 / Q8_0 / Q4_K_M | cross-platform, same artifact as Ollama; verified ~89 tok/s Q4_K_M on M4 |
| **MLX** (Apple Silicon only) | bf16 / 4-bit | ~60 % faster, automatic think/answer split via `reasoning_content`; verified ~143 tok/s Q4 on M4 |

## Required input

| Var | Example | Default |
| --- | --- | --- |
| Runtime | `gguf` or `mlx` | `mlx` on Apple Silicon, `gguf` elsewhere |
| `QUANT` | `Q4_K_M` (GGUF) or `4bit` (MLX) | `Q4_K_M` / `4bit` |
| `MODEL_NAME` | `minicpm5-1b` | `minicpm5-1b` |

## Steps

### 1. Install LM Studio + complete onboarding

```bash
brew install --cask lm-studio
open -a "LM Studio"     # accept EULA + pick model source
```

> ⚠️ The first launch MUST be GUI — `lms` (CLI) refuses with `Cannot find LM Studio installation` until LM Studio has run interactively at least once.

### 2A. GGUF runtime path

```bash
mkdir -p ~/.lmstudio/models/openbmb/MiniCPM5-1B-GGUF
huggingface-cli download openbmb/MiniCPM5-1B-GGUF MiniCPM5-1B-${QUANT}.gguf \
    --local-dir ~/.lmstudio/models/openbmb/MiniCPM5-1B-GGUF/

LMS="/Applications/LM Studio.app/Contents/Resources/app/.webpack/lms"
"$LMS" server start                              # binds 127.0.0.1:1234
"$LMS" load minicpm5-1b --gpu max --context-length 8192 -y
"$LMS" ps                                        # verify the model is loaded
```

### 2B. MLX runtime path (Apple Silicon, recommended on Mac)

The MLX runtime needs an MLX-format checkpoint. Either pull a pre-converted one from `openbmb/MiniCPM5-1B-MLX-{bf16,4bit}` (when published) or convert locally — see `minicpm5-deploy-mlx` for the conversion step. Then:

```bash
# Drop the converted MLX directory into LM Studio's model registry
mkdir -p ~/.lmstudio/models/openbmb/MiniCPM5-1B-MLX-${QUANT}
cp -r ./minicpm5-mlx-${QUANT}/* ~/.lmstudio/models/openbmb/MiniCPM5-1B-MLX-${QUANT}/

"$LMS" server start
"$LMS" load minicpm5-1b-mlx-${QUANT} --gpu max -y
```

### 3. Validate

```bash
curl http://127.0.0.1:1234/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "minicpm5-1b",
        "messages": [{"role":"user","content":"1+1=?"}],
        "temperature": 0.7, "top_p": 0.8, "max_tokens": 64
    }'
```

Expected: `"2"` in the reply.

For the MLX runtime, a think prompt produces output split into `message.reasoning_content` (the `<think>` block) and `message.content` (the final answer) automatically — that's an MLX-runtime feature, not a model setting.

## Think vs nothink

LM Studio 0.4.13's chat-completion endpoint does **not** propagate `chat_template_kwargs.enable_thinking` to the GGUF runtime. Instead:

- **Default = think mode** for both runtimes.
- **For nothink** with the GGUF runtime, prepend the closing think block manually:
  ```json
  "messages": [
    {"role":"user","content":"1+1=?"},
    {"role":"assistant","content":"<think>\n\n</think>\n\n"}
  ]
  ```
  and the model continues from there.
- **MLX runtime**: think/answer are auto-split, you don't need to do anything.

## Common pitfalls

- **`error loading model vocabulary: unknown pre-tokenizer type: 'minicpm5'`**: GGUF was not metadata-patched. Use the released GGUF (already patched), or apply the metadata patch from [`docs/deployment/llama_cpp.md`](../../docs/deployment/llama_cpp.md#self-built-gguf-metadata-patch) before importing.
- **MLX runtime not available**: only on Apple Silicon. On Intel Mac / Windows / Linux LM Studio, only the GGUF runtime works.

## When NOT to use

- Just want CLI / scripted runs → `minicpm5-deploy-ollama` is leaner
- Production server → `minicpm5-deploy-vllm`
- No GUI desired → `minicpm5-deploy-llama-cpp` (`llama-server`)

## Reference

[`docs/deployment/lmstudio.md`](../../docs/deployment/lmstudio.md)
