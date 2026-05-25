---
name: minicpm5-deploy
description: Pick the right inference backend for a MiniCPM5-1B checkpoint and route to a backend-specific cookbook skill. Use when the user wants to deploy / serve / chat-with / benchmark a MiniCPM5 model and has not yet committed to a specific engine, or when they say "deploy MiniCPM5", "run MiniCPM5", "serve MiniCPM5", "MiniCPM5 推理", "部署 MiniCPM5".
---

# Deploy MiniCPM5-1B — backend router

You're being asked to deploy / serve / chat-with a MiniCPM5-1B checkpoint. Your job is to **pick exactly one backend skill below** based on the user's hardware, format, and goal, then **invoke that skill** rather than improvising.

## 1. Required input from the user

Before picking a backend, you MUST know:

| Variable | Example | Where to ask |
| --- | --- | --- |
| `MODEL_PATH` | HF id `openbmb/MiniCPM5-1B` (post-release) **or** a local path | "Which checkpoint? HF id or local path?" |
| Hardware | NVIDIA GPU / Apple Silicon / CPU only | infer from context, otherwise ask |
| Goal | "interactive chat" / "OpenAI server" / "Python script" / "benchmark" | infer from context |

### Available checkpoints on Hugging Face

| Variant | HF repo | Use with |
| --- | --- | --- |
| **HF fp16 (recommended)** | [`openbmb/MiniCPM5-1B`](https://huggingface.co/openbmb/MiniCPM5-1B) | `transformers` / `vllm` (no `--quantization`) / `sglang` / any `minicpm5-finetune-*` |
| GGUF F16 / Q8_0 / Q4_K_M | [`openbmb/MiniCPM5-1B-GGUF`](https://huggingface.co/openbmb/MiniCPM5-1B-GGUF) | `minicpm5-deploy-llama-cpp` / `-ollama` / `-lmstudio` |
| MLX (Apple Silicon) | [`openbmb/MiniCPM5-1B-MLX`](https://huggingface.co/openbmb/MiniCPM5-1B-MLX) | `minicpm5-deploy-mlx` |

If the user has a local copy, accept any directory path that contains `config.json` and `model.safetensors` (or the equivalent GGUF / MLX layout).

## 2. Decision matrix — pick exactly one

| User says / wants | Hardware | Format | → Skill to invoke |
| --- | --- | --- | --- |
| "Quick Python script" / "one-shot generation" / "no server" | any GPU or CPU | HF safetensors | **`minicpm5-deploy-transformers`** |
| "OpenAI server" / "production serving" / "high QPS" | NVIDIA GPU | HF safetensors | **`minicpm5-deploy-vllm`** |
| "RadixAttention" / "prefix cache" / "batched eval" | NVIDIA GPU | HF safetensors | **`minicpm5-deploy-sglang`** |
| "GGUF" / "llama.cpp" / "llama-cli" / "CPU only" | any CPU + optional GPU | GGUF | **`minicpm5-deploy-llama-cpp`** |
| "Ollama" / "ollama run" / "Modelfile" | macOS / Linux laptop | GGUF | **`minicpm5-deploy-ollama`** |
| "LM Studio" / "desktop GUI" | macOS / Windows / Linux | GGUF or MLX | **`minicpm5-deploy-lmstudio`** |
| "MLX" / "Apple Silicon native" / "fastest on Mac" | Apple Silicon | MLX | **`minicpm5-deploy-mlx`** |

If the user **has not specified** any of the above and asks "how do I run this?":

- **CUDA box, want fastest server**: pick `minicpm5-deploy-vllm`.
- **CUDA box, want minimal Python**: pick `minicpm5-deploy-transformers`.
- **Apple Silicon laptop**: pick `minicpm5-deploy-ollama` (easiest) or `minicpm5-deploy-mlx` (fastest).
- **CPU only / Windows / low-VRAM**: pick `minicpm5-deploy-llama-cpp` (Q4_K_M).

## 3. Invocation contract

Once you've picked a backend skill, **invoke that skill with `MODEL_PATH` set**. Do NOT inline the backend's commands here — each backend has its own pitfalls (mandatory flags, env-var pins, install order) that the dedicated skill handles. The user explicitly does NOT want you to "improvise" — read the picked sub-skill in full first.

## 4. Sanity check after deploy

Whichever backend you pick, after launch run this universal sanity check:

```bash
# Replace localhost:PORT with the backend's actual port (default in each sub-skill)
curl http://localhost:PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniCPM5-1B",
        "messages": [{"role":"user","content":"1+1=?"}],
        "temperature": 0.7, "top_p": 0.95, "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": false}
    }'
```

Expected: HTTP 200 with `choices[0].message.content` containing `"2"`.

If you get `<think>...` instead, the request hit think mode — set `enable_thinking: false` in `chat_template_kwargs`.

## 5. Known cross-backend pitfalls

These are common to multiple backends — surface to the user up front:

- **Think vs no-think**: defaults are think mode (`temperature=0.9, top_p=0.95`). For nothink (faster, less verbose) use `enable_thinking=false` + `temperature=0.7, top_p=0.95`.
- **128 K context**: `max_position_embeddings=131072`, `rope_theta=5e6`, **no rope-scaling**. Pass `--max-model-len 131072` (vLLM) / `--context-length 131072` (SGLang) / `-c 131072` (llama.cpp) to use the full window. Lower if VRAM is tight.
- **Untied lm_head**: `tie_word_embeddings=false`. Tools that assume the Llama tied default (e.g. `mlx_lm.convert` < 0.31) will silently drop `lm_head` → output collapses to random tokens. The MLX skill bakes in the fix.

## 6. Don't reinvent: link to the cookbook

Each sub-skill is paired with a one-page cookbook in [`docs/deployment/`](../../docs/deployment/). The skill is the machine-readable shortcut; the cookbook is the human-readable reference. Both are kept in sync.
