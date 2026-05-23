---
name: minicpm5-deploy-ollama
description: Run MiniCPM5-1B via Ollama on macOS / Linux laptop. Uses the released GGUF (already metadata-patched). Use when the user wants "ollama run", "ollama pull", a Modelfile-driven setup, or one-line laptop deployment.
---

# Deploy MiniCPM5-1B with Ollama

One-binary, no-Python laptop deployment. Consumes the released GGUF.

## Required input

| Var | Example | Default |
| --- | --- | --- |
| `GGUF_REPO` | `openbmb/MiniCPM5-1B-GGUF` | required |
| `QUANT` | `Q4_K_M` (657 MB, recommended) / `Q8_0` / `F16` | `Q4_K_M` |
| `MODEL_NAME` | `minicpm5-1b` | `minicpm5-1b` |

## Steps

### 1. Install Ollama (once)

```bash
brew install ollama                                # macOS
# or:
curl -fsSL https://ollama.com/install.sh | sh      # Linux

OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve &
```

### 2. Download the GGUF + write Modelfile

```bash
mkdir -p ~/${MODEL_NAME} && cd ~/${MODEL_NAME}

huggingface-cli download ${GGUF_REPO} MiniCPM5-1B-${QUANT}.gguf --local-dir .

cat > Modelfile <<'EOF'
FROM ./MiniCPM5-1B-Q4_K_M.gguf

# MiniCPM5 chat template (matches release tokenizer)
TEMPLATE """{{- if .Messages -}}
{{- range .Messages -}}
<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end -}}
<|im_start|>assistant
{{ end -}}"""

PARAMETER stop "<|im_end|>"
PARAMETER stop "</s>"

# Defaults tuned for nothink mode
PARAMETER temperature 0.7
PARAMETER top_p 0.95
PARAMETER num_ctx 8192
EOF
```

> ⚠️ The `FROM ./MiniCPM5-1B-Q4_K_M.gguf` line is hard-coded to Q4_K_M; change the filename if you used a different `QUANT`.

### 3. Create + run

```bash
ollama create ${MODEL_NAME} -f Modelfile
ollama run ${MODEL_NAME}
```

### 4. Validate via OpenAI-compatible API

```bash
curl http://localhost:11434/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "minicpm5-1b",
        "messages": [{"role": "user", "content": "1+1=?"}],
        "temperature": 0.7, "top_p": 0.95, "max_tokens": 64
    }'
```

Expected: `"2"` in the reply. 

## Think mode

Default Modelfile is nothink. For think:

```bash
ollama run ${MODEL_NAME} --temperature 0.9 --top-p 0.95
```

Or bake it into a separate model tag by flipping `temperature 0.7` to `temperature 0.9` (top_p stays 0.95) and `ollama create ${MODEL_NAME}-think -f Modelfile.think`.

To force the auto-injected `<think>\n` prefix, use raw mode:

```bash
curl http://localhost:11434/api/generate -d '{
    "model": "minicpm5-1b",
    "raw": true,
    "prompt": "<|im_start|>user\n鸡兔同笼…<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "options": {"temperature": 0.9, "top_p": 0.95}
}'
```

## Common pitfalls

- **Output is `****\n****\n…` garbage**: the GGUF was built from a non-released checkpoint and wasn't metadata-patched. The released artifacts already have the fix; if you built your own, apply the GGUF metadata patch from [`docs/deployment/llama_cpp.md`](../../docs/deployment/llama_cpp.md#self-built-gguf-metadata-patch) before `ollama create`.
- **`Error: invalid file magic`**: corrupted download. Re-run `huggingface-cli download`.

## When NOT to use

- Highest throughput on Mac → `minicpm5-deploy-mlx` (Q4 build)
- GUI experience → `minicpm5-deploy-lmstudio`
- NVIDIA GPU production → `minicpm5-deploy-vllm`

## Reference

[`docs/deployment/ollama.md`](../../docs/deployment/ollama.md)
