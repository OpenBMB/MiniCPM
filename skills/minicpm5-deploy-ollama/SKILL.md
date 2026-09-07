---
name: minicpm5-deploy-ollama
description: Run MiniCPM5-1B or MiniCPM5-2B via Ollama on macOS / Linux laptop using the released GGUF. Use when the user wants "ollama run", "ollama pull", a Modelfile-driven setup, or one-line laptop deployment.
---

# Deploy MiniCPM5-1B and MiniCPM5-2B with Ollama

One-binary, no-Python laptop deployment. Consumes the released GGUF.

## Required input

| Var | Example | Default |
| --- | --- | --- |
| `GGUF_REPO` | `openbmb/MiniCPM5-2B-GGUF` | required; `openbmb/MiniCPM5-1B-GGUF` also works |
| `QUANT` | `Q4_K_M` (1.56 GB, recommended) / `Q8_0` (2.68 GB) / `F16` (5.04 GB) | `Q4_K_M` |
| `MODEL_NAME` | `minicpm5-2b` | `minicpm5-2b` |

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

huggingface-cli download ${GGUF_REPO} MiniCPM5-2B-${QUANT}.gguf --local-dir .

cat > Modelfile <<EOF
FROM ./MiniCPM5-2B-${QUANT}.gguf

# MiniCPM5 basic chat template
TEMPLATE """{{- if .Messages -}}
{{- range .Messages -}}
<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end -}}
<|im_start|>assistant
{{ end -}}"""

PARAMETER stop "<|im_end|>"
PARAMETER stop "</s>"

# Defaults tuned for think mode
PARAMETER temperature 1.0
PARAMETER top_p 0.95
PARAMETER num_ctx 8192
EOF
```

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
        "model": "minicpm5-2b",
        "messages": [{"role": "user", "content": "1+1=?"}],
        "temperature": 1.0, "top_p": 0.95, "max_tokens": 64
    }'
```

Expected: `"2"` in the reply. 

## MiniCPM5-1B Think vs No-think

A MiniCPM5-1B Modelfile configured with `temperature=0.7, top_p=0.95` defaults to **no-think**. To switch a single conversation to **think** mode, override the sampling params at request time:

```bash
ollama run minicpm5-1b --temperature 0.9 --top-p 0.95
```

Or bake it into a separate model tag by raising the temperature line (top_p stays 0.95):

```Modelfile
PARAMETER temperature 0.9
PARAMETER top_p 0.95
```

Then `ollama create minicpm5-1b-think -f Modelfile.think`.

Ollama 0.24 does **not** directly evaluate the GGUF-embedded Jinja chat template. It maps recognized Jinja templates to built-in Go templates, so MiniCPM5 uses the explicit Go `TEMPLATE` block above. To enter the MiniCPM5-2B think path, use raw mode and prepend `<think>\n` manually:

```bash
curl http://localhost:11434/api/generate -d '{
    "model": "minicpm5-2b",
    "raw": true,
    "prompt": "<|im_start|>user\n鸡兔同笼…<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "options": {"temperature": 1.0, "top_p": 0.95}
}'
```

## Common pitfalls

- **`Error: invalid file magic`**: corrupted download. Re-run `huggingface-cli download`.

## When NOT to use

- Highest throughput on Mac → `minicpm5-deploy-mlx` (Q4 build)
- GUI experience → `minicpm5-deploy-lmstudio`
- NVIDIA GPU production → `minicpm5-deploy-vllm`

## Reference

[`docs/deployment/ollama.md`](../../docs/deployment/ollama.md)
