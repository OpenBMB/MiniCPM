# Deploy MiniCPM5-1B and MiniCPM5-2B with Ollama

[Ollama](https://ollama.com) is the easiest CLI / daemon path to run MiniCPM5-1B or MiniCPM5-2B on a laptop or desktop — one binary, no Python, no CUDA toolkit. It consumes the same GGUF files we ship for `llama.cpp`.

## TL;DR

```bash
brew install ollama                 # macOS
# or: curl -fsSL https://ollama.com/install.sh | sh   (Linux)

OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve &

# Pull the released GGUF and create the Ollama model
mkdir -p ~/minicpm5-2b && cd ~/minicpm5-2b
huggingface-cli download openbmb/MiniCPM5-2B-GGUF MiniCPM5-2B-Q4_K_M.gguf --local-dir .

QUANT=Q4_K_M
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

# Stop on either EOS or the chat-template terminator
PARAMETER stop "<|im_end|>"
PARAMETER stop "</s>"

# Defaults are tuned for think mode
PARAMETER temperature 1.0
PARAMETER top_p 0.95
PARAMETER num_ctx 8192
EOF

ollama create minicpm5-2b -f Modelfile
ollama run minicpm5-2b
```

## Choosing a quant

| Quant | Disk | Quality | Ollama tag (suggested) |
| --- | --- | --- | --- |
| F16 | 5.04 GB | reference | `:fp16` |
| Q8_0 | 2.68 GB | ~indistinguishable from F16 | `:q8` |
| **Q4_K_M** | **1.56 GB** | small drop, ideal for laptops | **`:q4_k_m`** *(default)* |

## API access

Ollama serves an OpenAI-compatible REST endpoint on `http://localhost:11434/v1`:

```bash
curl http://localhost:11434/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "minicpm5-2b",
        "messages": [{"role": "user", "content": "用一句话解释 GQA。"}],
        "temperature": 1.0, "top_p": 0.95, "max_tokens": 1024
    }'
```

Or use the Ollama-native API:

```bash
curl http://localhost:11434/api/chat -d '{
    "model": "minicpm5-2b",
    "messages": [{"role":"user","content":"1+1=?"}],
    "stream": false,
    "options": {"temperature": 1.0, "top_p": 0.95}
}'
```

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

> ℹ️ Ollama 0.24 does **not** directly evaluate the GGUF-embedded Jinja chat template. It maps recognized Jinja templates to built-in Go templates, so MiniCPM5 uses the explicit Go `TEMPLATE` block above. To enter the MiniCPM5-2B think path, use raw mode and prepend `<think>\n` manually:
>
> ```bash
> curl http://localhost:11434/api/generate -d '{
>     "model": "minicpm5-2b",
>     "raw": true,
>     "prompt": "<|im_start|>user\n鸡兔同笼…<|im_end|>\n<|im_start|>assistant\n<think>\n",
>     "options": {"temperature": 1.0, "top_p": 0.95}
> }'
> ```

## Higher throughput on Apple Silicon

For long-context workloads, raise `num_ctx`:

```Modelfile
PARAMETER num_ctx 32768
```

For sustained throughput with larger KV caches, set `OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0` in the environment before `ollama serve`.

## See also

- [`llama_cpp.md`](./llama_cpp.md) — the engine behind Ollama; full GGUF build pipeline
- [`lmstudio.md`](./lmstudio.md) — desktop GUI consumer of the same GGUFs
- [`mlx.md`](./mlx.md) — alternative on-device path on Apple Silicon (faster for Q4)
