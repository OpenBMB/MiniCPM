# Deploy MiniCPM5-1B and MiniCPM5-2B with llama.cpp

`llama.cpp` is the recommended path for **CPU / edge / consumer-GPU** deployment. The released GGUF builds run on laptops, single-board computers, Apple Silicon, and Windows boxes with no Python at all.

## Released GGUF artifacts

| File | Size | Use case |
| --- | --- | --- |
| `MiniCPM5-2B-F16.gguf` | 5.04 GB | reference quality, uniform CPU/GPU performance |
| `MiniCPM5-2B-Q8_0.gguf` | 2.68 GB | very small quality drop vs F16, half the disk |
| `MiniCPM5-2B-Q4_K_M.gguf` | 1.56 GB | edge / mobile-class hardware, minimal VRAM |
| `MiniCPM5-1B-F16.gguf` | 2.1 GB | reference quality, uniform CPU/GPU performance |
| `MiniCPM5-1B-Q8_0.gguf` | 1.1 GB | very small quality drop vs F16, half the disk |
| `MiniCPM5-1B-Q4_K_M.gguf` | 657 MB | edge / mobile-class hardware, minimal VRAM |

These artifacts work directly with vanilla `llama.cpp` and every `llama.cpp`-based runtime (Ollama / LM Studio / `llama-cpp-python`).

## TL;DR — run a release GGUF

```bash
huggingface-cli download openbmb/MiniCPM5-2B-GGUF MiniCPM5-2B-Q4_K_M.gguf --local-dir ./minicpm5

# Interactive chat (auto-applies the chat template)
llama-cli -m ./minicpm5/MiniCPM5-2B-Q4_K_M.gguf -n 2048 --temp 1.0 --top-p 0.95 --min-p 0.0 -ngl 99
```

## OpenAI-compatible server

```bash
llama-server -m MiniCPM5-2B-Q4_K_M.gguf --port 8080 -ngl 99 -c 8192 --jinja

curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniCPM5-2B",
        "messages": [{"role": "user", "content": "1+1=?"}],
        "temperature": 1.0, "top_p": 0.95, "min_p": 0.0, "max_tokens": 256
    }'
```

In llama.cpp, the default `min_p=0.05` can lead to repetitive output: it filters out tokens whose probability is below 5% of the highest-probability token, potentially discarding the exact tokens needed to break out of a repetition loop. To prevent this, we set `min_p=0.0`.

## Generation parameters

| Mode | `--temp` | `--top-p` | When to use |
| --- | --- | --- | --- |
| MiniCPM5-2B Think | 1.0 | 0.95 | reasoning, math, code, multi-step |
| MiniCPM5-1B Think | 0.9 | 0.95 | reasoning, math, code, multi-step |
| MiniCPM5-1B No-think | 0.7 | 0.95 | fast assistant, latency-bound |

## Tool calling

`llama-server --jinja` parses MiniCPM5's `<function>` / `<param>` output into OpenAI `tool_calls`, so the first tool call works with the released GGUF. Multi-turn tool use needs one template change.

When the client sends an earlier tool call back, the chat template writes each non-string argument with `{{ param_value }}`. Transformers renders that in Python form; llama.cpp's Jinja engine joins a list and drops an object:

| Renderer | `{"keywords": ["blood pressure", "heart rate"], "window": {"unit": "month", "n": 2}}` in the history |
| --- | --- |
| Transformers | `<param name="keywords">['blood pressure', 'heart rate']</param><param name="window">{'unit': 'month', 'n': 2}</param>` |
| llama.cpp (b11630) | `<param name="keywords">blood pressureheart rate</param><param name="window"></param>` |

Serve the template from [`chat_templates/`](./chat_templates/), which writes these arguments as JSON:

```bash
curl -fsSLO https://raw.githubusercontent.com/OpenBMB/MiniCPM/main/docs/deployment/chat_templates/minicpm5_2b_llama_cpp.jinja
llama-server -m MiniCPM5-2B-Q4_K_M.gguf --port 8080 -ngl 99 -c 8192 --jinja \
    --chat-template-file minicpm5_2b_llama_cpp.jinja
```

For MiniCPM5-1B use `minicpm5_1b_llama_cpp.jinja`. Each file is the model's `chat_template.jinja` with only that change: a conversation whose arguments are all strings renders exactly as before, and llama.cpp and Transformers render every conversation identically.

The arguments are written as JSON rather than in Python form because llama.cpp constrains a tool call to its JSON schema: a model that copies `['blood pressure', ...]` from its history cannot write the quote, and the argument comes back empty. The vLLM parser in [`tool_parsers/`](../../tool_parsers/) accepts both forms.

Measured on llama.cpp b11630 with Q4_K_M, 16 seeds per row, at the sampling settings above (temperature 1.0 for 2B, 0.9 for 1B; top-p 0.95; min-p 0). The history holds one call with the arguments above and a tool result asking to search again with twice the window; a second call passes if it keeps both keywords and asks for `{"unit": "month", "n": 4}`.

| History written by | MiniCPM5-2B | MiniCPM5-1B |
| --- | --- | --- |
| the template in the released GGUF | 0 / 16 | 0 / 16 |
| Python form (`['a', 'b']`) | 6 / 16 | not run |
| JSON (`chat_templates/`) | 16 / 16 | 10 / 16 |

With the released GGUF's template, MiniCPM5-2B never made the second call: it kept reasoning about the malformed earlier call until `max_tokens`. MiniCPM5-1B called again in 8 of 16 runs, never with a valid window.

## Build a GGUF from your own checkpoint

If you've trained your own MiniCPM5-1B or MiniCPM5-2B variant (continue-pretraining, domain SFT, …) and want to publish a GGUF, the pipeline is:

```bash
git clone --depth=1 https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir -p build && cd build

# CPU-only build (sufficient for quantize + sanity check)
cmake .. -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j $(nproc) --target llama-quantize llama-cli llama-server

# Or a CUDA build for high-throughput inference
# cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_BUILD_TYPE=Release
# (set CMAKE_CUDA_ARCHITECTURES to your GPU compute capability, see NVIDIA docs)

cd ..
SRC=/path/to/your-MiniCPM5-fp16-hf
OUT=/path/to/output

# Run from the llama.cpp repository root cloned above.
python ./convert_hf_to_gguf.py "$SRC" --outfile "$OUT/F16.gguf" --outtype f16
build/bin/llama-quantize "$OUT/F16.gguf" "$OUT/Q4_K_M.gguf" Q4_K_M
build/bin/llama-quantize "$OUT/F16.gguf" "$OUT/Q8_0.gguf"   Q8_0
```

## See also

- [`ollama.md`](./ollama.md) — `ollama run` directly from these GGUFs
- [`lmstudio.md`](./lmstudio.md) — desktop GUI for the same GGUFs
