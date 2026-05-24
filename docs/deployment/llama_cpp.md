# Deploy MiniCPM5-1B with llama.cpp

`llama.cpp` is the recommended path for **CPU / edge / consumer-GPU** deployment. The released GGUF builds run on laptops, single-board computers, Apple Silicon, and Windows boxes with no Python at all.

## Released GGUF artifacts

| File | Size | Use case |
| --- | --- | --- |
| `MiniCPM5-1B-F16.gguf` | 2.1 GB | reference quality, uniform CPU/GPU performance |
| `MiniCPM5-1B-Q8_0.gguf` | 1.1 GB | very small quality drop vs F16, half the disk |
| `MiniCPM5-1B-Q4_K_M.gguf` | 657 MB | edge / mobile-class hardware, minimal VRAM |

These artifacts work directly with vanilla `llama.cpp` and every `llama.cpp`-based runtime (Ollama / LM Studio / `llama-cpp-python`).

## TL;DR — run a release GGUF

```bash
huggingface-cli download openbmb/MiniCPM5-1B-GGUF MiniCPM5-1B-Q4_K_M.gguf --local-dir ./minicpm5

# Interactive chat (auto-applies the chat template)
llama-cli -m ./minicpm5/MiniCPM5-1B-Q4_K_M.gguf -n 2048 --temp 0.7 --top-p 0.95 -ngl 99
```

## OpenAI-compatible server

```bash
llama-server -m MiniCPM5-1B-Q4_K_M.gguf --port 8080 -ngl 99 -c 8192 --jinja

curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniCPM5-1B",
        "messages": [{"role": "user", "content": "1+1=?"}],
        "temperature": 0.7, "top_p": 0.95, "max_tokens": 256
    }'
```

## Generation parameters

| Mode | `--temp` | `--top-p` | When to use |
| --- | --- | --- | --- |
| Think | 0.9 | 0.95 | reasoning, math, code, multi-step |
| No-think | 0.7 | 0.95 | fast assistant, latency-bound |

## Build a GGUF from your own checkpoint

If you've trained your own MiniCPM5-1B variant (continue-pretraining, domain SFT, …) and want to publish a GGUF, the pipeline is:

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

python convert_hf_to_gguf.py "$SRC" --outfile "$OUT/F16.gguf" --outtype f16
build/bin/llama-quantize "$OUT/F16.gguf" "$OUT/Q4_K_M.gguf" Q4_K_M
build/bin/llama-quantize "$OUT/F16.gguf" "$OUT/Q8_0.gguf"   Q8_0
```

## See also

- [`ollama.md`](./ollama.md) — `ollama run` directly from these GGUFs
- [`lmstudio.md`](./lmstudio.md) — desktop GUI for the same GGUFs
