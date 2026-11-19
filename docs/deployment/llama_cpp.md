# Deploy MiniCPM5-1B and MiniCPM5-2B with llama.cpp

`llama.cpp` is the recommended path for **CPU / edge / consumer-GPU** deployment. The released GGUF builds run on laptops, single-board computers, Apple Silicon, and Windows boxes with no Python at all.

For MiniCPM5-15B-A2.5B, see [MiniCPM5-15B-A2.5B](#minicpm5-15b-a25b).

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

## MiniCPM5-15B-A2.5B

### Build a compatible runtime

Use the [pinned llama.cpp fork revision](https://github.com/cyxu0401/llama.cpp/commit/21d8648528f1e492b227686b945b0af79c741e38) with support for the `minicpm5_moe` GGUF architecture. The following builds `llama-server` on Linux with the CUDA toolkit. For a CPU-only build, omit `-DGGML_CUDA=ON`.

```bash
git init llama.cpp-minicpm5
cd llama.cpp-minicpm5
git fetch --depth 1 https://github.com/cyxu0401/llama.cpp.git 21d8648528f1e492b227686b945b0af79c741e38
git checkout --detach FETCH_HEAD
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build --config Release --target llama-server -j
```

### Start a GGUF server

Set `GGUF_PATH` to the downloaded MiniCPM5-15B-A2.5B GGUF file. This example uses Q4_K_M, one slot, a 2,048-token context, and F16 K/V cache on an NVIDIA GPU.

```bash
LLAMA_SERVER="$PWD/build/bin/llama-server"
GGUF_PATH="/path/to/MiniCPM5-15B-A2.5B-Q4_K_M.gguf"
"$LLAMA_SERVER" -m "$GGUF_PATH" --alias MiniCPM5-15B-A2.5B \
    -ngl 99 -c 2048 -np 1 -b 256 -ub 128 \
    -fa on -ctk f16 -ctv f16 --jinja \
    --host 127.0.0.1 --port 8080
```

For a CPU-only smoke test, set `-ngl 0 -fa off` and adjust the context to fit available memory.

### Chat completion

```bash
curl -fsS http://127.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniCPM5-15B-A2.5B",
        "messages": [{"role": "user", "content": "Explain grouped-query attention in one sentence."}],
        "temperature": 0,
        "max_tokens": 128,
        "min_p": 0.0
    }'
```

### Speculative decoding with DSpark

Set `DRAFT_GGUF_PATH` to the local GGUF file for the matching DSpark draft. Stop the baseline server and run the DSpark command in the same terminal.

```bash
DRAFT_GGUF_PATH="/path/to/MiniCPM5-15B-A2.5B-DSpark.gguf"
"$LLAMA_SERVER" -m "$GGUF_PATH" --alias MiniCPM5-15B-A2.5B \
    -ngl 99 -c 2048 -np 1 -b 256 -ub 128 \
    -fa on -ctk f16 -ctv f16 --jinja \
    --spec-type draft-dspark --spec-draft-model "$DRAFT_GGUF_PATH" \
    --spec-draft-n-max 2 --n-gpu-layers-draft 99 \
    --host 127.0.0.1 --port 8080
```

Reuse the chat request and check the server log for proposed and accepted draft tokens. On one H800, BF16, Q8_0, and Q4_K_M GGUFs passed short ordinary and DSpark generation checks. Some greedy sequences differed between the two modes. [llama.cpp validation notes](https://github.com/ggml-org/llama.cpp/pull/32386)
