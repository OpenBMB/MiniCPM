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

Build the preview from the team source at [llama.cpp PR #32644](https://github.com/ggml-org/llama.cpp/pull/32644). This revision supports the `minicpm5_moe` GGUF architecture and the HF-to-GGUF target exporter.

### Build llama-server

The following builds the CUDA runtime on Linux. For CPU only, omit `-DGGML_CUDA=ON`.

```bash
git init llama.cpp-minicpm5
cd llama.cpp-minicpm5
git fetch --depth 1 https://github.com/tc-mb/llama.cpp.git c2dd69cf449ce7ae87ffda68663a06359a4b04b0
git checkout --detach FETCH_HEAD
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build --config Release --target llama-server llama-quantize -j
```

Use a BF16, Q8_0 or Q4_K_M GGUF for the target. To export the BF16 target from a local HF checkpoint and optionally produce a quantized GGUF, follow the [conversion instructions](https://github.com/tc-mb/llama.cpp/blob/c2dd69cf449ce7ae87ffda68663a06359a4b04b0/docs/models.md#minicpm5-moe). Obtain the matching DSpark draft GGUF separately.

### Start the server

Set `GGUF_PATH` to the target GGUF. This example uses one slot with an 8,192-token context for the prompt and completion combined.

```bash
LLAMA_SERVER="$PWD/build/bin/llama-server"
GGUF_PATH="/path/to/MiniCPM5-15B-A2.5B-Q4_K_M.gguf"
"$LLAMA_SERVER" -m "$GGUF_PATH" \
    --alias MiniCPM5-15B-A2.5B --min-p 0 -ngl 99 -fa on \
    -c 8192 -np 1 --jinja -ctk f16 -ctv f16 \
    --host 127.0.0.1 --port 8080
```

For CPU-only inference, use `-ngl 0 -fa off`. Choose the context size based on available memory.

### Chat completion

```bash
curl -fsS http://127.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniCPM5-15B-A2.5B",
        "messages": [{"role": "user", "content": "Explain grouped-query attention in one sentence."}],
        "temperature": 0,
        "max_tokens": 128
    }'
```

### Speculative decoding with DSpark

Stop the baseline server and use a compatible draft GGUF in the same terminal.

```bash
DRAFT_GGUF_PATH="/path/to/MiniCPM5-15B-A2.5B-DSpark.gguf"
"$LLAMA_SERVER" -m "$GGUF_PATH" --spec-draft-model "$DRAFT_GGUF_PATH" \
    --spec-type draft-dspark --spec-draft-n-max 2 --spec-draft-p-min 0 \
    --n-gpu-layers-draft 99 -ctkd f16 -ctvd f16 \
    --alias MiniCPM5-15B-A2.5B --min-p 0 -ngl 99 -fa on \
    -c 8192 -np 1 --jinja -ctk f16 -ctv f16 \
    --host 127.0.0.1 --port 8080
```

- `--min-p 0` disables the target sampler's Min-P filter. `--spec-draft-p-min 0` disables early draft truncation by confidence. The target still verifies each proposal.
- `-ctk` and `-ctv` set target K/V types. `-ctkd` and `-ctvd` set draft K/V types. Both caches use F16 regardless of the target weight format.
- The short CPU generation tests used a 2,048-token context. They covered pre-existing BF16, Q8_0, and Q4_K_M GGUFs with DSpark off and on, plus ordinary generation with a newly exported BF16 target. For DSpark, check the server log for drafted and accepted tokens. [llama.cpp startup reference](https://github.com/tc-mb/llama.cpp/blob/c2dd69cf449ce7ae87ffda68663a06359a4b04b0/docs/speculative.md#minicpm5-moe-target)
