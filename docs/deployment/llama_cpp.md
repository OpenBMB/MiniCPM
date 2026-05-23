# Deploy MiniCPM5-1B with llama.cpp

`llama.cpp` is the recommended path for **CPU / edge / consumer-GPU** deployment. The released GGUF builds run on laptops, single-board computers, Apple Silicon, and Windows boxes with no Python at all.

## Released GGUF artifacts

| File | Size | Use case |
| --- | --- | --- |
| `MiniCPM5-1B-F16.gguf` | 2.1 GB | reference quality, uniform CPU/GPU performance |
| `MiniCPM5-1B-Q8_0.gguf` | 1.1 GB | very small quality drop vs F16, half the disk |
| `MiniCPM5-1B-Q4_K_M.gguf` | 657 MB | edge / mobile-class hardware, minimal VRAM |

These artifacts already include the metadata patches needed by every `llama.cpp`-based runtime (Ollama / LM Studio / `llama-cpp-python`), so you can use them directly.

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

# One-time metadata fix-up so vanilla llama.cpp / Ollama / LM Studio
# tokenize the GGUF correctly. See "Self-built GGUF metadata patch" below.
```

### Self-built GGUF metadata patch

The released GGUF artifacts on [`openbmb/MiniCPM5-1B-GGUF`](https://huggingface.co/openbmb/MiniCPM5-1B-GGUF) are already metadata-correct; this section only applies if you re-quantize from the fp16 HF checkpoint yourself.

Two GGUF metadata fields need rewriting so unpatched `llama.cpp` consumers tokenize MiniCPM5 correctly:

- `tokenizer.ggml.pre`: `minicpm5` → `llama-bpe` (Llama-3 uses the same GPT-4 BPE regex, so renaming the field is sound)
- 508 special tokens (`<|im_start|>`, `<|im_end|>`, `<s>`, `</s>`, `<tool_call>`, `<|im_sep|>`, `<|fim_*|>`, …): `USER_DEFINED` (3) → `CONTROL` (6) — otherwise chat-template markers get BPE-split into multiple tokens

Minimal in-place patcher (writes `<basename>-fixed.gguf` next to the input):

```bash
pip install gguf

python - "$OUT/Q4_K_M.gguf" <<'PY'
import os, sys, numpy as np, gguf
src = sys.argv[1]
dst = os.path.splitext(src)[0] + "-fixed.gguf"
r = gguf.GGUFReader(src, "r")
w = gguf.GGUFWriter(dst, arch="llama", endianess=gguf.GGUFEndian.LITTLE)
V = gguf.GGUFValueType
for n, f in r.fields.items():
    if n in ("GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"): continue
    if n == "tokenizer.ggml.pre":
        w.add_string(n, "llama-bpe"); continue
    if n == "tokenizer.ggml.token_type":
        a = np.concatenate([np.asarray(f.parts[i]).flatten() for i in f.data]).astype(np.int32)
        a[a == 3] = 6
        w.add_array(n, a.tolist()); continue
    t = f.types[0]
    if t == V.STRING:
        w.add_string(n, bytes(f.parts[f.data[0]]).decode("utf-8", "replace"))
    elif t == V.ARRAY:
        inner = f.types[1] if len(f.types) > 1 else None
        if inner == V.STRING:
            w.add_array(n, [bytes(f.parts[i]).decode("utf-8", "replace") for i in f.data])
        else:
            a = np.concatenate([np.asarray(f.parts[i]).flatten() for i in f.data])
            w.add_array(n, a.tolist())
    else:
        v = f.parts[f.data[0]][0]
        {V.BOOL: w.add_bool, V.UINT8: w.add_uint8, V.INT32: w.add_int32,
         V.UINT32: w.add_uint32, V.UINT64: w.add_uint64,
         V.FLOAT32: w.add_float32}[t](n, type(v).__call__(v))
for t in r.tensors:
    w.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print(f"wrote {dst}")
PY
```

## See also

- [`ollama.md`](./ollama.md) — `ollama run` directly from these GGUFs
- [`lmstudio.md`](./lmstudio.md) — desktop GUI for the same GGUFs
