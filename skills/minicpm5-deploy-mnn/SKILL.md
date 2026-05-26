---
name: minicpm5-deploy-mnn
description: Run MiniCPM5-1B with Alibaba's MNN engine (mnn-llm) for mobile on-device deployment — Android (CPU / OpenCL / QNN) and iOS (CPU / Metal). Use when the user mentions "MNN", "mnn-llm", "llm_demo", "llmexport", "Android 部署", "iOS 部署", or wants to ship the model inside a mobile app.
---

# Deploy MiniCPM5-1B with MNN

Mobile on-device deployment via [mnn-llm](https://github.com/alibaba/MNN/tree/master/transformers/llm). The same `.mnn` artifact runs on Android (CPU / OpenCL / QNN) and iOS (CPU / Metal); Linux / macOS / Windows builds let you develop on a laptop before shipping to a phone.

## Required input

| Var | Example | Default |
| --- | --- | --- |
| `MNN_REPO` | HF `taobao-mnn/MiniCPM5-1B-MNN` / MS `MNN/MiniCPM5-1B-MNN` | required |
| `BACKEND` | `cpu` / `opencl` (Android GPU) / `metal` (iOS GPU) | `cpu` |
| `THREADS` | `4` for CPU; **must be `68` for OpenCL** | per backend |
| `PRECISION` | `low` (fp16, recommended) / `high` (fp32) | `low` |

## Steps

### 1. Download the pre-converted MNN model

```bash
# Hugging Face
huggingface-cli download taobao-mnn/MiniCPM5-1B-MNN --local-dir ./minicpm5-mnn

# OR ModelScope (China mirror)
modelscope download --model MNN/MiniCPM5-1B-MNN --local_dir ./minicpm5-mnn
```

The repo currently ships a single **block-wise int4** build (~626 MB total). Layout:

```
config.json  llm_config.json  llm.mnn  llm.mnn.weight  embeddings_int4.bin  tokenizer.mtok
```

For int8 / fp16, build your own with `llmexport` (see the advanced section).

### 2. Build `mnn-llm`

```bash
git clone --depth=1 https://github.com/alibaba/MNN.git
cd MNN && mkdir build && cd build

cmake .. -DMNN_BUILD_LLM=true        # add -DMNN_AVX512=true on x86 Linux/Mac
make -j16
```

For Android use the bundled script instead of plain cmake:

```bash
cd project/android && mkdir build_64 && cd build_64
../build_64.sh -DMNN_BUILD_LLM=true -DMNN_OPENCL=true -DMNN_USE_LOGCAT=true
```

For iOS:

```bash
sh package_scripts/ios/buildiOS.sh -DMNN_BUILD_LLM=true
```

The build produces two CLI binaries: `llm_demo` (chat) and `llm_bench` (benchmark).

### 3a. Interactive chat

```bash
./llm_demo ../../minicpm5-mnn/config.json
```

No prompt arg = interactive REPL. Passing a path to a prompt file batches one reply per line:

```bash
./llm_demo ../../minicpm5-mnn/config.json prompts.txt
```

`llm_demo` reads sampling / backend / memory settings from `config.json` — there are no flags to override them per invocation.

### 3b. Throughput benchmark

```bash
./llm_bench -m ../../minicpm5-mnn/config.json -a cpu -t 4 -p 32,64 -n 32 -rep 3
```

Reports prefill + decode tok/s; useful when sweeping backends / thread counts.

### 4. Validate

```bash
echo "1+1=?" > /tmp/prompt.txt
./llm_demo ../../minicpm5-mnn/config.json /tmp/prompt.txt
```

Expected: the reply contains `2`.

## `config.json` — edit, don't pass flags

`llm_demo` has no CLI flags for sampling or backend — edit `config.json` and re-run.

```json
{
    "backend_type": "cpu",
    "thread_num": 4,
    "precision": "low",
    "memory": "low",

    "max_new_tokens": 512,
    "reuse_kv": true,
    "use_mmap": false,

    "sampler_type": "mixed",
    "mixed_samplers": ["topK", "tfs", "typical", "topP", "min_p", "temperature"],
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.95
}
```

| Key | Meaning |
| --- | --- |
| `backend_type` | `cpu` / `opencl` (Android GPU) / `metal` (iOS GPU). Binary must be built with the matching flag. |
| `thread_num` | CPU thread count. **For OpenCL set to `68`** — it is a buffer/tuning mode flag, not a thread count. |
| `precision` | `low` = fp16 (recommended), `high` = fp32. |
| `memory` | `low` activates runtime activation quantisation. |
| `reuse_kv` | reuse KV cache across multi-turn dialog (recommended for chat). |
| `use_mmap` | mmap weights from disk — set `true` on phones to avoid OOM. |
| `sampler_type` | `mixed` is the default; runs `mixed_samplers` in order. See the [MNN LLM docs](https://github.com/alibaba/MNN/blob/master/docs/transformers/llm.md) for the full sampler list. |

## Sampling defaults (MiniCPM5-1B recommended)

| Mode | `temperature` | `top_p` |
| --- | --- | --- |
| Think | 0.9 | 0.95 |
| No-think | 0.7 | 0.95 |

## Mobile deployment

Drop the model directory into one of the bundled apps:

- **Android**: [`apps/Android/MnnLlmChat`](https://github.com/alibaba/MNN/tree/master/apps/Android/MnnLlmChat) — set `backend_type` to `"opencl"`.
- **iOS**: [`apps/iOS/MNNLLMChat`](https://github.com/alibaba/MNN/tree/master/apps/iOS/MNNLLMChat) — set `backend_type` to `"metal"`.

Both apps expect the same `config.json` + `llm.mnn` + `llm.mnn.weight` + `tokenizer.mtok` layout as the desktop CLI.

## Common pitfalls

- **OpenCL backend silently doesn't kick in**: two things must line up — the binary must be built with `-DMNN_OPENCL=true`, AND `config.json` must have `"thread_num": 68` (for OpenCL this is a tuning-mode flag, not a thread count). Default `4` will fall back to CPU-like behaviour.
- **OOM on phone**: set `"use_mmap": true` and `"memory": "low"`. Drop `max_new_tokens` if the KV cache is the dominant cost.
- **First OpenCL run is slow**: the OpenCL backend tunes kernels on first launch and caches the result. Run a second time to see steady-state numbers — this is also how the official `llm_bench` is measured.
- **No `llm_server`**: `mnn-llm` does NOT ship an OpenAI-compatible HTTP server. If you need that, wrap `llm_demo` yourself or pick another backend (`minicpm5-deploy-llama-cpp` ships `llama-server`).

## Building MNN weights from your own checkpoint (advanced)

Use `llmexport` only if you have a self-trained HF fp16 checkpoint, OR if you want a precision other than int4:

```bash
git clone https://github.com/alibaba/MNN.git
cd MNN/transformers/llm/export
pip install -r requirements.txt
pip install MNN     # provides pymnn / mnnconvert needed by llmexport

HF=/path/to/your-MiniCPM5-fp16-hf
OUT=/path/to/output

# int4 (mobile default) — --hqq improves quality at no inference cost
python llmexport.py --path "$HF" --export mnn --quant_bit 4 --quant_block 128 \
    --hqq --dst_path "$OUT/MiniCPM5-1B-MNN-int4"

# int8 (desktop / server)
python llmexport.py --path "$HF" --export mnn --quant_bit 8 --quant_block 128 \
    --dst_path "$OUT/MiniCPM5-1B-MNN-int8"
```

Approximate output sizes for MiniCPM5-1B: int4 ~597 MB, int8 ~1.1 GB, fp16 ~2.1 GB (fp16 requires the two-step ONNX → MNNConvert path documented in the MNN LLM docs).

The output directory uses the same layout as the released artifact, so steps 3a–3b work unchanged.

## When NOT to use

- NVIDIA datacentre serving with high QPS → `minicpm5-deploy-vllm`
- Apple Silicon native, Python-only → `minicpm5-deploy-mlx` (mlx-lm is more Pythonic)
- GGUF tooling already in place / need an OpenAI-compatible HTTP server → `minicpm5-deploy-llama-cpp`
- Desktop GUI → `minicpm5-deploy-lmstudio`

## Reference

- [`docs/deployment/mnn.md`](../../docs/deployment/mnn.md) — paired cookbook
- [MNN LLM module docs](https://github.com/alibaba/MNN/blob/master/docs/transformers/llm.md) — full config reference (sampler types, attention modes, NPU export)
- [MNN repository](https://github.com/alibaba/MNN)
- [`llmexport`](https://github.com/alibaba/MNN/tree/master/transformers/llm/export) — HF → MNN converter
