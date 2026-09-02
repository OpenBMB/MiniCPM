# Deploy MiniCPM5-1B with MNN

[MNN](https://github.com/alibaba/MNN) is Alibaba's lightweight deep learning inference engine. Its LLM runtime [`mnn-llm`](https://github.com/alibaba/MNN/tree/master/transformers/llm) targets **on-device deployment**, with first-class support for Android (CPU / OpenCL / QNN) and iOS (CPU / Metal). The same `.mnn` artifact also runs on Linux / macOS / Windows so you can develop on a laptop before shipping to a phone.

For MiniCPM5-1B, MNN is the recommended path when you need to ship the model **inside a mobile app**.

## Released MNN artifact

Pre-converted weights are published on Hugging Face at [`taobao-mnn/MiniCPM5-1B-MNN`](https://huggingface.co/taobao-mnn/MiniCPM5-1B-MNN) and mirrored on ModelScope at [`MNN/MiniCPM5-1B-MNN`](https://www.modelscope.cn/models/MNN/MiniCPM5-1B-MNN). The repo currently ships a single **block-wise int4** build (~597 MB for the weight file, ~626 MB for the directory):

```
.
├── config.json              # runtime config (edit this to switch backend / sampling)
├── llm_config.json          # model architecture config
├── llm.mnn                  # model graph
├── llm.mnn.weight           # quantised weights
├── embeddings_int4.bin      # embedding table
└── tokenizer.mtok           # tokenizer
```

If you need a different precision (int8 ~1.1 GB or fp16 ~2.1 GB), build it yourself with `llmexport` — see the [Build an MNN artifact from your own checkpoint](#build-an-mnn-artifact-from-your-own-checkpoint-advanced) section below.

## TL;DR — run the pre-converted MNN model

```bash
# 1. Download the model
huggingface-cli download taobao-mnn/MiniCPM5-1B-MNN --local-dir ./minicpm5-mnn
# or: modelscope download --model MNN/MiniCPM5-1B-MNN --local_dir ./minicpm5-mnn

# 2. Build mnn-llm
git clone --depth=1 https://github.com/alibaba/MNN.git
cd MNN && mkdir build && cd build
cmake .. -DMNN_BUILD_LLM=true       # add -DMNN_AVX512=true on x86 Linux/Mac for AVX-512
make -j16

# 3. Interactive chat (no prompt arg = interactive)
./llm_demo ../../minicpm5-mnn/config.json
```

## Build options at a glance

`mnn-llm` builds on top of the standard MNN engine. The only required flag is `-DMNN_BUILD_LLM=true`; everything else is a platform-specific accelerator.

| Platform | Extra cmake flag | Notes |
| --- | --- | --- |
| Linux / Mac (x86) | `-DMNN_AVX512=true` | AVX-512 CPU kernels |
| Android | `-DMNN_OPENCL=true` | GPU via OpenCL |
| Android (Qualcomm NPU) | `-DMNN_QNN=true -DMNN_WITH_PLUGIN=true` | requires QNN SDK; see the MNN docs |
| iOS | `-DMNN_METAL=ON` | use `package_scripts/ios/buildiOS.sh` instead of plain cmake |
| Web (WASM) | `-DMNN_FORBID_MULTI_THREAD=ON -DMNN_USE_THREAD_POOL=OFF -DMNN_USE_SSE=OFF` | use `emcmake cmake` |

For Android, build through the bundled script under `project/android/` rather than calling cmake directly:

```bash
cd project/android
mkdir build_64 && cd build_64
../build_64.sh -DMNN_BUILD_LLM=true -DMNN_OPENCL=true -DMNN_USE_LOGCAT=true
```

## CLI: `llm_demo` + `llm_bench`

`mnn-llm` currently ships two CLI binaries:

- **`llm_demo`** — interactive chat or batch generation from a prompt file
- **`llm_bench`** — throughput benchmark (prefill + decode tok/s)

```bash
# Interactive chat
./llm_demo ../../minicpm5-mnn/config.json

# Batch: one reply per line of prompts.txt
./llm_demo ../../minicpm5-mnn/config.json prompts.txt

# Throughput benchmark across configs
./llm_bench -m ../../minicpm5-mnn/config.json -a cpu -t 4 -p 32,64 -n 32 -rep 3
```

`llm_demo` reads sampling and hardware settings from `config.json` — there are no per-invocation flags to override them, so edit the config and re-run.

## `config.json` — the one knob that matters

`config.json` is the runtime config. Edit it to switch backend / sampling / memory mode without rebuilding the binary.

```json
{
    "backend_type": "cpu",
    "thread_num": 4,
    "precision": "low",
    "memory": "low",

    "max_new_tokens": 512,
    "reuse_kv": true,

    "sampler_type": "mixed",
    "mixed_samplers": ["topK", "tfs", "typical", "topP", "min_p", "temperature"],
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.95
}
```

| Key | Meaning |
| --- | --- |
| `backend_type` | `"cpu"` / `"opencl"` (Android GPU) / `"metal"` (iOS GPU). The compiled binary must have the matching flag (e.g. `MNN_OPENCL=true`). |
| `thread_num` | CPU thread count. **For OpenCL, use `68`** — it is not a thread count but an internal buffer/tuning mode flag. |
| `precision` | `"low"` = fp16 (recommended); `"high"` = fp32. |
| `memory` | `"low"` enables runtime quantisation of activations to save memory. |
| `reuse_kv` | reuse KV cache across multi-turn dialog (recommended for chat). |
| `use_mmap` | mmap weights from disk when memory is tight — **recommended on phones**. |
| `sampler_type` | `"mixed"` is the default and the most flexible; runs `mixed_samplers` in order. See the [MNN LLM sampler docs](https://github.com/alibaba/MNN/blob/master/docs/transformers/llm.md) for the full sampler list and all knobs (`min_p`, `tfs_z`, `repetition_penalty`, etc.). |

## Recommended sampling for MiniCPM5-1B

| Mode | `temperature` | `top_p` | When to use |
| --- | --- | --- | --- |
| Think | 0.9 | 0.95 | reasoning, math, code, multi-step |
| No-think | 0.7 | 0.95 | fast assistant, latency-bound |

Both modes are activated by sampling parameters only — the released chat template auto-injects `<think>\n` when no `system` message disables it, so you get think-mode behaviour by default.

## Mobile deployment

Drop the model directory into one of the bundled apps:

- **Android**: [`apps/Android/MnnLlmChat`](https://github.com/alibaba/MNN/tree/master/apps/Android/MnnLlmChat) — set `backend_type` to `"opencl"` in `config.json`.
- **iOS**: [`apps/iOS/MNNLLMChat`](https://github.com/alibaba/MNN/tree/master/apps/iOS/MNNLLMChat) — set `backend_type` to `"metal"` in `config.json`.

Both apps expect the same `config.json` + `llm.mnn` + `llm.mnn.weight` + `tokenizer.mtok` layout as the desktop CLI.

## Build an MNN artifact from your own checkpoint (advanced)

If you've trained your own MiniCPM5-1B variant (continue-pretraining, domain SFT, …) and want to publish an MNN build, use `llmexport`:

```bash
# 1. Install
git clone https://github.com/alibaba/MNN.git
cd MNN/transformers/llm/export
pip install -r requirements.txt
pip install MNN              # provides pymnn / mnnconvert needed by llmexport

# 2. Export
SRC=/path/to/your-MiniCPM5-fp16-hf
OUT=/path/to/output

# int4 (mobile default) — adding --hqq improves quality at no inference cost
python llmexport.py --path "$SRC" --export mnn --quant_bit 4 --quant_block 128 \
    --hqq --dst_path "$OUT/MiniCPM5-1B-MNN-int4"

# int8 (desktop / server, higher quality)
python llmexport.py --path "$SRC" --export mnn --quant_bit 8 --quant_block 128 \
    --dst_path "$OUT/MiniCPM5-1B-MNN-int8"
```

The output directory uses the same `config.json` + `llm.mnn` + `llm.mnn.weight` + `tokenizer.mtok` layout as the released artifact, so the CLI steps above work unchanged.

Approximate output sizes for MiniCPM5-1B:

| `--quant_bit` | Weight file | Use case |
| --- | --- | --- |
| 4 | ~597 MB | mobile / edge default |
| 8 | ~1.1 GB | desktop / server, higher quality |
| (none, fp16 via onnx → MNNConvert) | ~2.1 GB | reference quality |

For fp16 (no quantisation), first export to ONNX then convert:

```bash
python llmexport.py --path "$SRC" --export onnx --dst_path "$OUT/onnx"
./MNNConvert --modelFile "$OUT/onnx/onnx/llm.onnx" --MNNModel "$OUT/llm.mnn" \
    --keepInputFormat -f ONNX --transformerFuse=1 --allowCustomOp --saveExternalData
```

## Q&A

### Out-of-memory on a low-RAM device

Set `"use_mmap": true` (recommended on phones) and `"memory": "low"` in `config.json`. The first streams weights from disk instead of pinning them in RAM; the second activates runtime quantisation of activations. Drop `max_new_tokens` if the KV cache is the dominant cost.

### OpenCL backend isn't kicking in

Two things must line up:

1. The binary was built with `-DMNN_OPENCL=true`.
2. `config.json` has `"backend_type": "opencl"` AND `"thread_num": 68` (for OpenCL, this is a tuning-mode flag, not a thread count — using the default `4` will silently fall back to CPU-like behaviour).

### First OpenCL run is slow

The OpenCL backend tunes kernels on first launch and caches the result. Run a second time to see the steady-state numbers — this is also how the official `llm_bench` is measured.

## See also

- [MNN LLM module docs](https://github.com/alibaba/MNN/blob/master/docs/transformers/llm.md) — full config reference (sampler types, attention modes, NPU export)
- [`llmexport`](https://github.com/alibaba/MNN/tree/master/transformers/llm/export) — HF → MNN converter
- [`llama_cpp.md`](./llama_cpp.md) — alternative on-device path (GGUF)
- [`mlx.md`](./mlx.md) — Apple Silicon native path
