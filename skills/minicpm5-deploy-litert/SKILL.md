---
name: minicpm5-deploy-litert
description: Run MiniCPM5-2B or MiniCPM5-1B on-device with Google's LiteRT-LM runtime — the litert-lm CLI on a desktop, the Kotlin API or the AI Edge Gallery app on Android, the same .litertlm bundle on CPU or GPU. Use when the user says "LiteRT", "LiteRT-LM", "litertlm", ".litertlm", "Android", "Edge Gallery", "on-device app", or wants one bundle for phones and desktops.
---

# Deploy MiniCPM5-2B / MiniCPM5-1B with LiteRT-LM (Android / iOS / desktop)

Google's on-device runtime (LiteRT, formerly TensorFlow Lite). One `.litertlm` bundle runs on CPU and GPU through a CLI, Python, Kotlin and Swift. The bundles are hosted in `litert-community`. Human-readable reference: [`docs/deployment/litert.md`](../../docs/deployment/litert.md).

## Required input

| Var | Example | Default |
| --- | --- | --- |
| `LITERT_REPO` | `litert-community/MiniCPM5-2B` or `litert-community/MiniCPM5-1B` | `litert-community/MiniCPM5-2B` |
| `FILE` | `MiniCPM5-2B_int4.litertlm` (1.55 GB, the phone file) or `MiniCPM5-2B_int8.litertlm` (2.60 GB, reasoning that has to finish); 1B: `minicpm_wi4b32_wi8_afp32.litertlm` (CPU) or `minicpm_wi4b32_wi8_afp32_gpu_opt.litertlm` (GPU) | `MiniCPM5-2B_int4.litertlm` |
| `BACKEND` | `cpu` or `gpu` | `cpu` |
| `THINKING` | `true` (reason first) or `false` (direct answer) | the model decides; in practice it reasons |
| `PROMPT` | `1+1=?` | `1+1=?` |

## Steps

### 1. Install (once)

```bash
uv tool install litert-lm      # litert-lm 0.17.0 at the time of writing
```

### 2A. Run a pre-converted bundle (desktop CLI)

```bash
litert-lm run --from-huggingface-repo="${LITERT_REPO}" "${FILE}" \
    --backend "${BACKEND}" --thinking "${THINKING}" --prompt "${PROMPT}"
```

The first run downloads the file into `~/.litert-lm/cache/huggingface/<repo>/`; later runs skip the download. With thinking on, the reasoning prints between `[thought]` and `[/thought]` and the answer follows; `--thinking-budget 2048` caps it (never lower: a chain cut mid-thought yields no answer). Sampling as OpenBMB recommends: `--top-k 40 --top-p 0.95 --temperature 1.0` (the CLI's default top-k is 1 = greedy, so temperature alone changes nothing).

### 2B. Android

- **No code**: the [AI Edge Gallery](https://play.google.com/store/apps/details?id=com.google.ai.edge.gallery) app, Model manager → **+** → **Import from HF** → paste the file's Hugging Face link.
- **Your own app**: `implementation("com.google.ai.edge.litertlm:litertlm-android:0.17.0")` from Google Maven, the `libOpenCL.so` `<uses-native-library>` entries in the manifest for the GPU, then:

```kotlin
val engine = Engine(EngineConfig(modelPath = path, backend = Backend.GPU(), cacheDir = context.cacheDir.absolutePath))
engine.initialize()
engine.createConversation(ConversationConfig(maxOutputToken = 1024)).use { conversation ->
    val reply = conversation.sendMessage("1+1=?")
    println(reply); println(reply.channels["thought"])   // answer; reasoning
}
```

`ConversationConfig(thinkingConfig = ThinkingConfig(enableThinking = false))` turns thinking off. Full snippet, manifest and the Galaxy S26 check: the cookbook.

### 3. Validate

The reply contains `2` for `1+1=?`. With thinking on it comes after `[/thought]`; with `--thinking false` it is the whole reply. On the int4 file prefer `--thinking false` or `--backend gpu` for this check: on the CPU with thinking on, `1+1=?` can run to the token budget without an answer (pitfalls).

## Common pitfalls

- **int4 + thinking on may not close its reasoning** (CPU especially): the card's 10-question thinking-on subset closes 0/10 on int4 CPU, 9/10 on int8. Use `--thinking false` on int4, or the int8 file, when the answer matters.
- **int8 declares fp32 activations** (a correctness fix for the GPU's fp16 default); GPU decode is ~14 % slower for it, and on Adreno int8 GPU decode is level with the CPU. int4 is the phone file.
- **int8 is not for iOS**: its 2.33 GB weight section exceeds a default-entitlement app's single-section memory-map budget. iPhone: int4 (checked on iPhone 17 Pro, GPU and CPU).
- **First run per backend is slow**: a 1.3 GB XNNPACK cache is written beside the file on the CPU, kernels are compiled on the GPU (`--cache no` to skip the disk cache).
- **Android GPU without the manifest entries**: `initialize()` succeeds, the first message fails with `Can not find OpenCL library on this device`.
- **Temperature without top-k does nothing**: pass `--top-k` above 1 together with `--top-p` / `--temperature` (default top-k 1 = greedy).
- **Scripts**: the CLI reads stdin; redirect it (`< /dev/null`) when running from a job with an open pipe.

## When NOT to use

- Python on a Mac, no app → `minicpm5-deploy-mlx`
- GGUF, one-line CLI on a laptop → `minicpm5-deploy-ollama`; CPU / CUDA build → `minicpm5-deploy-llama-cpp`
- Server with an OpenAI-compatible endpoint → `minicpm5-deploy-vllm` or `minicpm5-deploy-sglang`

## Reference

[`docs/deployment/litert.md`](../../docs/deployment/litert.md)
