---
name: minicpm5-deploy-litert
description: Run MiniCPM5-2B or MiniCPM5-1B on-device with Google's LiteRT-LM runtime — the litert-lm CLI or its OpenAI-compatible server on a desktop, the Kotlin API or the AI Edge Gallery app on Android, the same .litertlm bundle on CPU or GPU. Use when the user says "LiteRT", "LiteRT-LM", "litertlm", ".litertlm", "Android", "Edge Gallery", "on-device app", or wants one bundle for phones and desktops.
---

# Deploy MiniCPM5-2B / MiniCPM5-1B with LiteRT-LM (Android / iOS / desktop / IoT)

[LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM), Google's on-device runtime built on [LiteRT](https://github.com/google-ai-edge/litert) (formerly TensorFlow Lite). One `.litertlm` bundle, CPU or GPU, through a CLI, an OpenAI-compatible server, Python, Kotlin and Swift. The bundles are hosted in `litert-community`. Human-readable reference: [`docs/deployment/litert.md`](../../docs/deployment/litert.md).

## Required input

| Var | Example | Default |
| --- | --- | --- |
| `LITERT_REPO` | `litert-community/MiniCPM5-2B` or `litert-community/MiniCPM5-1B` | `litert-community/MiniCPM5-2B` |
| `FILE` | `MiniCPM5-2B_int4.litertlm` (1.55 GB, the phone file) or `MiniCPM5-2B_int8.litertlm` (2.60 GB, reasoning that has to finish); 1B: `minicpm_wi4b32_wi8_afp32.litertlm` (CPU) or `minicpm_wi4b32_wi8_afp32_gpu_opt.litertlm` (GPU) | `MiniCPM5-2B_int4.litertlm` |
| `BACKEND` | `cpu` or `gpu` | `cpu` |
| `THINKING` | `true` (reason first) or `false` (direct answer) | `false` |
| `PROMPT` | `1+1=?` | `1+1=?` |

## Steps

### 1. Install (once)

```bash
uv tool install litert-lm
```

### 2A. Run a pre-converted bundle (desktop CLI)

```bash
litert-lm run --from-huggingface-repo="${LITERT_REPO}" "${FILE}" \
    --backend "${BACKEND}" --thinking "${THINKING:-false}" --prompt "${PROMPT}"
```

The first run downloads the file into `~/.litert-lm/cache/huggingface/<repo>/`; later runs skip the download. With thinking on, the reasoning prints between `[thought]` and `[/thought]` and the answer follows; `--thinking-budget 2048` caps the reasoning (4096 for math); `THINKING=true` makes the model reason first. Sampling as OpenBMB recommends: `--top-k 40 --top-p 0.95 --temperature 1.0` (the CLI's default top-k is 1 = greedy, so temperature alone changes nothing).

### 2B. Android

- **No code**: the [AI Edge Gallery](https://play.google.com/store/apps/details?id=com.google.ai.edge.gallery) app, Model manager → **+** → **Import from HF** → paste the file's Hugging Face link.
- **Your own app**: `implementation("com.google.ai.edge.litertlm:litertlm-android:0.17.0")` from Google Maven, the `libOpenCL.so` `<uses-native-library>` entries in the manifest for the GPU, then:

```kotlin
val engine = Engine(EngineConfig(modelPath = path, backend = Backend.GPU(), cacheDir = context.cacheDir.absolutePath))
engine.initialize()
engine.createConversation(ConversationConfig(maxOutputToken = 1024)).use { conversation ->
    val reply = conversation.sendMessage("1+1=?")
    println(reply); println(reply.channels["thought"])
}
```

`ConversationConfig(thinkingConfig = ThinkingConfig(enableThinking = false))` turns thinking off. Full snippet, manifest and the Galaxy S26 check: the cookbook.

### 2C. OpenAI-compatible server (desktop)

```bash
litert-lm import --from-huggingface-repo="${LITERT_REPO}" "${FILE}" minicpm5-2b
litert-lm serve --host 127.0.0.1 --port 9379
curl http://127.0.0.1:9379/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"model": "minicpm5-2b", "messages": [{"role": "user", "content": "1+1=?"}], "reasoning_effort": "none"}'
```

`reasoning_effort` `none` = direct answer, any other value = thinking on; the cap field is `max_completion_tokens`; `top_k` is accepted alongside `temperature` / `top_p` / `seed`; `"stream": true` streams ([server guide](https://developers.google.com/edge/litert-lm/cli/openai_server)).

### 3. Validate

The reply contains `2` for `1+1=?`: with `--thinking false` it is the whole reply; with thinking on it follows `[/thought]`. Through the server (2C), the same check is HTTP 200 with `choices[0].message.content` containing `2`, and `GET /v1/models` lists `minicpm5-2b`.

## Common pitfalls

- **Thinking on the int4 file** runs long chains: pass `--thinking false` for direct answers, or use the int8 file when the reasoning has to complete.
- **int8 on iOS needs the increased-memory-limit entitlement** (2.33 GB weight section; the default entitlements do not map it). int4's 1.28 GB main section is under the limit (checked on iPhone 17 Pro, GPU and CPU).
- **First run per backend is slow**: an XNNPACK weight cache is written beside the file on the CPU, kernels are compiled on the GPU (`--cache no` to skip the disk cache).
- **Android GPU without the manifest entries**: `initialize()` succeeds, the first message fails with `Can not find OpenCL library on this device`.
- **Temperature without top-k does nothing**: pass `--top-k` above 1 together with `--top-p` / `--temperature` (the default is greedy).
- **Scripts**: the CLI reads stdin; redirect it (`< /dev/null`) when running from a job with an open pipe.

## Reference

[`docs/deployment/litert.md`](../../docs/deployment/litert.md)
