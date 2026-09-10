# Deploy MiniCPM5-2B and MiniCPM5-1B with LiteRT-LM (Android / iOS / desktop / IoT)

[LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) is Google's on-device runtime for language models, built on [LiteRT](https://github.com/google-ai-edge/litert) (formerly TensorFlow Lite). A model is packed once into a `.litertlm` bundle, and the same file runs on the CPU and the GPU of Android phones, iPhones, desktops and IoT boards such as the Raspberry Pi, through a command-line tool, an OpenAI-compatible local server, a Python package, and Kotlin / Swift APIs. Use this path when the target is a **phone, an app, or a local server on a laptop**.

The MiniCPM5-2B bundles are hosted in [litert-community/MiniCPM5-2B](https://huggingface.co/litert-community/MiniCPM5-2B); that card carries the conversion recipe and the measurements quoted here. The MiniCPM5-1B bundles are hosted in [litert-community/MiniCPM5-1B](https://huggingface.co/litert-community/MiniCPM5-1B). Every command on this page was run with `litert-lm` 0.17.0 on an Apple M4 Max; the Android check used `litertlm-android` 0.17.0 on a Galaxy S26.

## TL;DR

```bash
uv tool install litert-lm

# first run downloads the 1.55 GB int4 bundle
litert-lm run --from-huggingface-repo=litert-community/MiniCPM5-2B MiniCPM5-2B_int4.litertlm \
    --prompt "What is the capital of France?"
```

The file lands in `~/.litert-lm/cache/huggingface/litert-community/MiniCPM5-2B/`; later runs skip the download. GPU, the thinking switches, and sampling:

```bash
MODEL=~/.litert-lm/cache/huggingface/litert-community/MiniCPM5-2B/MiniCPM5-2B_int4.litertlm

litert-lm run "$MODEL" --backend gpu --prompt "1+1=?"
litert-lm run "$MODEL" --backend gpu --thinking false --prompt "1+1=?"
litert-lm run "$MODEL" --backend gpu --thinking-budget 2048 --prompt "1+1=?"
litert-lm run "$MODEL" --top-k 40 --top-p 0.95 --temperature 1.0 --thinking false --prompt "1+1=?"
```

## Pre-converted bundles

| Model | File | Recipe | Size | Tested on |
| --- | --- | --- | --- | --- |
| MiniCPM5-2B | [`MiniCPM5-2B_int4.litertlm`](https://huggingface.co/litert-community/MiniCPM5-2B/blob/main/MiniCPM5-2B_int4.litertlm) | int4 blockwise-32 + OCTAV on linears, int8 embedding | 1.55 GB | CPU + GPU: Mac, Galaxy S26, iPhone 17 Pro |
| MiniCPM5-2B | [`MiniCPM5-2B_int8.litertlm`](https://huggingface.co/litert-community/MiniCPM5-2B/blob/main/MiniCPM5-2B_int8.litertlm) | int8 dynamic linears + embedding, fp32 activations declared | 2.60 GB | CPU + GPU: Mac, Galaxy S26 (iOS: see below) |
| MiniCPM5-1B | [`minicpm_wi4b32_wi8_afp32.litertlm`](https://huggingface.co/litert-community/MiniCPM5-1B/blob/main/minicpm_wi4b32_wi8_afp32.litertlm) | int4 block-32 linears, int8 embedding and LM head, fp32 activations | 0.79 GB | CPU (Mac) |
| MiniCPM5-1B | [`minicpm_wi4b32_wi8_afp32_gpu_opt.litertlm`](https://huggingface.co/litert-community/MiniCPM5-1B/blob/main/minicpm_wi4b32_wi8_afp32_gpu_opt.litertlm) | same recipe, optimized for GPU execution | 0.79 GB | GPU (Mac) |
| MiniCPM5-1B | [`MiniCPM5-1B_dynamic_wi8_afp32.litertlm`](https://huggingface.co/litert-community/MiniCPM5-1B/blob/main/MiniCPM5-1B_dynamic_wi8_afp32.litertlm) | int8 dynamic, fp32 activations | 1.11 GB | CPU (Mac) |

Which 2B file: **int4 is the phone file** (smaller, fastest GPU decode on both devices measured) and the right one for direct answers or short reasoning. **int8 is the file when the reasoning has to complete**: on the same questions its thinking chains are 3–4× shorter than int4's and close where int4's keep going. Both 2B files embed the checkpoint's own `chat_template.jinja`, so `enable_thinking` and the tool-calling format work unchanged, and both declare the `thought` channel (next section). The 2B repo also carries two CPU-only files described in its card.

The int8 file, here on the Mac's Metal GPU:

```bash
litert-lm run --from-huggingface-repo=litert-community/MiniCPM5-2B MiniCPM5-2B_int8.litertlm --backend gpu \
    --prompt "A train travels 60 km in 45 minutes. What is its average speed in km/h?"
```

MiniCPM5-1B:

```bash
litert-lm run --from-huggingface-repo=litert-community/MiniCPM5-1B minicpm_wi4b32_wi8_afp32.litertlm --prompt "1+1=?"
litert-lm run --from-huggingface-repo=litert-community/MiniCPM5-1B minicpm_wi4b32_wi8_afp32_gpu_opt.litertlm --backend gpu --prompt "1+1=?"
```

## Thinking

Thinking is the default of the 2B files and the two 1B int4 files: with no thinking flag the model decides for itself and, in practice, reasons before every answer (`MiniCPM5-1B_dynamic_wi8_afp32` is packaged with thinking off by default). The bundles declare the reasoning as a `thought` channel, so the CLI prints it between `[thought]` and `[/thought]` and the Kotlin / Swift APIs hand it to you separately (`channels["thought"]`); the answer text stays clean.

- `--thinking false`: direct answers with no reasoning block; on `1+1=?` (GPU) the turn took 3.0 s against 24.7 s with thinking on.
- `--thinking-budget N`: caps the reasoning at N tokens. This model's chains run long, so start at 2048 and use 4096 for math.
- Sampling: OpenBMB recommends `temperature 1.0`, `top_p 0.95`. The bundles ship a greedy sampler (temperature 0, no top-k), which stays greedy whatever the temperature, so pass all three: `--top-k 40 --top-p 0.95 --temperature 1.0` (with `--top-p --temperature` alone the output stays byte-identical across seeds). The card's correctness numbers are greedy.
- The KV cache takes the bundle's setting: 4096 tokens for the two 2B files and `MiniCPM5-1B_dynamic_wi8_afp32`, 1024 for the two 1B int4 files (`--max-num-tokens` to change it); the prompt format is ChatML, from the template embedded in each bundle.

## OpenAI-compatible server

`litert-lm serve` exposes every bundle in the local registry on `/v1/models` and `/v1/chat/completions` ([server guide](https://developers.google.com/edge/litert-lm/cli/openai_server)). Import the file once under a name, start the server, then call it like any OpenAI endpoint:

```bash
litert-lm import --from-huggingface-repo=litert-community/MiniCPM5-2B MiniCPM5-2B_int4.litertlm minicpm5-2b
litert-lm serve --host 127.0.0.1 --port 9379

curl http://127.0.0.1:9379/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "minicpm5-2b", "messages": [{"role": "user", "content": "1+1=?"}], "reasoning_effort": "none"}'
```

`"reasoning_effort": "none"` gives a direct answer; any other value (`low` to `xhigh`) turns thinking on, and the reasoning is counted in `usage.completion_tokens_details.reasoning_tokens`. With thinking on, the reasoning needs room (the int4 file filled the 4096-token context on `1+1=?` and returned no content); when the reasoning has to complete, use the int8 file. `max_completion_tokens` caps the reply, `temperature`, `top_p`, `top_k` and `seed` are honored, and `"stream": true` streams.

## Android

**AI Edge Gallery (no code).** Install [Google AI Edge Gallery](https://play.google.com/store/apps/details?id=com.google.ai.edge.gallery), open **Model manager**, tap **+** and choose **Import from HF**, then paste the Hugging Face link of the `.litertlm` file (or download the file to the phone first and use **From local model file**). The [litert-community card](https://huggingface.co/litert-community/MiniCPM5-2B#edge-gallery-app-android) and the [Gallery wiki](https://github.com/google-ai-edge/gallery/wiki) have the details.

**Kotlin API (your own app).** The runtime is one dependency from Google Maven ([Android guide](https://developers.google.com/edge/litert-lm/android)):

```kotlin
// build.gradle.kts
implementation("com.google.ai.edge.litertlm:litertlm-android:0.17.0")
```

The GPU backend needs these entries in `AndroidManifest.xml` ([Android guide](https://developers.google.com/edge/litert-lm/android)); apps targeting Android 12 (API 31) or higher cannot load vendor libraries they do not declare, and without them `initialize()` succeeds and the first message fails with `Can not find OpenCL library on this device`:

```xml
<application>
    <uses-native-library android:name="libvndksupport.so" android:required="false" />
    <uses-native-library android:name="libOpenCL.so" android:required="false" />
</application>
```

```kotlin
import com.google.ai.edge.litertlm.*

val engine = Engine(EngineConfig(
    modelPath = modelFile.absolutePath,
    backend = Backend.GPU(),
    cacheDir = context.cacheDir.absolutePath,
))
engine.initialize()

engine.createConversation(ConversationConfig(maxOutputToken = 1024)).use { conversation ->
    val reply = conversation.sendMessage("Explain on-device AI in simple terms.")
    println(reply)
    println(reply.channels["thought"])
}
engine.close()
```

Tested on a Galaxy S26 (Snapdragon SM8850, Adreno) with `litertlm-android` 0.17.0 and the int4 file on `Backend.GPU()` (`Backend.CPU()` selects the CPU): the engine initializes with every node of every signature delegated to OpenCL, and the reply arrives with the reasoning in `channels["thought"]`. `ConversationConfig` also takes `thinkingConfig = ThinkingConfig(enableThinking = false)` or `ThinkingConfig(enableThinking = true, thinkingTokenBudget = 2048)` and a `samplerConfig = SamplerConfig(topK = 40, topP = 0.95, temperature = 1.0)`, as in the Android guide. `conversation.sendMessageAsync(...)` streams.

## iOS

Tested on an iPhone 17 Pro with the int4 file on both backends (Metal GPU and CPU; init 5.7 s and 2.2 s). The int8 file's main weight section is 2.33 GB, more than an iOS app maps in one piece with the default entitlements; the `com.apple.developer.kernel.increased-memory-limit` entitlement lifts that limit (single sections of 2.87 GB and 4.24 GB from other models have loaded and run on an iPhone 17 Pro with it), not checked with this file. The int4 file's main section is 1.28 GB. The Swift API is documented in the [LiteRT-LM Swift guide](https://developers.google.com/edge/litert-lm/swift) (Swift Package Manager, `https://github.com/google-ai-edge/LiteRT-LM`); this page does not cover the Xcode steps.

## Speed and accuracy

All numbers are from the [litert-community card](https://huggingface.co/litert-community/MiniCPM5-2B). Apple M4 Max, `litert-lm benchmark` 0.17.0, 256-token prefill / 256-token decode, 3 runs, 1024-token KV cache, no compiled-model cache (`--cache no`):

| File | Backend | Prefill | Decode | Time to first token | Init |
| --- | --- | --- | --- | --- | --- |
| int4 | GPU (Metal) | 1699 tok/s | 92.8 tok/s | 0.16 s | 3.7 s |
| int4 | CPU | 149 tok/s | 31.1 tok/s | 1.76 s | 4.5 s |
| int8 (fp32 activations) | GPU (Metal) | 1405 tok/s | 74.7 tok/s | 0.20 s | 3.0 s |
| int8 (fp32 activations) | CPU | 161 tok/s | 30.0 tok/s | 1.62 s | 15.0 s |

Galaxy S26 (Snapdragon SM8850, Adreno), the LiteRT-LM v0.16.0 release binary, 205-token prompt, 2 runs per cell (ranges shown; a reasoning model decodes its own full response, so decode lengths vary):

| File | Backend | Prefill | Decode | Time to first token | Peak RSS |
| --- | --- | --- | --- | --- | --- |
| int4 | GPU (OpenCL) | 401–411 tok/s | 16.1–18.6 tok/s | 0.56 s | 1.14 GB |
| int4 | CPU | 39–72 tok/s | 15.6–15.8 tok/s | 2.9–5.3 s | 2.12 GB |
| int8 (fp32 activations) | GPU (OpenCL) | 150–160 tok/s | 10.9–12.8 tok/s | 1.4 s | 1.10 GB |
| int8 (fp32 activations) | CPU | 103–157 tok/s | 11.7 tok/s | 1.4–2.1 s | 2.90 GB |

Accuracy, GSM8K first 100 test questions, greedy, thinking off, up to 2048 new tokens: bf16 PyTorch 92 %, int8 CPU 91 %, int4 CPU 86 %, int4 GPU 87 %. Both 2B files score 8/8 on the card's 8-question sanity gate on the Mac, CPU and GPU.

## Common pitfalls

- **Thinking on the int4 file** runs long chains: on the card's ten-question thinking-on subset (3584-token budget) it closes 0/10 on the CPU, where int8 closes 9/10 like the bf16 model. Pass `--thinking false` for direct answers; when the reasoning has to complete, use the int8 file.
- **int8 on iOS needs the increased-memory-limit entitlement.** With the default entitlements the 2.33 GB weight section does not map; add `com.apple.developer.kernel.increased-memory-limit` (see iOS). int4's 1.28 GB main section is under the limit.
- **The output cap is what leaves no answer**: when the reasoning runs into the context (`--max-num-tokens`, 4096 for the 2B files) or the app's `maxOutputToken`, the chain is cut and nothing follows. When the reasoning has to complete, keep those at 2048 or more (4096 for math); 1024, as in the snippets, is enough for short answers.
- **First run per backend is slow**: the CPU run writes an XNNPACK weight cache beside the file and the GPU run compiles its kernels (`--cache no` skips the disk cache; the Kotlin `cacheDir` is the same mechanism).
- **GPU in your own Android app**: without the `<uses-native-library>` entries above, `initialize()` succeeds and the first message fails with `Can not find OpenCL library on this device`.
- **Temperature without top-k does nothing**: `--temperature` and `--top-p` only take effect together with `--top-k` above 1 (the default is greedy). The Kotlin `SamplerConfig` takes the same three fields.
- **Scripts**: the CLI reads stdin; when it runs from a script or a job with an open pipe on stdin, redirect it (`< /dev/null`) or it waits for end-of-input before generating.

## See also

- [litert-community/MiniCPM5-2B](https://huggingface.co/litert-community/MiniCPM5-2B) — recipe, correctness, all measurements, `litertlm_manifest.json`
- [litert-community/MiniCPM5-1B](https://huggingface.co/litert-community/MiniCPM5-1B)
- LiteRT-LM guides: [CLI](https://developers.google.com/edge/litert-lm/cli), [OpenAI-compatible server](https://developers.google.com/edge/litert-lm/cli/openai_server), [CLI configuration](https://developers.google.com/edge/litert-lm/cli/configuration), [model management](https://developers.google.com/edge/litert-lm/cli/model_management), [Android](https://developers.google.com/edge/litert-lm/android), [Swift](https://developers.google.com/edge/litert-lm/swift), [Python](https://developers.google.com/edge/litert-lm/python)
