# Deploy MiniCPM5-2B and MiniCPM5-1B with LiteRT-LM (Android / iOS / desktop)

[LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) is Google's on-device runtime for language models (LiteRT is the runtime formerly called TensorFlow Lite). A model is packed once into a `.litertlm` bundle, and the same file runs on the CPU and the GPU of Android phones, iPhones and desktops, through a command-line tool, a Python package, and Kotlin / Swift APIs. Use this path when the target is a **phone or an app**. For a Python process on a Mac use [MLX](./mlx.md); for a GGUF workflow use [llama.cpp](./llama_cpp.md) or [Ollama](./ollama.md).

The MiniCPM5-2B bundles below are **community conversions** (not an OpenBMB or Google release), hosted in [litert-community/MiniCPM5-2B](https://huggingface.co/litert-community/MiniCPM5-2B); that card carries the conversion recipe and the measurements quoted here. The MiniCPM5-1B bundles are hosted in [litert-community/MiniCPM5-1B](https://huggingface.co/litert-community/MiniCPM5-1B). Every command on this page was run with `litert-lm` 0.17.0 on an Apple M4 Max; the Android check used `litertlm-android` 0.17.0 on a Galaxy S26.

## TL;DR

```bash
uv tool install litert-lm     # the LiteRT-LM CLI (0.17.0 at the time of writing)

# Downloads the 1.55 GB int4 bundle on first use, runs it on the CPU:
litert-lm run --from-huggingface-repo=litert-community/MiniCPM5-2B MiniCPM5-2B_int4.litertlm \
    --prompt "What is the capital of France?"
# [thought] … The capital of France is Paris. I should provide a clear and concise answer. [/thought]
# The capital of France is Paris.
```

The file lands in `~/.litert-lm/cache/huggingface/litert-community/MiniCPM5-2B/`; later runs skip the download. GPU, and the thinking switches:

```bash
MODEL=~/.litert-lm/cache/huggingface/litert-community/MiniCPM5-2B/MiniCPM5-2B_int4.litertlm

litert-lm run "$MODEL" --backend gpu --prompt "1+1=?"                    # reasoning on the thought channel, then: 2
litert-lm run "$MODEL" --backend gpu --thinking false --prompt "1+1=?"   # direct answer, no reasoning
litert-lm run "$MODEL" --backend gpu --thinking-budget 2048 --prompt "1+1=?"
litert-lm run "$MODEL" --backend gpu --top-k 40 --top-p 0.95 --temperature 1.0 --thinking false --prompt "1+1=?"   # sampled (see Thinking)
```

## Pre-converted bundles

| Model | File | Recipe | Size | Runs on |
| --- | --- | --- | --- | --- |
| MiniCPM5-2B | [`MiniCPM5-2B_int4.litertlm`](https://huggingface.co/litert-community/MiniCPM5-2B/blob/main/MiniCPM5-2B_int4.litertlm) | int4 blockwise-32 linears, int8 embedding | 1.55 GB | CPU + GPU: Mac, Galaxy S26, iPhone 17 Pro |
| MiniCPM5-2B | [`MiniCPM5-2B_int8.litertlm`](https://huggingface.co/litert-community/MiniCPM5-2B/blob/main/MiniCPM5-2B_int8.litertlm) | int8 dynamic linears + embedding, fp32 activations declared | 2.60 GB | CPU + GPU: Mac, Galaxy S26 (not iOS, see pitfalls) |
| MiniCPM5-1B | [`minicpm_wi4b32_wi8_afp32.litertlm`](https://huggingface.co/litert-community/MiniCPM5-1B/blob/main/minicpm_wi4b32_wi8_afp32.litertlm) | int4 block-32 linears, int8 embedding and LM head, fp32 activations | 0.79 GB | CPU (Mac) |
| MiniCPM5-1B | [`minicpm_wi4b32_wi8_afp32_gpu_opt.litertlm`](https://huggingface.co/litert-community/MiniCPM5-1B/blob/main/minicpm_wi4b32_wi8_afp32_gpu_opt.litertlm) | same recipe, graph laid out for the GPU | 0.79 GB | GPU (Mac) |
| MiniCPM5-1B | [`MiniCPM5-1B_dynamic_wi8_afp32.litertlm`](https://huggingface.co/litert-community/MiniCPM5-1B/blob/main/MiniCPM5-1B_dynamic_wi8_afp32.litertlm) | int8 dynamic, fp32 activations | 1.11 GB | CPU (Mac) |

Which 2B file: **int4 is the phone file** (smaller, fastest GPU decode on every device measured) and the right one for direct answers or short reasoning. **int8 is the file when the reasoning has to complete**: on the same questions its thinking chains are 3–4× shorter than int4's and they terminate where int4 runs into the token budget. Both 2B files embed the checkpoint's own `chat_template.jinja`, so `enable_thinking` and the tool-calling format work unchanged, and both declare the `thought` channel (next section). The 2B repo also carries two CPU-only files described in its card.

```bash
# int8 (2.60 GB), when the reasoning has to finish; the GPU here is the Mac's Metal:
litert-lm run --from-huggingface-repo=litert-community/MiniCPM5-2B MiniCPM5-2B_int8.litertlm --backend gpu \
    --prompt "A train travels 60 km in 45 minutes. What is its average speed in km/h?"

# MiniCPM5-1B:
litert-lm run --from-huggingface-repo=litert-community/MiniCPM5-1B minicpm_wi4b32_wi8_afp32.litertlm --prompt "1+1=?"
litert-lm run --from-huggingface-repo=litert-community/MiniCPM5-1B minicpm_wi4b32_wi8_afp32_gpu_opt.litertlm --backend gpu --prompt "1+1=?"
```

## Thinking

Thinking is the model's default: with no thinking flag it decides for itself and, in practice, reasons before every answer. The bundles declare the reasoning as a `thought` channel, so the CLI prints it between `[thought]` and `[/thought]` and the Kotlin / Swift APIs hand it to you separately (`channels["thought"]`); the answer text stays clean.

- `--thinking false`: direct answers. Two- to seven-token replies on trivial questions, about 10× faster turns.
- `--thinking-budget N`: caps the reasoning. Keep N at 2048 or more (4096 for math): a chain cut mid-thought yields no final answer at all.
- Sampling: OpenBMB recommends `temperature 1.0`, `top_p 0.95`. The CLI's default top-k is 1, which is greedy whatever the temperature, so pass all three: `--top-k 40 --top-p 0.95 --temperature 1.0` (with `--top-p --temperature` alone the output stays byte-identical across seeds). The card's correctness numbers are greedy.
- The KV cache is 4096 tokens by default (`--max-num-tokens` to change it); the prompt format is the checkpoint's ChatML template.

## Android

**AI Edge Gallery (no code).** Install [Google AI Edge Gallery](https://play.google.com/store/apps/details?id=com.google.ai.edge.gallery), open **Model manager**, tap **+** and choose **Import from HF**, then paste the Hugging Face link of the `.litertlm` file (or download the file to the phone first and use **From local model file**). The [litert-community card](https://huggingface.co/litert-community/MiniCPM5-2B#edge-gallery-app-android) and the [Gallery wiki](https://github.com/google-ai-edge/gallery/wiki) have the details.

**Kotlin API (your own app).** The runtime is one dependency from Google Maven ([Android guide](https://developers.google.com/edge/litert-lm/android)):

```kotlin
// build.gradle.kts — repositories { google() }
implementation("com.google.ai.edge.litertlm:litertlm-android:0.17.0")
```

The GPU backend needs the OpenCL library declared in `AndroidManifest.xml` (Android 12+ hides vendor libraries from apps otherwise; without it the first message fails with `Can not find OpenCL library on this device`):

```xml
<application …>
    <uses-native-library android:name="libOpenCL.so" android:required="false" />
    <uses-native-library android:name="libOpenCL-car.so" android:required="false" />
    <uses-native-library android:name="libOpenCL-pixel.so" android:required="false" />
</application>
```

```kotlin
import com.google.ai.edge.litertlm.*

val engine = Engine(EngineConfig(
    modelPath = modelFile.absolutePath,          // MiniCPM5-2B_int4.litertlm in your app's storage
    backend = Backend.GPU(),                     // or Backend.CPU()
    cacheDir = context.cacheDir.absolutePath,    // compiled-kernel cache; the first load is slower
))
engine.initialize()

engine.createConversation(ConversationConfig(maxOutputToken = 1024)).use { conversation ->
    val reply = conversation.sendMessage("Explain on-device AI in simple terms.")
    println(reply)                       // the answer
    println(reply.channels["thought"])   // the reasoning, kept out of the answer
}
engine.close()
```

Checked on a Galaxy S26 (Snapdragon SM8850, Adreno) with `litertlm-android` 0.17.0 and the int4 file: the engine initializes on `Backend.GPU()` with every node of every signature delegated to OpenCL (1873 of 1873 on the 1024-token prefill, 1692 of 1692 on decode; only the externalized embedding lookup runs on the CPU), and the reply arrives with the reasoning in `channels["thought"]`. `ConversationConfig` also takes `thinkingConfig = ThinkingConfig(enableThinking = false)` or `ThinkingConfig(enableThinking = true, thinkingTokenBudget = 2048)` and a `samplerConfig = SamplerConfig(topK = 40, topP = 0.95, temperature = 1.0)`, as in the Android guide. `conversation.sendMessageAsync(...)` streams.

## iOS

The int4 file passes the card's 8-question check on an iPhone 17 Pro on both backends (Metal GPU and CPU, init 5.7 s and 2.2 s). The int8 file is a desktop / Android build: its main weight section is 2.33 GB, above the single-section memory-map budget of an iOS app with default entitlements. The Swift API is documented in the [LiteRT-LM Swift guide](https://developers.google.com/edge/litert-lm/swift) (Swift Package Manager, `https://github.com/google-ai-edge/LiteRT-LM`); this page does not cover the Xcode steps.

## Measured (from the litert-community card)

Apple M4 Max, `litert-lm benchmark` 0.17.0, 256-token prefill / 256-token decode, 3 runs, no compiled-kernel cache:

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

Accuracy, GSM8K first 100 test questions, greedy, thinking off (the protocol of OpenBMB's MiniCPM5 cards), 2048 new tokens: bf16 PyTorch 92 %, int8 CPU 91 %, int4 CPU 86 %, int4 GPU 87 %. Both 2B files score 8/8 on the card's 8-question sanity gate on CPU and GPU.

## Common pitfalls

- **int4 with thinking on may never close its reasoning.** Quantization costs this 42-layer model its thinking discipline first: on the card's 10-question thinking-on subset the int4 file closes 0/10 on the CPU (it keeps re-checking until the budget), while int8 closes 9/10 like the bf16 model. The same shows on the CLI: `1+1=?` on the CPU and `Explain GQA in one sentence.` on the GPU ran into the 4096-token budget with no answer. When the answer matters, use `--thinking false` on int4, or the int8 file.
- **int8 declares fp32 activations in the bundle.** With the GPU's default fp16 activations the int8 model's reasoning on one gate question ran 2000+ tokens without closing; with fp32 it closes in ~450 tokens. The cost is about 14 % of GPU decode speed, and on Adreno the int8 GPU decode ends up level with the same phone's CPU.
- **int8 on iOS**: not loadable by a default-entitlement app (2.33 GB single section, above). Use int4 on iPhone.
- **Budget cuts leave no answer**: a `--thinking-budget` below ~2048 (4096 for math) truncates the chain and the model emits nothing after it.
- **First run per backend is slow**: the CPU run writes a 1.3 GB XNNPACK cache beside the file and the GPU run compiles its kernels (`--cache no` skips the disk cache; the Kotlin `cacheDir` is the same mechanism).
- **GPU in your own Android app**: without the `<uses-native-library>` entries above, `initialize()` succeeds and the first message fails with `Can not find OpenCL library on this device`.
- **Temperature without top-k does nothing**: `--temperature` and `--top-p` only take effect together with `--top-k` above 1 (default 1 = greedy). The Kotlin `SamplerConfig` takes the same three fields.
- **Scripts**: the CLI reads stdin; when it runs from a script or a job with an open pipe on stdin, redirect it (`< /dev/null`) or it waits for end-of-input before generating.

## See also

- [`ollama.md`](./ollama.md) — one-line CLI path on a laptop, GGUF
- [`mlx.md`](./mlx.md) — Python on Apple Silicon
- [`llama_cpp.md`](./llama_cpp.md) — GGUF on CPU / CUDA
- [litert-community/MiniCPM5-2B](https://huggingface.co/litert-community/MiniCPM5-2B) — recipe, correctness, all measurements, `litertlm_manifest.json`
- [litert-community/MiniCPM5-1B](https://huggingface.co/litert-community/MiniCPM5-1B)
- LiteRT-LM guides: [CLI](https://developers.google.com/edge/litert-lm/cli), [Android](https://developers.google.com/edge/litert-lm/android), [Swift](https://developers.google.com/edge/litert-lm/swift), [Python](https://developers.google.com/edge/litert-lm/python)
