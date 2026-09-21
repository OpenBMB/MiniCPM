# Deploy MiniCPM5-2B / MiniCPM5-1B with Apple Core AI (iPhone / iPad / Mac)

[Core AI](https://developer.apple.com/documentation/coreai) is Apple's on-device ML runtime in iOS 27 / macOS 27. A model is exported once to an `.aimodel` bundle and then runs inside a Swift app on iPhone, iPad and Apple Silicon Mac, with no Python at runtime. The bundle below runs on the GPU through Core AI's pipelined engine. Use this path when the target is an **app** (Swift / Xcode). For Python on a Mac, use [MLX](./mlx.md).

The bundles below are **community conversions**, maintained in the [Core AI Model Zoo](https://github.com/john-rocky/coreai-model-zoo) (not an OpenBMB or Apple release). Each Hugging Face card carries the conversion recipe and the measurements quoted here. This page covers MiniCPM5-2B and MiniCPM5-1B; the catalog ids are `minicpm5-2b` and `minicpm5-1b`.

## TL;DR

```bash
# Requires macOS 27 + Xcode 27 (CoreAIKit 0.4.1 targets Xcode 27 beta 5).
git clone https://github.com/john-rocky/coreai-kit
cd coreai-kit/Examples/ChatDemo

# Headless run on the Mac (downloads the bundle on first use, then loads from cache):
swift run -c release chat-cli --model minicpm5-2b --prompt "1+1=?"
# → 1 + 1 = 2.
swift run -c release chat-cli --model minicpm5-1b --prompt "1+1=?"   # the 1B: same command, 1.1 GB

# On an iPhone: open ChatDemo.xcodeproj, pick your device as the run destination,
# set your signing team, Run, then choose "MiniCPM5 2B" (or "MiniCPM5 1B") in the model picker.
```

## Pre-converted bundle

| Model | Hugging Face repo | Quantization | Size | Numerics vs HF fp32 |
| --- | --- | --- | --- | --- |
| MiniCPM5-2B | [mlboydaisuke/MiniCPM5-2B-CoreAI](https://huggingface.co/mlboydaisuke/MiniCPM5-2B-CoreAI) | int8 weight-only, per-block-32 | 2.7 GB | 24/24 + 24/24 greedy tokens exact on iPhone, 16/16 on Mac |
| MiniCPM5-1B | [mlboydaisuke/MiniCPM5-1B-CoreAI](https://huggingface.co/mlboydaisuke/MiniCPM5-1B-CoreAI) (revision b8a6ac397ccd5fb815f97336f8a8b1800b110da1 or newer) | int8 weight-only, per-block-32 | 1.1 GB | 24/24 greedy tokens exact + the stop on iPhone, 16/16 on Mac |

Both are dynamic-shape bundles: one file runs unchanged on macOS and iOS through Core AI's pipelined engine. SDPA, RoPE and RMSNorm stay in full precision; the chat template's end-of-turn token `<|im_end|>` (id 130073) is set as the bundle's `eos_token`, and the model stops there (checked on `1+1=?` with both runtimes below, for both sizes).

Use the 1B revision named above or newer: the earlier 1B revision (`5ad650f`, per-channel int8) did not stop at the end of a turn, and its card says why.

## Measured (greedy, 128-token random prompt, Release build)

| | iPhone 17 Pro decode | iPhone 17 Pro prefill | M4 Max decode |
| --- | --- | --- | --- |
| MiniCPM5-2B int8 | 22.4 tok/s | 27.3 tok/s | 127.6 tok/s |
| MiniCPM5-1B int8 | 61.7 tok/s | 65.6 tok/s | 246.6 tok/s |

Per-block-32 int8 lands on the Mac GPU's quantized-matmul path; the per-channel int8 sibling of the 2B decoded 5× slower on the Mac (25.6 tok/s) and the same on the phone, which is why both bundles ship per-block-32 scales.

## Swift API (CoreAIKit)

[CoreAIKit](https://github.com/john-rocky/coreai-kit) is a community Swift package over Core AI: it downloads a catalog model from Hugging Face on first use, caches it in Application Support, and exposes a chat session. Add the package in Xcode (**File → Add Package Dependencies…**, exact version `0.4.1`, product **CoreAIKit**), or in `Package.swift`:

```swift
.package(url: "https://github.com/john-rocky/coreai-kit", exact: "0.4.1")
// target dependency:
.product(name: "CoreAIKit", package: "coreai-kit")
```

```swift
import CoreAIKit

let chat = try await ChatSession(catalog: "minicpm5-2b")     // or "minicpm5-1b"
let reply = try await chat.respond(to: "1+1=?")
print(reply)                                                  // final answer only

// Streaming: answer and reasoning arrive as separate events.
for try await event in await chat.streamResponse(to: "Explain GQA in one sentence.") {
    switch event {
    case .response(let delta): print(delta, terminator: "")
    case .thinking(let delta): _ = delta                       // the <think> trace, if you want it
    default: break
    }
}
```

- The session keeps the conversation history; call `respond(to:)` again for the next turn.
- `ChatSession.Configuration`: `temperature` (default 0.7; `nil` = greedy), `maxResponseTokens` (default 2048), `systemPrompt`. Version 0.4.1 exposes temperature only, not top-p.
- Thinking is on by default for MiniCPM5, as in the released chat template. The trace is surfaced as `.thinking` events and on `Message.thinking`; `respond(to:)` returns the answer without it. The trace alone can run several hundred tokens, so keep `maxResponseTokens` generous.
- `KitLanguageModel(model:)` plugs the same bundles into Apple's `FoundationModels` `LanguageModelSession`, with tool calling on ChatML-style models. See the kit README.

## Apple's own runtime package (no CoreAIKit)

The bundle also loads with Apple's Swift package from [apple/coreai-models](https://github.com/apple/coreai-models). Download the bundle folder, then:

```bash
hf download mlboydaisuke/MiniCPM5-2B-CoreAI --include "int8/*" --local-dir ./MiniCPM5-2B-CoreAI
# ./MiniCPM5-2B-CoreAI/int8/ holds metadata.json, the .aimodel, and tokenizer/
# 1B: hf download mlboydaisuke/MiniCPM5-1B-CoreAI --include "int8/*" --local-dir ./MiniCPM5-1B-CoreAI
```

```swift
import FoundationModels
import CoreAILanguageModels

let model = try await CoreAILanguageModel(resourcesAt: bundleFolderURL)   // the int8/ folder
let session = LanguageModelSession(model: model)
print(try await session.respond(to: "1+1=?"))
```

The same repo's CLI runs the folder on a Mac: `swift run -c release llm-runner --model ./MiniCPM5-2B-CoreAI/int8 --prompt "1+1=?"`.

## Building the bundle from your own checkpoint (advanced)

The export uses Apple's `coreai-torch` (`coreai.llm.export`) with two adjustments that the zoo's recipe applies for you:

- MiniCPM5 is a plain `LlamaForCausalLM`; the exporter's Mistral graph builder is architecturally identical (GQA, no qkv bias, no qk-norm, explicit `head_dim`), so the recipe maps `llama → mistral`.
- int8 weight-only symmetric quantization via `--compression-config` (per-block-32 scales), and `eos_token` set to `<|im_end|>` in the bundle's tokenizer.

```bash
git clone https://github.com/john-rocky/coreai-model-zoo && cd coreai-model-zoo
python3 conversion/zoo_convert.py show minicpm5-2b    # prints the exact export command (minicpm5-1b for the 1B)
python3 conversion/zoo_convert.py run  minicpm5-2b    # export_minicpm5.py --hf-id openbmb/MiniCPM5-2B --qconfig minicpm5_int8sym_b32.yaml
python3 cli/coreai_verify.py <bundle> -n 16           # greedy token check against the fp32 HF reference
python3 cli/coreai_verify.py <bundle> --chat no-think --prompt "Reply with only the number: 1+1=?" -n 16 --must-stop-within 16
#   ^ the end-of-turn check: the fp32 reference answers `2` and stops; a bundle that runs past that stop fails
```

Point `--hf-id` at a local fine-tuned checkpoint to convert your own weights. Keep per-block-32 scales: a per-channel int8 export of the 1B produced a bundle whose LM head scored every vocab id above ~65024 at zero through the engine, `<|im_end|>` included, while a short greedy parity check still passed — the stop check above is what catches it. Full notes: [`models/minicpm5-2b`](https://github.com/john-rocky/coreai-model-zoo/blob/main/models/minicpm5-2b/README.md) and [`models/minicpm5-1b`](https://github.com/john-rocky/coreai-model-zoo/blob/main/models/minicpm5-1b/README.md) in the zoo.

## Common pitfalls

- **Toolchain**: macOS 27 / iOS 27 and Xcode 27 are required. CoreAIKit 0.4.1 is pinned to Xcode 27 beta 5; if `swift` picks another Xcode, export `DEVELOPER_DIR=/Applications/Xcode-27.0.0-Beta.5.app/Contents/Developer` first.
- **First launch on iPhone**: the bundle specializes on the phone once (2B: 28.9 s on iPhone 17 Pro; 1B: 7.3 s), then the cache persists. The 2B needs about 3 GB of free storage and the `com.apple.developer.kernel.increased-memory-limit` entitlement (ChatDemo has it). A full phone fails with `No space left on device`.
- **Context on iPhone**: prompt + generated tokens must stay under 1024 on iOS (the shipped pipelined engine caps growing-KV capacity there). Trim or chunk the history on the phone; macOS has no cap.
- **Debug builds** are about 3× slower per token on host-side work. Measure in Release.
- **macOS App Sandbox**: enable **Outgoing Connections (Client)** for the first-use download.
- **Model runs past the end of turn**: not expected with these bundles (the 1B from revision b8a6ac397ccd5fb815f97336f8a8b1800b110da1 on), since `eos_token` is `<|im_end|>` and the stop was checked end to end. If you re-export yourself, keep that setting, keep per-block-32 scales, and run the stop check above (`Reply with only the number: 1+1=?` should end at `2`).
