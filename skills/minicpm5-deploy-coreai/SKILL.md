---
name: minicpm5-deploy-coreai
description: Run MiniCPM5-2B or MiniCPM5-1B on-device inside a Swift app with Apple Core AI (iOS 27 / macOS 27). Use when the user targets an iPhone, iPad or Mac app, or says "iOS", "iPhone", "Swift", "Xcode", "Core AI", "CoreAIKit", ".aimodel", "on-device in my app", or wants a Foundation Models LanguageModelSession backed by MiniCPM5.
---

# Deploy MiniCPM5-2B / MiniCPM5-1B with Apple Core AI (iPhone / iPad / Mac)

Apple's on-device runtime in iOS 27 / macOS 27. The model is an `.aimodel` bundle loaded from Swift; no Python at runtime. The bundles are community conversions from the [Core AI Model Zoo](https://github.com/john-rocky/coreai-model-zoo) (not an OpenBMB or Apple release). Covers MiniCPM5-2B and MiniCPM5-1B. Human-readable reference: [`docs/deployment/coreai.md`](../../docs/deployment/coreai.md).

## Required input

| Var | Example | Default |
| --- | --- | --- |
| `MODEL` | `minicpm5-2b` or `minicpm5-1b` (CoreAIKit catalog id) | `minicpm5-2b` |
| `TARGET` | `mac` (headless CLI) or `iphone` (Xcode app) | `mac` |
| `PROMPT` | `1+1=?` | `1+1=?` |

Bundles: [`mlboydaisuke/MiniCPM5-2B-CoreAI`](https://huggingface.co/mlboydaisuke/MiniCPM5-2B-CoreAI) (int8, 2.7 GB) and [`mlboydaisuke/MiniCPM5-1B-CoreAI`](https://huggingface.co/mlboydaisuke/MiniCPM5-1B-CoreAI) (int8, 1.1 GB; revision b8a6ac397ccd5fb815f97336f8a8b1800b110da1 or newer). Download on first use.

## Steps

### 0. Check the toolchain (once)

macOS 27 and Xcode 27 are required; CoreAIKit 0.4.1 is pinned to Xcode 27 beta 5. iPhone runs need iOS 27 on the device.

```bash
xcodebuild -version            # expect Xcode 27
# If another Xcode is selected:
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-Beta.5.app/Contents/Developer
```

### 1. Get the runner

```bash
git clone https://github.com/john-rocky/coreai-kit
cd coreai-kit/Examples/ChatDemo
```

### 2A. `TARGET=mac`: headless run

```bash
swift run -c release chat-cli --model "${MODEL}" --prompt "${PROMPT}"
```

Progress (download, load) goes to stderr; stdout carries only the final answer, so assert on stdout. First run downloads the bundle into Application Support; later runs load from cache.

### 2B. `TARGET=iphone`: run the app on the device

```bash
open ChatDemo.xcodeproj
```

In Xcode: choose the iPhone as the run destination, set your signing team under **Signing & Capabilities**, Run, then pick "MiniCPM5 2B" or "MiniCPM5 1B" in the model picker. The app already carries the `com.apple.developer.kernel.increased-memory-limit` entitlement the bundle needs.

### 2C. Integrate into your own app

Add the package (Xcode: **File → Add Package Dependencies…**, `https://github.com/john-rocky/coreai-kit`, exact `0.4.1`, product **CoreAIKit**), then:

```swift
import CoreAIKit

let chat = try await ChatSession(catalog: "minicpm5-2b")     // or "minicpm5-1b"
let reply = try await chat.respond(to: "1+1=?")               // answer only; thinking is separate
```

`chat.streamResponse(to:)` yields `.response` and `.thinking` deltas. `ChatSession.Configuration` has `temperature` (default 0.7; `nil` = greedy), `maxResponseTokens` (default 2048) and `systemPrompt`; 0.4.1 exposes no top-p.

### 3. Validate

stdout (or the app's reply) contains `2` for `1+1=?`, and generation stops on its own. Measured on an M4 Max with the bundle already cached: the 2B answers `1 + 1 = 2.` in 18 s wall time including engine load; Apple's `llm-runner` on the same bundles thinks, answers `2`, and stops at `<|im_end|>` (2B after 190 tokens, 1B after 171).

## Without CoreAIKit (Apple's own package)

```bash
hf download mlboydaisuke/MiniCPM5-2B-CoreAI --include "int8/*" --local-dir ./MiniCPM5-2B-CoreAI
# 1B: hf download mlboydaisuke/MiniCPM5-1B-CoreAI --include "int8/*" --local-dir ./MiniCPM5-1B-CoreAI
```

```swift
import FoundationModels
import CoreAILanguageModels          // Swift package: https://github.com/apple/coreai-models

let model = try await CoreAILanguageModel(resourcesAt: folderURL)   // the int8/ folder
let session = LanguageModelSession(model: model)
print(try await session.respond(to: "1+1=?"))
```

Mac CLI from the same repo: `swift run -c release llm-runner --model ./MiniCPM5-2B-CoreAI/int8 --prompt "1+1=?"`.

## Common pitfalls

- **Wrong Xcode selected**: `swift` builds against whichever Xcode `xcode-select` points at. Set `DEVELOPER_DIR` as in step 0.
- **First launch on iPhone**: one-time on-device specialization (2B: 28.9 s on iPhone 17 Pro; 1B: 7.3 s), then cached. The 2B needs about 3 GB free storage; a full phone fails with `No space left on device`.
- **1B revision**: use `b8a6ac397ccd5fb815f97336f8a8b1800b110da1` or newer. The earlier 1B revision (`5ad650f`) did not stop at the end of a turn.
- **Context on iPhone**: prompt + generated tokens must stay under 1024 on iOS. Trim or chunk history on the phone; macOS has no cap.
- **Thinking is on by default** (as in the released chat template) and can run several hundred tokens before the answer. Keep `maxResponseTokens` generous; read the trace from `.thinking` events if needed.
- **Debug builds** are about 3× slower per token. Measure in Release (`-c release`).
- **macOS App Sandbox**: enable **Outgoing Connections (Client)** for the first-use download.

## When NOT to use

- Python on a Mac, no app → `minicpm5-deploy-mlx` (fastest on Apple Silicon in Python)
- Desktop GUI, no code → `minicpm5-deploy-lmstudio`
- Linux / Windows / CPU-only → `minicpm5-deploy-llama-cpp`
- Server with an OpenAI-compatible endpoint → `minicpm5-deploy-vllm` or `minicpm5-deploy-sglang`
