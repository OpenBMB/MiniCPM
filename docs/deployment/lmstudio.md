# Deploy MiniCPM5-1B with LM Studio

[LM Studio](https://lmstudio.ai/) is the **GUI-first** path for running MiniCPM5-1B on a Mac / Windows / Linux laptop — drag a model in, click *Load*, chat, or expose an OpenAI-compatible REST endpoint with one toggle. On Apple Silicon it ships **two** inference runtimes that both work for MiniCPM5-1B:

| Runtime | Format | Version | When to use |
| --- | --- | --- | --- |
| `llama.cpp-mac-arm64-apple-metal` | **GGUF** | 2.14.0 | cross-platform, same artifact as Ollama |
| `mlx-llm-mac-arm64-apple-metal` | **MLX** | 1.6.0 | Apple Silicon only — **~60% faster** at 4-bit, automatic think/answer split via OpenAI `reasoning_content` |

LM Studio does **not** load raw Hugging Face `transformers` checkpoints directly. To use the MLX runtime, point LM Studio at a pre-converted MLX repo, or run `mlx_lm.convert` first — see [`mlx.md`](./mlx.md) for the full pipeline.

## TL;DR

```bash
# 1. Install LM Studio (one-time) and run it once to complete onboarding.
brew install --cask lm-studio
open -a "LM Studio"          # accept EULA + pick model source

# 2. Drop the released GGUF into LM Studio's model registry.
#    The expected layout is <publisher>/<repo>/<file>.gguf:
mkdir -p ~/.lmstudio/models/openbmb/MiniCPM5-1B-GGUF
cp MiniCPM5-1B-Q4_K_M.gguf ~/.lmstudio/models/openbmb/MiniCPM5-1B-GGUF/

# 3. Use the bundled `lms` CLI to start the local OpenAI-compatible server.
LMS="/Applications/LM Studio.app/Contents/Resources/app/.webpack/lms"
"$LMS" ls
"$LMS" server start                                   # binds 127.0.0.1:1234
"$LMS" load minicpm5-1b --gpu max --context-length 8192 -y
"$LMS" ps
```

> 💡 The very first launch needs the GUI — `lms` will refuse with `Cannot find LM Studio installation` until LM Studio has run interactively at least once. Subsequent sessions are fully scriptable.

## Importing the model

LM Studio scans `~/.lmstudio/models/` recursively for `*.gguf`, **but only displays models whose path matches `<publisher>/<model>/<file>.gguf`**. Two equivalent options:

```bash
# Option A — drop into the registry directly (recommended for scripted envs)
mkdir -p ~/.lmstudio/models/openbmb/MiniCPM5-1B-GGUF
cp MiniCPM5-1B-Q4_K_M.gguf ~/.lmstudio/models/openbmb/MiniCPM5-1B-GGUF/

# Option B — drag-and-drop into the LM Studio GUI's "My Models" view.
```

After either path, `lms ls` should show:

```text
LLM              PARAMS    ARCH     SIZE         DEVICE
minicpm5-1b    1B      Llama    688.07 MB    Local
```

## Loading and serving

```bash
LMS="/Applications/LM Studio.app/Contents/Resources/app/.webpack/lms"

"$LMS" server start
"$LMS" load minicpm5-1b \
    --gpu max \
    --context-length 8192 \
    -y
```

`--gpu max` offloads every transformer layer to Metal; the M4's unified memory model means you don't need to budget VRAM separately. Reduce to `--gpu 0.5` or similar on machines where you want to leave more memory for other apps.

Verify with:

```bash
"$LMS" ps
# IDENTIFIER       MODEL            STATUS    SIZE         CONTEXT    PARALLEL    DEVICE    TTL
# minicpm5-1b    minicpm5-1b    IDLE      688.07 MB    8192       4           Local
```

## Inference

LM Studio exposes an OpenAI-compatible API on `http://localhost:1234/v1`. The chat template baked into the GGUF is auto-applied:

### No-think (fast)

```bash
curl -sS http://localhost:1234/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "minicpm5-1b",
        "messages": [{"role": "user", "content": "用一句话解释什么是 GQA。"}],
        "temperature": 0.7, "top_p": 0.95, "max_tokens": 200,
        "stop": ["<|im_end|>", "<|im_start|>"]
    }'
```

### Think (reasoning)

```bash
curl -sS http://localhost:1234/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "minicpm5-1b",
        "messages": [{"role": "user", "content": "鸡兔同笼，头共10个，脚共28只，问鸡和兔各几只？请逐步推理后给出答案。"}],
        "temperature": 0.9, "top_p": 0.95, "max_tokens": 400,
        "stop": ["<|im_end|>", "<|im_start|>"]
    }'
```

> 💡 On Apple Silicon the **MLX Q4 build is the clear winner**: ~60% faster than the equivalent GGUF Q4_K_M, reasoning-tokens are routed to OpenAI's `reasoning_content` extension (so the visible `content` only contains the final answer), and the math answer matches F16 quality. Use the GGUF build only when cross-platform parity matters.

### Auto think/answer split (MLX runtime only)

When LM Studio's `mlx-llm` runtime detects a `<think>...</think>` block in the model's output, it splits the OpenAI response automatically:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "鸡有6只，兔子有4只 …",
      "reasoning_content": "我们被问到：…设鸡的数量为x …"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "completion_tokens": 1428,
    "completion_tokens_details": {"reasoning_tokens": 1147}
  }
}
```

The GGUF runtime does not do this split — `<think>...</think>` is emitted inline as part of `content`, and you have to strip it client-side. If your downstream client supports the OpenAI reasoning extension (Cursor, Continue, etc.), prefer the MLX runtime.

> 💡 Always pass an explicit `stop` array including both `<|im_end|>` and `<|im_start|>`. LM Studio's bundled chat template emits the assistant turn correctly, but its OpenAI-compat layer does not auto-attach the `<|im_end|>` stop string, so without it the model will sometimes continue past the EOT marker into a second hallucinated turn.

## Recommended sampling

| Mode | `temperature` | `top_p` | When to use |
| --- | --- | --- | --- |
| Think | 0.9 | 0.95 | reasoning, math, code (model auto-emits a `<think>` block) |
| No-think | 0.7 | 0.95 | latency-bound assistant |

## Toggling Think / No-think

As of 2026-05, LM Studio's OpenAI-compatible layer does not consistently honour the standard `chat_template_kwargs.enable_thinking` flag. Passing it in the request body may have no effect, and natural-language hints (`/no_think` suffix, system prompt saying "do not include reasoning", etc.) do not reliably suppress thinking either. So in the GUI chat pane there may be no built-in toggle.

The cleanest workaround is to **register two extra model variants with the `enable_thinking` switch hard-coded into the Jinja template**, then switch modes by switching model in the GUI's model dropdown. Variant folders symlink the weights, so each variant adds **~10 KB on disk** (one rewritten `chat_template.jinja`).

### One-time setup script

```bash
python3 - <<'PY'
import os, re, shutil

ROOT = os.path.expanduser("~/.lmstudio/models/openbmb")
SOURCES = ["MiniCPM5-1B-MLX-Q4", "MiniCPM5-1B-MLX-bf16"]

block_re = re.compile(
    r"\{%-\s*if\s+enable_thinking\s+is\s+defined\s*%\}.*?"
    r"\{%-\s*endif\s*%\}\s*\{%-\s*endif\s*%\}",
    re.DOTALL,
)

def make_variant(src, dst, mode):
    if os.path.exists(dst): shutil.rmtree(dst)
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        sp, dp = os.path.join(src, f), os.path.join(dst, f)
        if f == "chat_template.jinja":
            tpl = open(sp).read()
            replacement = (
                "{{- '<think>\\n\\n</think>\\n\\n' }}"  # nothink
                if mode == "nothink"
                else "{{- '<think>\\n' }}"             # think
            )
            new_tpl, n = block_re.subn(replacement, tpl, count=1)
            assert n == 1, "enable_thinking block not found in template"
            open(dp, "w").write(new_tpl)
        else:
            os.symlink(os.path.realpath(sp), dp)

for src_name in SOURCES:
    src = f"{ROOT}/{src_name}"
    make_variant(src, f"{ROOT}/{src_name}-NoThink", "nothink")
    make_variant(src, f"{ROOT}/{src_name}-Think",   "think")
print("done — restart LM Studio (or run `lms ls`) to re-scan the registry.")
PY
```

After running it, `lms ls` shows four extra rows alongside the originals:

```text
LLM                               PARAMS    ARCH     SIZE         DEVICE
minicpm5-1b-mlx@4bit            1B      Llama    617.97 MB    Local      ← jinja default (think)
minicpm5-1b-mlx-nothink@4bit    1B      Llama    617.97 MB    Local      ← always no-think
minicpm5-1b-mlx-think@4bit      1B      Llama    617.97 MB    Local      ← always think
…and bf16 counterparts
```

In the GUI chat pane just pick the variant you want from the model dropdown — the chat template baked into each variant decides whether `<think>` activates.

### Variant behaviour example (`用一句话告诉我中国首都。`, `max_tokens=120`)

| Variant | `reasoning_tokens` | `content` (first 80 chars) |
| --- | ---: | --- |
| `…-nothink@4bit` | **0** | `好的,让我来为你介绍中国的首都: 1. 北京 - 中国的政治、文化中心 …` |
| `…-think@4bit` | 119 | `(empty — reasoning hit max_tokens)` |
| Default `…@4bit` (no override) | 94 | `中国首都为北京。` (reasoning routed to `reasoning_content`) |

> ⚠️ Don't try to write the assistant pre-fill trick (`{"role":"assistant","content":"<think>\n\n</think>\n\n"}`) from the GUI — LM Studio's chat pane doesn't allow appending an assistant turn before send. The variant approach is the only path that works in pure-GUI use. From the OpenAI HTTP API, the assistant pre-fill **also** works and avoids needing extra variants.

## Q&A

### `lms` says `Cannot find LM Studio installation`

You haven't completed LM Studio's first-run GUI onboarding yet. Open the app from Launchpad, accept the EULA + pick a model source, then re-run the CLI command.

### `lms import` hangs with no output

Skip `lms import` and use the registry-drop path (Option A above) — `cp` the GGUF into `~/.lmstudio/models/<pub>/<repo>/`. LM Studio picks it up within a few seconds without any further command.

### Output looks like the assistant impersonates the user / continues past one turn

You forgot the explicit `stop` array — see the sanity-check snippets above.

## See also

- [`ollama.md`](./ollama.md) — the same GGUF, CLI / daemon path
- [`llama_cpp.md`](./llama_cpp.md) — the underlying engine; build / quantize recipe
- [`mlx.md`](./mlx.md) — alternative MLX path on Apple Silicon (no GUI)
