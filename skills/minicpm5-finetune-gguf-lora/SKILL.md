---
name: minicpm5-finetune-gguf-lora
description: Fine-tune MiniCPM5-1B into a LoRA adapter and convert it to a GGUF adapter that loads directly into llama.cpp / llama-server and the MiniCPM Desk Pet app's custom-LoRA upload. Use when the user wants "GGUF LoRA", "convert LoRA to GGUF", "convert_lora_to_gguf", a custom persona/skin for the desktop pet, "桌宠自定义 LoRA", "上传 LoRA 到桌宠", or asks how to take a trained adapter and run it on a GGUF base.
---

# Fine-tune MiniCPM5-1B → GGUF LoRA adapter

The framework skills (`minicpm5-finetune-*`) all emit a **PEFT adapter** (`adapter_model.safetensors` + `adapter_config.json`). But `llama.cpp` / `llama-server` — and the **MiniCPM Desk Pet** app's custom-LoRA upload — load a **GGUF LoRA adapter** (`--lora some-adapter.gguf`). This skill is the bridge: train a PEFT LoRA, then convert it to GGUF and run/upload it.

> This is the path you want when the base model is served as GGUF (Desk Pet, Ollama, LM Studio, plain `llama-server`). If you serve the fp16 HF base with vLLM / transformers, you don't need GGUF — load the PEFT adapter directly.

## Pipeline overview

```
  train (any minicpm5-finetune-* skill)        this skill
 ┌────────────────────────────────────┐   ┌──────────────────────────────────┐
 BASE (fp16 HF) ─► adapter_model.safetensors ─► convert_lora_to_gguf.py ─► adapter.gguf
                   adapter_config.json                                          │
                                                                  llama-server --lora / Desk Pet upload
```

## Required input

| Var | Example | Default |
| --- | --- | --- |
| `ADAPTER_DIR` | `./runs/minicpm5_unsloth/adapter_final` | required — a PEFT dir with `adapter_config.json` + `adapter_model.safetensors` |
| `BASE_MODEL` | `openbmb/MiniCPM5-1B` (HF id) **or** a local fp16 HF dir | required — must be the **same base** the adapter was trained on |
| `OUTTYPE` | `f16` (recommended) / `q8_0` / `bf16` / `f32` | `f16` |
| `OUT_GGUF` | `./minicpm5-mylora.gguf` | `<ADAPTER_DIR>/adapter_model.f16.gguf` |

> **Don't have an adapter yet?** First run a training skill — start from the router **`minicpm5-finetune`** (or go straight to **`minicpm5-finetune-unsloth`** for single-GPU LoRA). Come back here with `ADAPTER_DIR` pointing at its output.

## Steps

### 1. Get llama.cpp (has the converter)

```bash
git clone --depth=1 https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
pip install -r requirements.txt          # converter deps: torch, safetensors, gguf, transformers
```

`convert_lora_to_gguf.py` lives at the repo root. You only need the Python script for conversion; build the C++ binaries (step 4) only if you want to test locally.

### 2. Convert the PEFT adapter → GGUF

```bash
python convert_lora_to_gguf.py "$ADAPTER_DIR" \
    --base "$BASE_MODEL" \
    --outtype f16 \
    --outfile "$OUT_GGUF"
```

- `--base` only needs the base model's **config** (`config.json`, `tokenizer.json`) — not the weights. A local fp16 HF dir or an HF id both work.
- If the base is only on the Hub, use `--base-model-id openbmb/MiniCPM5-1B` instead of `--base` (it pulls just the config).
- Output is a single `.gguf` of roughly the same size as the input `.safetensors` (a 22 MB r=16 adapter → ~22 MB GGUF).

> 🔑 **The #1 gotcha — `base_model_name_or_path` points at the training machine.**
> PEFT writes the absolute path of the base used during training into `adapter_config.json`
> (e.g. `/user/.../MiniCPM5-models-fixed/official`). On any other machine the converter
> can't find it. **Always pass `--base` (or `--base-model-id`) explicitly** to override it —
> don't rely on whatever is baked into the config. The base you pass MUST match the one you
> trained on, or the adapter math is meaningless.

### 3. (Optional) Sanity-check the GGUF metadata

```bash
python -c "import gguf,sys; r=gguf.GGUFReader('$OUT_GGUF'); print('tensors:', len(r.tensors)); print('arch:', r.get_field('general.architecture').parts[-1].tobytes().decode() if r.get_field('general.architecture') else '?')"
```

Expect `general.type = adapter` and a nonzero tensor count. MiniCPM5-1B is a **Llama-architecture** model, so the converter treats it as `llama` — this is correct, not an error.

### 4. Test with llama-server before shipping

You need a GGUF **base** model too (the adapter is applied on top of it). Grab the released base:

```bash
huggingface-cli download openbmb/MiniCPM5-1B-GGUF MiniCPM5-1B-Q8_0.gguf --local-dir .
```

Then:

```bash
# CLI
llama-cli -m MiniCPM5-1B-Q8_0.gguf --lora "$OUT_GGUF" \
    -p "你好" -n 128 --temp 0.7 --top-p 0.95

# OpenAI-compatible server
llama-server -m MiniCPM5-1B-Q8_0.gguf --lora "$OUT_GGUF" --port 8080 --jinja
```

If the persona/behavior you trained shows up, the GGUF is good. If output is identical to the base, the adapter didn't load (check the `--lora` path and that `--base` matched in step 2).

> **Quant compatibility:** a GGUF LoRA built from an fp16 adapter applies fine on top of a
> quantized base (Q8_0 / Q4_K_M). You do **not** need a separate adapter per base quant.

### 5. Use it in MiniCPM Desk Pet

The Desk Pet app loads exactly this kind of file. The fp16 GGUF you just built is upload-ready:

1. Open **Settings → MiniCPM → 适配器 (LoRA)**.
2. Click **上传** (Upload), pick your `.gguf` — the app copies it into its
   `<userData>/adapters/uploads/` directory and registers it.
3. Give it a display name + comma-separated aliases (used so you can switch personas by
   voice/chat, e.g. "换成 XX").
4. Select it in the adapter list to activate. The sidecar reloads `llama-server` with your
   `--lora`.

Constraints the app enforces (match them or the upload is rejected):

- **Must be a single `.gguf` file** (the GGUF LoRA from step 2 — *not* the `.safetensors`,
  *not* a merged full model).
- It is applied on top of whatever MiniCPM5-1B GGUF base the app already runs, so **train
  against the MiniCPM5-1B base**, not some other model.

## Common pitfalls

- **`FileNotFoundError` / `can't load base model config`** during conversion → the
  `adapter_config.json` base path doesn't exist locally. Fix with `--base <local-dir>` or
  `--base-model-id openbmb/MiniCPM5-1B` (step 2 gotcha).
- **Adapter loads but output is unchanged** → base mismatch (trained on base A, applied on
  base B), or the persona needs its **system prompt** too — a LoRA biases style but the
  system prompt still matters. In Desk Pet the persona's system prompt is wired to the
  adapter; standalone `llama-cli` users must pass it themselves.
- **`KeyError` on an unknown tensor / target module** → the adapter targeted modules the
  converter doesn't map. Stick to the standard attention+MLP projections
  (`q/k/v/o_proj`, `gate/up/down_proj`) when training (all `minicpm5-finetune-*` skills
  already default to these).
- **Uploaded `.gguf` but Desk Pet shows nothing** → you uploaded the merged full model or a
  base GGUF by mistake. A LoRA adapter GGUF is small (tens of MB) and has
  `general.type = adapter` (step 3).

## When NOT to use

- Serving the fp16 HF base with **vLLM / transformers / SGLang** → load the PEFT adapter
  directly, skip GGUF. See `minicpm5-deploy-vllm` / `-transformers` / `-sglang`.
- You want to **bake the LoRA into the weights** (no runtime `--lora`) → merge first
  (`model.save_pretrained_merged(...)` or `peft ... merge_and_unload()`), then convert the
  **merged fp16 model** with `convert_hf_to_gguf.py` (see the "Building your own GGUF"
  section of `minicpm5-deploy-llama-cpp`). That produces a standalone base GGUF, not an
  adapter.

## Reference

- Training frameworks: **`minicpm5-finetune`** (router) and its sub-skills.
- Running GGUF: **`minicpm5-deploy-llama-cpp`**.
- Converter: `convert_lora_to_gguf.py` in [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp).
