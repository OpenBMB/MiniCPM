---
name: minicpm5-finetune
description: Pick the right fine-tuning framework for a MiniCPM5-1B base checkpoint and route to a framework-specific cookbook skill. Use when the user wants to SFT / LoRA / DPO / continue-pretrain MiniCPM5 and has not yet committed to a specific framework, or when they say "fine-tune MiniCPM5", "train MiniCPM5", "MiniCPM5 微调", "LoRA MiniCPM5", "继续训练 MiniCPM5".
---

# Fine-tune MiniCPM5-1B — framework router

You're being asked to fine-tune MiniCPM5-1B. Pick exactly one framework skill below and **invoke that skill** rather than improvising — every framework has at least one MiniCPM5-specific gotcha that the dedicated skill knows about.

## 1. Required input from the user

| Variable | Example | Notes |
| --- | --- | --- |
| `BASE_MODEL` | HF id `openbmb/MiniCPM5-1B` (post-release) **or** a local path | see cluster paths below |
| `DATA` | path to JSONL in messages format | `[{"messages": [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]}]` |
| `OUTPUT_DIR` | where to write checkpoints | mkdir if missing |
| Goal | "LoRA SFT" / "full SFT" / "DPO" / "QLoRA on consumer GPU" / "continue-pretrain at scale" | drives skill choice |
| Hardware | 1× H100/H200 / 1× consumer GPU / multi-node | drives skill choice |

### Default base model

If `BASE_MODEL` is not pinned, default to the Hugging Face fp16 release:

```
openbmb/MiniCPM5-1B
```

Any local directory containing `config.json` + `model.safetensors` + `tokenizer.json` also works.

## 2. Decision matrix — pick exactly one

| User says / wants | Best fit | → Skill to invoke |
| --- | --- | --- |
| YAML / WebUI driven SFT, broad community support | LLaMA-Factory | **`minicpm5-finetune-llamafactory`** |
| ChatML template + ModelScope-native SFT/DPO/KTO/ORPO | ms-swift | **`minicpm5-finetune-ms-swift`** |
| Bare-metal Python, assistant-only loss, minimal abstractions | TRL + PEFT | **`minicpm5-finetune-trl`** |
| Single-GPU LoRA / QLoRA, tight VRAM (24 GB or less) | unsloth | **`minicpm5-finetune-unsloth`** |
| mmengine config-driven SFT, OpenMMLab stack | xtuner | **`minicpm5-finetune-xtuner`** |

### Decision shortcuts

- **First time fine-tuning MiniCPM5**: pick **`minicpm5-finetune-llamafactory`** — most documented, fewest surprises.
- **Need DPO / KTO / ORPO**: pick **`minicpm5-finetune-ms-swift`** (best out-of-the-box) or **`minicpm5-finetune-trl`** (most control).
- **Single 24 GB consumer GPU**: pick **`minicpm5-finetune-unsloth`** with `load_in_4bit=True`.

## 3. Invocation contract

Each sub-skill expects `BASE_MODEL`, `DATA`, `OUTPUT_DIR` and outputs an LoRA adapter (or full checkpoint) at `OUTPUT_DIR/`. **Read the picked sub-skill in full before running** — every framework has at least one MiniCPM5-specific gotcha:

| Framework | Gotcha (skill handles it for you) |
| --- | --- |
| LLaMA-Factory | `template: empty` (delegate to model's own jinja, NOT `template: llama3`) |
| ms-swift | mandatory `--model_type llama --template chatml` flags |
| TRL | training-only chat template patch for `assistant_only_loss=True` |
| unsloth | `transformers==4.57.3` pin if vLLM is in the same env |
| xtuner | `prompt_template=PROMPT_TEMPLATE.qwen_chat` (ChatML), use `openai_map_fn` for messages-format data, `start_factor` ≥ 1e-2 |

## 4. Universal sanity check after training

Regardless of framework:

```bash
# 1. Verify adapter (or full ckpt) was saved
ls "$OUTPUT_DIR"        # adapter_model.safetensors + adapter_config.json (LoRA) OR a full HF directory

# 2. Quick inference check (HF-side, works for both LoRA and full)
python -c "
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained('$BASE_MODEL', torch_dtype=torch.bfloat16, device_map='auto').eval()
model = PeftModel.from_pretrained(base, '$OUTPUT_DIR').eval()    # skip this line for full SFT
tok = AutoTokenizer.from_pretrained('$BASE_MODEL')
inputs = tok.apply_chat_template([{'role':'user','content':'1+1=?'}], add_generation_prompt=True, enable_thinking=False, return_tensors='pt').to(model.device)
out = model.generate(inputs, max_new_tokens=32, do_sample=False)
print(tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True))
"
```

Expected: a coherent answer (e.g. `"2"`). If it's gibberish or empty, training broke (most likely chat template misalignment — ⮕ check the framework's gotcha row above).

## 5. Don't reinvent: link to the cookbook

Each sub-skill is paired with a one-page cookbook in [`docs/finetune/`](../../docs/finetune/). The skill is the machine-readable shortcut; the cookbook is the human-readable reference.
