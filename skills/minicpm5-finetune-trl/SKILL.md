---
name: minicpm5-finetune-trl
description: Fine-tune MiniCPM5-1B with bare-metal TRL + PEFT, including assistant-only loss via a chat-template patch. Use when the user wants minimal Python, no YAML, full control, or asks for "TRL", "SFTTrainer", "PEFT", "LoraConfig", "assistant_only_loss".
---

# Fine-tune MiniCPM5-1B with TRL + PEFT

Bare-metal Python recipe with **assistant-only loss mask**. Minimal abstractions, full control.

> ⚠️ **Driver-aware torch pin**: if `nvidia-smi` shows `CUDA Version: 12.x` and you let pip resolve `torch` freely, you may land on a cu13 wheel that fails to use the GPU. The install snippet below explicitly pins `torch==2.7.1` (cu126) which works on cu12.x drivers.

## Required input

| Var | Example | Default |
| --- | --- | --- |
| `BASE_MODEL` | `openbmb/MiniCPM5-1B` | required |
| `DATA` | path to messages-format jsonl | required |
| `OUTPUT_DIR` | `./runs/minicpm5_trl` | required |

## Steps

### 1. Install (once)

```bash
pip install "torch==2.7.1" "torchvision==0.22.1" \
            "trl>=0.18" "peft>=0.11" "transformers>=4.51" \
            datasets accelerate
```

### 2. Patch the tokenizer with a training-only chat template

Save as `train_lora.py`:

```python
import json, os, torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

BASE = os.environ["BASE_MODEL"]
DATA = os.environ["DATA"]
OUT  = os.environ["OUTPUT_DIR"]

# Training-only chat template — adds {% generation %} so SFTConfig(assistant_only_loss=True)
# masks all non-assistant tokens. Token sequence stays identical to the model's full chat
# template, so the trained adapter is fully compatible at inference time.
TRAIN_CHAT_TEMPLATE = (
    "{{- bos_token }}"
    "{%- for message in messages %}"
    "{%- if message['role'] == 'system' %}"
    "{{- '<|im_start|>system\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{%- elif message['role'] == 'user' %}"
    "{{- '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{%- elif message['role'] == 'assistant' %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- generation %}"
    "{{- message['content'] + '<|im_end|>' }}"
    "{%- endgeneration %}"
    "{{- '\\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- endif %}"
)

set_seed(42)

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.chat_template = TRAIN_CHAT_TEMPLATE       # 🔑 do NOT save back to disk

rows = [json.loads(l) for l in open(DATA, encoding="utf-8") if l.strip()]
ds = Dataset.from_list([{"messages": r["messages"]} for r in rows])

model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir=OUT,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=True,
        max_length=2048,
        packing=False,
        assistant_only_loss=True,                  # 🔑 only assistant tokens contribute to loss
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        report_to=["tensorboard"],
        dataloader_num_workers=2,
        remove_unused_columns=False,
        seed=42,
    ),
    train_dataset=ds,
    processing_class=tok,
)
trainer.train()
trainer.model.save_pretrained(f"{OUT}/adapter_final")
```

### 3. Train

```bash
BASE_MODEL=openbmb/MiniCPM5-1B \
DATA=/path/to/messages.jsonl \
OUTPUT_DIR=./runs/minicpm5_trl \
CUDA_VISIBLE_DEVICES=0 python train_lora.py
```

### 4. Validate

You should see:

```
trainable params: 11,206,656 || all params: 1,091,839,488 || trainable%: 1.0264
{'loss': 4.07, 'mean_token_accuracy': 0.29, 'epoch': 0.2}
{'loss': 3.52, 'mean_token_accuracy': 0.36, 'epoch': 1.0}
```

Adapter saved to `${OUTPUT_DIR}/adapter_final/adapter_model.safetensors`.

## Inference with the LoRA

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = AutoModelForCausalLM.from_pretrained("openbmb/MiniCPM5-1B", torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, "./runs/minicpm5_trl/adapter_final").eval()
tok = AutoTokenizer.from_pretrained("openbmb/MiniCPM5-1B")    # 🔑 reload original tokenizer for full chat_template

inputs = tok.apply_chat_template([{"role":"user","content":"用一句话解释 GQA。"}],
                                 add_generation_prompt=True, enable_thinking=True, return_tensors="pt").to(model.device)
out = model.generate(inputs, max_new_tokens=512, do_sample=True, temperature=0.9, top_p=0.95)
print(tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True))
```

> 🔑 **Always reload the original tokenizer at inference time** — the patched chat template is for training only. The LoRA adapter is fully compatible with the original chat template (token sequence is identical).

## Merge for serving

```python
merged = model.merge_and_unload()
merged.save_pretrained("./minicpm5-trl-merged")
tok.save_pretrained("./minicpm5-trl-merged")        # use the ORIGINAL tokenizer
```

## Common pitfalls

- **`SFTConfig has no attribute 'max_length'`**: your `trl` is too old. Need `trl>=0.18` (we tested 0.20).
- **Loss does not decrease**: `assistant_only_loss=True` requires the `{% generation %}` block in the chat template. If you forgot to set the patched template, TRL falls back to loss-over-all-tokens.

## Reference

[`docs/finetune/trl.md`](../../docs/finetune/trl.md)
