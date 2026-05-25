---
name: minicpm5-finetune-unsloth
description: Fine-tune MiniCPM5-1B with unsloth for tight-VRAM single-GPU LoRA / QLoRA. Use when the user wants "unsloth", "FastLanguageModel", QLoRA on a 24 GB consumer GPU, or asks for the smallest VRAM footprint.
---

# Fine-tune MiniCPM5-1B with unsloth

Single-GPU LoRA / QLoRA. Heavy custom kernels for memory savings (~2× reduction at 4-bit).

> ⚠️ **Two install-time pins (read this BEFORE running)**:
> 1. **`transformers==4.57.3`** — required if vLLM is in the same env (unsloth's vLLM coexistence patch only handles transformers ≤ 4.57.x).
> 2. **`torch==2.7.1` + `torchvision==0.22.1`** — `pip install unsloth` may pull a cu13 torch wheel that fails on cu12.x drivers (`cuda.is_available()` returns False).
>
> ```bash
> pip install --force-reinstall "torch==2.7.1" "torchvision==0.22.1" "transformers==4.57.3"
> ```

## Required input

| Var | Example | Default |
| --- | --- | --- |
| `BASE_MODEL` | `openbmb/MiniCPM5-1B` | required |
| `DATA` | path to messages-format jsonl | required |
| `OUTPUT_DIR` | `./runs/minicpm5_unsloth` | required |
| `LOAD_IN_4BIT` | `True` (QLoRA, lowest VRAM) / `False` (LoRA bf16) | `False` |

## Steps

### 1. Install (once) — pin transformers to 4.57.3

```bash
pip install "unsloth>=2026.5"
pip install --force-reinstall "transformers==4.57.3"
```

> 🔑 **`transformers==4.57.3` is required** if vLLM is in the same env. unsloth's vLLM coexistence patch only handles transformers ≤ 4.57.x; if you skip this pin you'll get a `dataclass` error at import time.

### 2. Train — save as `train_unsloth.py`

```python
import json, os, torch
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

BASE = os.environ["BASE_MODEL"]
DATA = os.environ["DATA"]
OUT  = os.environ["OUTPUT_DIR"]

# 1. Load with unsloth's fast wrapper
model, tok = FastLanguageModel.from_pretrained(
    model_name=BASE,
    max_seq_length=2048,
    dtype=torch.bfloat16,
    load_in_4bit=os.environ.get("LOAD_IN_4BIT", "False") == "True",
    full_finetuning=False,
)

# 2. Add LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# 3. Data — unsloth's TRL wrapper expects a `text` column
rows = [json.loads(l) for l in open(DATA, encoding="utf-8") if l.strip()]
ds = Dataset.from_list([
    {"text": tok.apply_chat_template(r["messages"], tokenize=False)}
    for r in rows
])
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# 4. Train
trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir=OUT,
        dataset_text_field="text",                 # 🔑 needed for unsloth's wrapper
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=True,
        max_length=2048,
        packing=False,
        logging_steps=10,
        save_steps=200,
        seed=42,
        report_to="none",
        dataloader_num_workers=2,
    ),
    train_dataset=ds,
    processing_class=tok,
)
trainer.train()
trainer.model.save_pretrained(f"{OUT}/adapter_final")
```

### 3. Run

```bash
BASE_MODEL=openbmb/MiniCPM5-1B \
DATA=/path/to/messages.jsonl \
OUTPUT_DIR=./runs/minicpm5_unsloth \
LOAD_IN_4BIT=False \
python train_unsloth.py
```

For QLoRA on a consumer GPU (24 GB or less), set `LOAD_IN_4BIT=True`.

### 4. Validate

You should see:

```
==((====))== Unsloth: Fast Llama patching. Transformers: 4.57.3.
🦥 Unsloth: Padding-free auto-enabled, enabling faster training.
trainable params: 11,206,656 of 1,091,839,488 (1.03 % trained)

{'loss': 4.67, 'epoch': 0.2}
{'loss': 3.52, 'epoch': 1.0}
```

First iter is slow (~13 s, JIT compile); subsequent iters < 600 ms.

## Inference

```python
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained(
    model_name="./runs/minicpm5_unsloth",       # path to adapter
    max_seq_length=4096, dtype=torch.bfloat16, load_in_4bit=False,
)
FastLanguageModel.for_inference(model)           # 2× faster generation

inputs = tok.apply_chat_template([{"role":"user","content":"用一句话解释 GQA。"}],
                                 add_generation_prompt=True, enable_thinking=True, return_tensors="pt").to("cuda")
out = model.generate(inputs, max_new_tokens=512, temperature=0.9, top_p=0.95)
print(tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True))
```

## Merge for serving

```python
model.save_pretrained_merged("./minicpm5-unsloth-merged", tok)
```

## Common pitfalls

- **`RuntimeError: vLLM with version 0.10.x does not yet support transformers>=5.0.0`**: pin `transformers==4.57.3` (see install step).
- **`Unsloth: You must specify a formatting_func`**: forgot `dataset_text_field="text"` in `SFTConfig` AND/OR forgot to pre-tokenize messages → text column.

## Reference

[`docs/finetune/unsloth.md`](../../docs/finetune/unsloth.md)
