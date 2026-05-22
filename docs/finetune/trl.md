# Fine-tune MiniCPM5-1B with TRL + PEFT (bare-metal recipe)

A minimal Python recipe that uses the [TRL](https://github.com/huggingface/trl) `SFTTrainer` + [PEFT](https://github.com/huggingface/peft) `LoraConfig` directly, with an **assistant-only loss mask** delivered via a small chat-template patch.

## Verified versions

| Component | Version | Result |
| --- | --- | --- |
| `trl` | **0.20.0** | LoRA SFT ✅ loss 4.07 → 3.52 (200 samples / 1 epoch / H200) |
| `peft` | 0.11.1 | 11.2 M trainable / 1.09 B total (1.03%) |
| `transformers` | 4.57.1 | |
| `torch` | 2.7.1 + cu126 | |

## Why bare-metal TRL?

- **Assistant-only loss out-of-the-box**: TRL's `SFTConfig(assistant_only_loss=True)` masks tokens outside `{% generation %}` blocks, so the loss only sees what the model is *actually generating*. This typically gives a ~10-15 % faster wall-clock per epoch and noticeably cleaner gradients.
- **Smaller surface area**: no YAML, no dataset_info, no template registry — just Python.
- **Same final adapter format** (`adapter_model.safetensors` + `adapter_config.json`) as LLaMA-Factory / ms-swift, so the resulting LoRA loads with `PeftModel.from_pretrained` anywhere.

## Recipe

The full self-contained recipe is below — copy it into a file (e.g. `finetune_lora.py`) and run it directly.

### Chat template patch (for assistant-only loss)

The base model's full `chat_template.jinja` supports tools / think-mode / tool-calls, which is great for inference. For training, we want only **assistant content** to contribute to loss. We patch the tokenizer with a training-only template:

```python
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
```

> 🔑 **Don't save this template to disk!** It's training-only. Re-load the original tokenizer at inference time so the full chat template (with think / tool-call support) is preserved. Adapters trained with this patched template stay *fully compatible* with the base model's chat template at inference.

### Full training script (skeleton)

```python
import json, os, torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

BASE = "openbmb/MiniCPM5-1B"
DATA = "/path/to/my_chat_data.jsonl"
OUT  = "./runs/minicpm5_trl"

set_seed(42)

# 1. tokenizer + training-only chat template
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=False, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.chat_template = TRAIN_CHAT_TEMPLATE   # the template above

# 2. data: load jsonl, keep only `messages` column
rows = [json.loads(l) for l in open(DATA, encoding="utf-8") if l.strip()]
ds = Dataset.from_list([{"messages": r["messages"]} for r in rows])

# 3. model + LoRA
model = AutoModelForCausalLM.from_pretrained(
    BASE, trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
)
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

# 4. trainer
sft = SFTConfig(
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
    assistant_only_loss=True,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    report_to=["tensorboard"],
    dataloader_num_workers=2,
    remove_unused_columns=False,
    seed=42,
)
trainer = SFTTrainer(model=model, args=sft, train_dataset=ds, processing_class=tok)
trainer.train()
trainer.model.save_pretrained(f"{OUT}/adapter_final")   # adapter_model.safetensors + adapter_config.json
```

### Verified output

```
trainable params: 11,206,656 || all params: 1,091,839,488 || trainable%: 1.0264
{'loss': 4.0696, 'mean_token_accuracy': 0.2944, 'epoch': 0.2}
{'loss': 3.7437, 'mean_token_accuracy': 0.3315, 'epoch': 0.4}
{'loss': 3.6741, 'mean_token_accuracy': 0.3392, 'epoch': 0.6}
{'loss': 3.5366, 'mean_token_accuracy': 0.3524, 'epoch': 0.8}
{'loss': 3.5181, 'mean_token_accuracy': 0.3616, 'epoch': 1.0}
{'train_runtime': 14.91, 'train_samples_per_second': 13.4, 'train_loss': 3.71}
```

200-sample × 1-epoch tiny-LoRA on a single H200 — converges cleanly.

## Inference with the LoRA adapter

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("openbmb/MiniCPM5-1B", torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, "./runs/minicpm5_trl/adapter_final").eval()
tok = AutoTokenizer.from_pretrained("openbmb/MiniCPM5-1B")  # 🔑 re-load original tokenizer (for full chat_template)

msgs = [{"role": "user", "content": "用一句话解释什么是 GQA。"}]
inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=True, return_tensors="pt").to(model.device)
out = model.generate(inputs, max_new_tokens=512, do_sample=True, temperature=0.9, top_p=0.95)
print(tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True))
```

To merge the adapter into the base for serving:

```python
merged = model.merge_and_unload()
merged.save_pretrained("./minicpm5-trl-merged")
tok.save_pretrained("./minicpm5-trl-merged")    # important: original tokenizer (full chat_template)
```

## When to pick TRL over LLaMA-Factory / ms-swift

| Scenario | Pick |
| --- | --- |
| Want minimal Python control + zero CLI tooling | **TRL (this recipe)** |
| Need DPO / KTO / ORPO / RLOO out-of-the-box | TRL (covers all of them natively) or ms-swift |
| Need a YAML-driven pipeline with WebUI | LLaMA-Factory |
| Need ChatML template and Chinese-community-tested defaults | ms-swift |

## Q&A

### Loss doesn't go down

Make sure `assistant_only_loss=True` and the patched chat template have a `{% generation %}` block. Without it, TRL falls back to loss over *all* tokens, which dilutes signal and may even cause loss to *rise* on small adapters.

### `SFTConfig has no attribute 'max_length'`

Your `trl` is too old. `max_length` was added around 0.16. Upgrade to `trl>=0.18` (we tested 0.20.0).

## See also

- [`llamafactory.md`](./llamafactory.md), [`ms_swift.md`](./ms_swift.md) — higher-level alternatives
