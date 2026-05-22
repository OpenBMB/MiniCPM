# Fine-tune MiniCPM5-1B with unsloth

[unsloth](https://github.com/unslothai/unsloth) is the "single-GPU LoRA on a budget" framework — heavy custom kernels for memory savings, and a wrapper around TRL `SFTTrainer` for the actual training loop. MiniCPM5-1B is small enough that unsloth's kernel speedups are less critical, but the **2× memory reduction** is still useful for long-context fine-tuning on a 24 GB consumer GPU.

> 🔑 If you're using unsloth alongside vLLM in the same env, pin `transformers==4.57.3` — unsloth's vLLM coexistence patch only handles transformers ≤ 4.57.x. Recipe baked in below.

## Verified versions

| Component | Version | Result |
| --- | --- | --- |
| unsloth | **2026.5.2** | LoRA SFT ✅ loss 4.67 → 3.52 (200 samples / 1 epoch / H200) |
| `unsloth_zoo` | 2026.5.1 | |
| `transformers` | **4.57.3** (pinned by unsloth, see Q&A) | |
| `peft` | 0.19.1 | |
| `trl` | 0.24.0 | |
| `torch` | 2.7.1 + cu126 | |

## Install

```bash
pip install "unsloth==2026.5.2"
# or latest:
pip install unsloth
```

> ⚠️ unsloth pulls in `transformers>=5.0`, but their `import_fixes.py` then **explicitly refuses to load** if vLLM is in the same env and `transformers>=5.0` (because vLLM ≤ 0.10.x doesn't yet support transformers 5.x). The fix is to pin `transformers==4.57.3`:
> ```bash
> pip install --force-reinstall "transformers==4.57.3"
> ```
>
> If you don't have vLLM in the same env, you can keep `transformers>=5.0` and unsloth will skip the vLLM patch.

## Recipe

```python
import json
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

BASE = "openbmb/MiniCPM5-1B"
DATA = "/path/to/my_chat_data.jsonl"
OUT  = "./runs/minicpm5_unsloth"

# 1. Load with unsloth's fast wrapper (this auto-patches the model)
model, tok = FastLanguageModel.from_pretrained(
    model_name=BASE,
    max_seq_length=2048,
    dtype=torch.bfloat16,
    load_in_4bit=False,           # set True for QLoRA on consumer GPUs
    full_finetuning=False,
)

# 2. Add LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",   # 30 % less VRAM than vanilla GC
    random_state=42,
)

# 3. Data — unsloth's SFTTrainer expects either `text` column or formatting_func
rows = [json.loads(l) for l in open(DATA, encoding="utf-8") if l.strip()]
ds = Dataset.from_list([
    {"text": tok.apply_chat_template(r["messages"], tokenize=False)}
    for r in rows
])
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# 4. Train (TRL SFTTrainer under the hood, with unsloth patches)
trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir=OUT,
        dataset_text_field="text",
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

## Verified output

```
==((====))==  Unsloth 2026.5.2: Fast Llama patching. Transformers: 4.57.3.
   \\   /|    NVIDIA H200. Num GPUs = 1. Max memory: 139.7 GB.
O^O/ \_/ \    Torch: 2.7.1+cu126. CUDA: 9.0. CUDA Toolkit: 12.6.
\        /    Bfloat16 = TRUE. Trainable parameters = 11,206,656 (1.03 %)

🦥 Unsloth: Padding-free auto-enabled, enabling faster training.

{'loss': 4.6738, 'epoch': 0.2}
{'loss': 3.9919, 'epoch': 0.4}
{'loss': 3.5176, 'epoch': 0.6}
{'loss': 3.4641, 'epoch': 0.8}
{'loss': 3.5212, 'epoch': 1.0}
{'train_runtime': 26.6, 'train_loss': 3.83}
```

Loss 4.67 → 3.52 over 25 optimizer steps (1.03 % trainable). The first iteration is slow (~13 s) because unsloth JIT-compiles patched kernels; subsequent iters are < 600 ms each.

## QLoRA (Int4) for consumer GPUs

```python
model, tok = FastLanguageModel.from_pretrained(
    model_name=BASE,
    max_seq_length=4096,
    dtype=torch.bfloat16,
    load_in_4bit=True,            # ← ~3-4 GB VRAM for the base model
    full_finetuning=False,
)
```

With `load_in_4bit=True`, MiniCPM5-1B + LoRA fits in **< 6 GB VRAM** at 4K context — so you can do QLoRA on a single 8 GB consumer GPU (RTX 3060 / 4060 Ti).

## Inference with the LoRA adapter

```python
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained(
    model_name="./runs/minicpm5_unsloth",   # path to adapter
    max_seq_length=4096,
    dtype=torch.bfloat16,
    load_in_4bit=False,
)
FastLanguageModel.for_inference(model)        # 2× faster generation

inputs = tok.apply_chat_template(
    [{"role": "user", "content": "用一句话解释 GQA。"}],
    add_generation_prompt=True, enable_thinking=True, return_tensors="pt",
).to("cuda")
out = model.generate(inputs, max_new_tokens=512, temperature=0.9, top_p=0.95)
print(tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True))
```

To merge for general serving:

```python
model.save_pretrained_merged("./minicpm5-unsloth-merged", tok)
```

## Q&A

### `RuntimeError: vLLM with version 0.10.x does not yet support transformers>=5.0.0`

unsloth pulls in `transformers>=5.0` but checks for vLLM compatibility on import. Pin `transformers==4.57.3`:

```bash
pip install --force-reinstall "transformers==4.57.3"
```

### `Unsloth: You must specify a formatting_func`

unsloth's TRL wrapper expects a flat `text` column or an explicit `formatting_func`. The recipe above uses `tok.apply_chat_template(...)` to convert messages → text up-front and sets `dataset_text_field="text"` in `SFTConfig`.

### Compared to LLaMA-Factory / ms-swift / TRL-direct

| Scenario | Pick |
| --- | --- |
| Single 8-24 GB consumer GPU + QLoRA | **unsloth** (2× memory savings, JIT kernels) |
| Multi-GPU + DeepSpeed | LLaMA-Factory or ms-swift |
| WebUI / YAML pipeline | LLaMA-Factory |
| Bare-metal Python + assistant-only loss | TRL direct (see [`trl.md`](./trl.md)) |

For a 1B model, unsloth's headline 2× speedup is usually 1.2-1.5× in practice (smaller models give the kernels less room to amortize their fixed overhead). The memory savings are still real and matter for long-context (16K-128K).

## See also

- [`llamafactory.md`](./llamafactory.md), [`ms_swift.md`](./ms_swift.md), [`trl.md`](./trl.md), [`xtuner.md`](./xtuner.md) — alternative frameworks
