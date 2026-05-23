---
name: minicpm5-finetune-xtuner
description: Fine-tune MiniCPM5-1B with xtuner (mmengine config-driven SFT). Use when the user mentions "xtuner", "mmengine", InternLM's training framework, or wants config-file-driven training.
---

# Fine-tune MiniCPM5-1B with xtuner

mmengine config-driven SFT. Uses Python config files (not YAML) and integrates with mmengine's runner / hook system.

## Required input

| Var | Example | Default |
| --- | --- | --- |
| `BASE_MODEL` | `openbmb/MiniCPM5-1B` | required |
| `DATA` | path to messages-format jsonl | required |
| `WORK_DIR` | `./runs/minicpm5_xtuner` | required |

## Steps

### 1. Install (once)

```bash
pip install "xtuner==0.2.0"
# Replace opencv-python with the headless variant (avoids libGL linkage)
pip install --force-reinstall opencv-python-headless
pip uninstall -y opencv-python
```

### 2. Save the config — `${WORK_DIR}/minicpm5_lora.py`

```python
import torch
from datasets import load_dataset
from mmengine.dataset import DefaultSampler
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from peft import LoraConfig
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from xtuner.dataset import process_hf_dataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import openai_map_fn, template_map_fn_factory
from xtuner.engine.hooks import DatasetInfoHook
from xtuner.engine.runner import TrainLoop
from xtuner.model import SupervisedFinetune
from xtuner.utils import PROMPT_TEMPLATE

pretrained_model_name_or_path = "${BASE_MODEL}"      # ← replace
data_path = "${DATA}"                                # ← replace
prompt_template = PROMPT_TEMPLATE.qwen_chat          # 🔑 ChatML — DO NOT use llama3_chat
max_length = 2048

batch_size = 4
accumulative_counts = 4
max_epochs = 2
lr = 2e-4
warmup_ratio = 0.03

tokenizer = dict(type=AutoTokenizer.from_pretrained,
                 pretrained_model_name_or_path=pretrained_model_name_or_path,
                 trust_remote_code=False, padding_side="right")

model = dict(
    type=SupervisedFinetune, use_varlen_attn=False,
    llm=dict(type=AutoModelForCausalLM.from_pretrained,
             pretrained_model_name_or_path=pretrained_model_name_or_path,
             trust_remote_code=False, torch_dtype=torch.bfloat16),
    lora=dict(type=LoraConfig, r=16, lora_alpha=32, lora_dropout=0.05,
              bias="none", task_type="CAUSAL_LM",
              target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]),
)

train_dataset = dict(
    type=process_hf_dataset,
    dataset=dict(type=load_dataset, path="json", data_files=dict(train=data_path)),
    tokenizer=tokenizer, max_length=max_length,
    dataset_map_fn=openai_map_fn,                                # 🔑 messages format
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True, shuffle_before_pack=True,
    pack_to_max_length=False, use_varlen_attn=False,
)
train_dataloader = dict(
    batch_size=batch_size, num_workers=2,
    dataset=train_dataset,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn, use_varlen_attn=False),
)

optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=AdamW, lr=lr, betas=(0.9, 0.999), weight_decay=0),
    clip_grad=dict(max_norm=1, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts, loss_scale="dynamic", dtype="bfloat16",
)
param_scheduler = [
    dict(type=LinearLR, start_factor=1e-2,                       # 🔑 use 1e-2 not 1e-5 (default is too small)
         by_epoch=True, begin=0, end=warmup_ratio * max_epochs, convert_to_iter_based=True),
    dict(type=CosineAnnealingLR, eta_min=0.0, by_epoch=True,
         begin=warmup_ratio * max_epochs, end=max_epochs, convert_to_iter_based=True),
]
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=CheckpointHook, by_epoch=False, interval=200, max_keep_ckpts=2),
    sampler_seed=dict(type=DistSamplerSeedHook),
)
custom_hooks = [dict(type=DatasetInfoHook, tokenizer=tokenizer)]
env_cfg = dict(cudnn_benchmark=False, mp_cfg=dict(mp_start_method="fork"), dist_cfg=dict(backend="nccl"))
log_level = "INFO"
load_from = None
resume = False
randomness = dict(seed=42, deterministic=False)
log_processor = dict(by_epoch=False)
```

> 🔑 **`PROMPT_TEMPLATE.qwen_chat` is correct** — that's the ChatML template. Do NOT use `llama3_chat` (which is `<|start_header_id|>...<|eot_id|>` and would corrupt every example).
> 🔑 **`start_factor=1e-2`** for LinearLR — xtuner's default `start_factor=1e-5` combined with `convert_to_iter_based=True` produces an effective LR of ~1e-9 (way too small).

### 3. Train — invoke `train.py` directly (not `xtuner train`)

```bash
TRAIN_PY=$(python -c "import xtuner.tools.train as t; print(t.__file__)")
CUDA_VISIBLE_DEVICES=0 python $TRAIN_PY ${WORK_DIR}/minicpm5_lora.py --work-dir ${WORK_DIR}
```

> 🔑 **Use `python TRAIN_PY` directly** — `xtuner train` invokes `subprocess.run(["python", train.py, ...])` which uses the *system* `python`, not your conda env. If your training deps live in a non-default env, the subprocess silently dies with no logs.

For multi-GPU, use the standard wrapper: `NPROC_PER_NODE=8 xtuner train CONFIG --work-dir ...`.

### 4. Validate

```
05/17 09:33:59 - mmengine - INFO - Num train samples 200
05/17 09:34:00 - mmengine - INFO - train example:
<s><|im_start|>system
你是 ...<|im_end|>
<|im_start|>user
...<|im_end|>
<|im_start|>assistant
...<|im_end|>

Iter(train) [10/100]  loss: 4.10
Iter(train) [50/100]  loss: 3.50
Saving checkpoint at 200 iterations
```

The chat template should resolve into proper `<|im_start|>...<|im_end|>` markers (visible from `DatasetInfoHook` output). Loss should decrease.

## Convert pth → HF adapter

xtuner saves `epoch_X.pth` (mmengine format). Convert to PEFT adapter:

```bash
xtuner convert pth_to_hf ${WORK_DIR}/minicpm5_lora.py ${WORK_DIR}/iter_XXXX.pth ./adapter_hf
```

Then load with PEFT:

```python
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("openbmb/MiniCPM5-1B", torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, "./adapter_hf").eval()
```

## Common pitfalls

- **`libGL.so.1: cannot open shared object file`**: replace `opencv-python` → `opencv-python-headless` (see install step).
- **`xtuner train` hangs without logs**: invoke `train.py` directly (see step 3).
- **`Failed to import mmengine.runner: ALLOWED_LAYER_TYPES`**: `transformers` too new. Pin `transformers==4.57.x`.
- **Loss flat**: scheduler LR underestimated. Use `start_factor=1e-2` (see config above).

## Reference

[`docs/finetune/xtuner.md`](../../docs/finetune/xtuner.md)
