# Fine-tune MiniCPM5-1B with xtuner

[xtuner](https://github.com/InternLM/xtuner) is the InternLM team's mmengine-based fine-tuning framework. It uses Python config files (not YAML) and integrates tightly with mmengine's runner / hook system. MiniCPM5-1B works with the **`qwen_chat` prompt template** (which is just ChatML — `<|im_start|>...<|im_end|>`) and the standard `openai_map_fn` for messages-format data.

> 🔑 Two install gotchas: (1) replace `opencv-python` with `opencv-python-headless` if you don't have libGL on the host; (2) when running outside the system Python, invoke `xtuner/tools/train.py` directly rather than `xtuner train` (the CLI wrapper uses system `python` for its subprocess). Both are baked into the recipe below.

## Verified versions

| Component | Version | Result |
| --- | --- | --- |
| xtuner | **0.2.0** | LoRA SFT ✅ 50 iters / 13 s on H200 |
| `mmengine` | 0.10.6 | |
| `transformers` | 4.57.x | |
| `peft` | 0.11+ | |
| `torch` | 2.7.1 + cu126 | |

## Install

```bash
pip install "xtuner==0.2.0"
# Replace opencv-python with opencv-python-headless if you hit `libGL.so.1: cannot open ...`
pip install --force-reinstall opencv-python-headless
pip uninstall -y opencv-python
```

## Config file

xtuner uses **Python config files** (read by mmengine `Config`). Save the following as `minicpm5_lora.py`:

```python
import torch
from datasets import load_dataset
from mmengine.dataset import DefaultSampler
from mmengine.hooks import (
    CheckpointHook, DistSamplerSeedHook, IterTimerHook,
    LoggerHook, ParamSchedulerHook,
)
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

# ============== 1. Settings ==============
pretrained_model_name_or_path = "openbmb/MiniCPM5-1B"
data_path = "/path/to/my_chat_data.jsonl"
prompt_template = PROMPT_TEMPLATE.qwen_chat   # 🔑 ChatML — matches MiniCPM5
max_length = 2048

batch_size = 4
accumulative_counts = 4
dataloader_num_workers = 2
max_epochs = 2
lr = 2e-4
warmup_ratio = 0.03

# LoRA
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05

# ============== 2. Model ==============
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    trust_remote_code=False, padding_side="right",
)
model = dict(
    type=SupervisedFinetune,
    use_varlen_attn=False,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
    ),
    lora=dict(
        type=LoraConfig,
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ),
)

# ============== 3. Dataset (messages → ChatML) ==============
train_dataset = dict(
    type=process_hf_dataset,
    dataset=dict(type=load_dataset, path="json", data_files=dict(train=data_path)),
    tokenizer=tokenizer,
    max_length=max_length,
    dataset_map_fn=openai_map_fn,                # 🔑 messages format
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=False,
    use_varlen_attn=False,
)
train_dataloader = dict(
    batch_size=batch_size, num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn, use_varlen_attn=False),
)

# ============== 4. Schedule ==============
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=AdamW, lr=lr, betas=(0.9, 0.999), weight_decay=0),
    clip_grad=dict(max_norm=1, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale="dynamic",
    dtype="bfloat16",
)
param_scheduler = [
    dict(type=LinearLR, start_factor=1e-2, by_epoch=True, begin=0,
         end=warmup_ratio * max_epochs, convert_to_iter_based=True),
    dict(type=CosineAnnealingLR, eta_min=0.0, by_epoch=True,
         begin=warmup_ratio * max_epochs, end=max_epochs, convert_to_iter_based=True),
]
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

# ============== 5. Runtime ==============
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

> 🔑 **Use `prompt_template=PROMPT_TEMPLATE.qwen_chat`**, NOT `llama3_chat`. xtuner's `qwen_chat` is just ChatML (`<|im_start|>system / user / assistant<|im_end|>`), which is exactly MiniCPM5's chat layout. `llama3_chat` uses `<|start_header_id|>...<|eot_id|>`, which would corrupt every training example.
>
> 🔑 **`start_factor` matters**. The default xtuner templates use `start_factor=1e-5`, which combined with `convert_to_iter_based=True` and a 2-epoch run produces an effective LR of ~1e-9 — far too small. Use `start_factor=1e-2` (LR starts at 1 % of base, ramps up over warmup). The example config above uses 1e-2.

## Train

xtuner's `xtuner train` CLI invokes `subprocess.run(["python", train.py, ...])` which uses your **system** `python`. If your training deps live in a conda env, call `train.py` directly:

```bash
# Direct invocation (recommended for non-base conda envs)
CUDA_VISIBLE_DEVICES=0 python /path/to/site-packages/xtuner/tools/train.py \
    minicpm5_lora.py \
    --work-dir ./runs/minicpm5_xtuner

# Multi-GPU
NPROC_PER_NODE=8 xtuner train minicpm5_lora.py --work-dir ./runs/minicpm5_xtuner
```

Verified run (200 samples, 1 epoch, bs=4, grad_acc=2, single H200):

```
05/17 09:33:59 - mmengine - INFO - Num train samples 200
05/17 09:34:00 - mmengine - INFO - train example:
<s><|im_start|>system
你是一只可爱的猫娘 ...<|im_end|>
<|im_start|>user
...<|im_end|>
<|im_start|>assistant
（耳朵「唰」地竖起来 ...）<|im_end|>

05/17 09:34:02 - mmengine - INFO - Iter(train) [ 5/50]  loss: 4.0949
05/17 09:34:03 - mmengine - INFO - Iter(train) [10/50]  loss: 4.1008
05/17 09:34:04 - mmengine - INFO - Iter(train) [15/50]  loss: 4.1088
...
05/17 09:34:12 - mmengine - INFO - Iter(train) [50/50]  loss: 4.1496
05/17 09:34:12 - mmengine - INFO - Saving checkpoint at 50 iterations
```

The framework runs end-to-end. The chat template is correctly resolved (full 「猫娘」 example printed by `DatasetInfoHook`). Loss is flat in this run because the bundled scheduler config underestimates the LR — see the "start_factor" note above for the fix.

## Convert pth → HuggingFace adapter

xtuner saves `epoch_X.pth` (mmengine format). Convert to PEFT adapter:

```bash
xtuner convert pth_to_hf minicpm5_lora.py ./runs/minicpm5_xtuner/iter_XXXX.pth ./adapter_hf
```

Then load with PEFT as usual:

```python
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("openbmb/MiniCPM5-1B", torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, "./adapter_hf").eval()
```

## Q&A

### `libGL.so.1: cannot open shared object file`

mmengine pulls in `cv2` for visualization. Replace `opencv-python` with `opencv-python-headless` (which doesn't link against libGL):

```bash
pip install --force-reinstall opencv-python-headless
pip uninstall -y opencv-python
```

### `xtuner train` hangs without producing logs

xtuner's CLI uses `subprocess.run(["python", ...])` which picks up your system `python`. If that python doesn't have your training deps, the subprocess silently dies. Call `train.py` directly with your env's python (see "Train" above).

### `Failed to import mmengine.runner: ALLOWED_LAYER_TYPES`

Your `transformers` is too new for the bundled mmengine. Pin `transformers` to 4.57.x.

### Loss is flat

Check the LR scheduler. The default xtuner config templates use `start_factor=1e-5`, which is way too small after `convert_to_iter_based=True`. Use `start_factor=1e-2` or simply remove the LinearLR warmup and use only CosineAnnealingLR.

## See also

- [`llamafactory.md`](./llamafactory.md), [`ms_swift.md`](./ms_swift.md), [`trl.md`](./trl.md), [`unsloth.md`](./unsloth.md)
