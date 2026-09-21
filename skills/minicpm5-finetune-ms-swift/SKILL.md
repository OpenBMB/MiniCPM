---
name: minicpm5-finetune-ms-swift
description: Fine-tune MiniCPM5-1B or MiniCPM5-2B with ms-swift or Megatron-SWIFT. Use when the user mentions "ms-swift", "swift sft", "swift rlhf", or "megatron sft". MiniCPM5-1B uses the PyPI release with `--template minicpm5`; MiniCPM5-2B requires `ms-swift==4.6.0.dev0` with `--template minicpm5_2b`. Both use `--model_type llama`.
---

# Fine-tune MiniCPM5-1B and MiniCPM5-2B with ms-swift

Select the version and template that match the model:

| Model | ms-swift version | Arguments |
| --- | --- | --- |
| MiniCPM5-1B | PyPI `4.5.3` | `--model_type llama --template minicpm5` |
| MiniCPM5-2B | `4.6.0.dev0` | `--model_type llama --template minicpm5_2b` |

Both models require `transformers>=5.6`.

> **ms-swift 4.x renamed `--train_type` to `--tuner_type`.** Use `--tuner_type lora` for LoRA training.

## Required input

| Var | Example | Default |
| --- | --- | --- |
| `BASE_MODEL` | `openbmb/MiniCPM5-2B` | required; `openbmb/MiniCPM5-1B` is also supported |
| `TEMPLATE` | `minicpm5_2b` | use `minicpm5` for MiniCPM5-1B |
| `DATA` | path to messages-format jsonl | required |
| `OUTPUT_DIR` | `./runs/minicpm5_swift` | required |
| `GPU_ID` | `0` | `0` |

Each line of `DATA`: `{"messages": [{"role":"...","content":"..."}, ...]}`.

## Steps

### 1. Install (once)

For MiniCPM5-1B:

```bash
pip install "ms-swift==4.5.3" "transformers>=5.6"
```

For MiniCPM5-2B:

```bash
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
git checkout 654e24f17b5d9f40ed4b9ee4c56a723320244db5
pip install -e .
pip install "transformers>=5.6,<5.17"
```

For Megatron-SWIFT:

```bash
pip install mcore-bridge==1.6.4
```

### 2. Train (LoRA SFT)

```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} swift sft \
    --model "${BASE_MODEL}" \
    --model_type llama \
    --template "${TEMPLATE}" \
    --tuner_type lora \
    --dataset "${DATA}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 \
    --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --max_length 4096 \
    --warmup_ratio 0.03 \
    --bf16 true \
    --logging_steps 10 \
    --save_steps 200
```

Set `TEMPLATE=minicpm5` for MiniCPM5-1B or `TEMPLATE=minicpm5_2b` for MiniCPM5-2B.

### 3. Validate

Loss should decrease over the first few hundred steps:

```
{'loss': 4.52, 'token_acc': 0.26, 'epoch': 0.04}
{'loss': 3.57, 'token_acc': 0.35, 'epoch': 1.00}
```

Adapter is at `${OUTPUT_DIR}/v0-${TIMESTAMP}/checkpoint-${STEP}/`.

## Merge for serving

```bash
swift export \
    --model "${BASE_MODEL}" \
    --adapters "${OUTPUT_DIR}/v0-${TIMESTAMP}/checkpoint-${STEP}" \
    --merge_lora true \
    --output_dir ./minicpm5-swift-merged
```

The merged model is a regular `LlamaForCausalLM` and serves with any `minicpm5-deploy-*` skill.

## Full SFT / DPO / RLHF

Same flag surface, just swap the trainer:

```bash
# Full SFT
swift sft --tuner_type full ...

# DPO
swift rlhf --rlhf_type dpo \
    --model "${BASE_MODEL}" --model_type llama --template "${TEMPLATE}" \
    --dataset preference.jsonl \
    --output_dir ${OUTPUT_DIR} ...

# Megatron-SWIFT SFT
megatron sft \
    --model "${BASE_MODEL}" \
    --model_type llama \
    --template "${TEMPLATE}" \
    --dataset "${DATA}" \
    --finetune true \
    --output_dir "${OUTPUT_DIR}"
```

Set `TEMPLATE=minicpm5` for MiniCPM5-1B or `TEMPLATE=minicpm5_2b` for MiniCPM5-2B.

For complete training options and configuration details, see the
[ms-swift command-line parameters](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html)
and the
[Megatron-SWIFT quick start](https://swift.readthedocs.io/en/latest/Megatron-SWIFT/Quick-start.html).

## Multi-GPU

```bash
NPROC_PER_NODE=8 swift sft \
    --model "${BASE_MODEL}" --model_type llama --template "${TEMPLATE}" \
    --tuner_type lora --deepspeed default-zero2 \
    ...
```

## Common pitfalls

- **`Failed to automatically match model_type`**: add `--model_type llama`.
- **`Failed to automatically match template_type`**: use `--template minicpm5` for MiniCPM5-1B or `--template minicpm5_2b` for MiniCPM5-2B.
- **`minicpm5_2b is not registered`**: the active Python environment is using an older ms-swift release. Install the pinned `4.6.0.dev0` source revision.
- **Conflict with LLaMA-Factory in same env**: LLaMA-Factory pins `transformers==4.52`, ms-swift wants the latest (currently transformers ≥5.6). Use separate venvs, or set `PYTHONNOUSERSITE=1` to ignore user-site `transformers`.

## Reference

[`docs/finetune/ms_swift.md`](../../docs/finetune/ms_swift.md)
