---
name: minicpm5-finetune-ms-swift
description: Fine-tune MiniCPM5-1B with ms-swift (ModelScope's SFT / DPO / KTO / ORPO toolkit). Use when the user mentions "ms-swift", "swift sft", "swift rlhf", or wants ModelScope-native training. The two mandatory flags `--model_type llama --template chatml` are baked in.
---

# Fine-tune MiniCPM5-1B with ms-swift

ModelScope-native SFT / DPO / KTO / ORPO. ChatML template + standard `llama` model_type.

> ⚠️ **ms-swift 4.x renamed `--train_type` → `--tuner_type`**. Older tutorials still use `--train_type lora`, which on 4.x produces `ValueError: remaining_argv: ['--train_type', 'lora']`. Use `--tuner_type lora` (or just omit it — `lora` is the default in 4.x).

## Required input

| Var | Example | Default |
| --- | --- | --- |
| `BASE_MODEL` | `openbmb/MiniCPM5-1B` | required |
| `DATA` | path to messages-format jsonl | required |
| `OUTPUT_DIR` | `./runs/minicpm5_swift` | required |
| `GPU_ID` | `0` | `0` |

Each line of `DATA`: `{"messages": [{"role":"...","content":"..."}, ...]}`.

## Steps

### 1. Install (once)

```bash
pip install "ms-swift>=3.0"
# or for the dev branch:
pip install git+https://github.com/modelscope/ms-swift.git
```

### 2. Train (LoRA SFT)

```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} swift sft \
    --model "${BASE_MODEL}" \
    --model_type llama \
    --template chatml \
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

> 🔑 **`--model_type llama` AND `--template chatml` are MANDATORY.**
> Without them ms-swift errors with `Multiple possible types found: ['codefuse_codellama', 'llama', ...]` because MiniCPM5's disk-level structure is shared with several Llama-family models and ms-swift refuses to guess.

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

## Other trainers (DPO / KTO / ORPO / SimPO)

Same flag surface, just swap the trainer:

```bash
swift rlhf --rlhf_type dpo \
    --model "${BASE_MODEL}" --model_type llama --template chatml \
    --dataset preference.jsonl \
    --output_dir ${OUTPUT_DIR} ...
```

`dpo / kto / orpo / simpo` all work with the `chatml` template.

## Multi-GPU

```bash
NPROC_PER_NODE=8 swift sft \
    --model "${BASE_MODEL}" --model_type llama --template chatml \
    --tuner_type lora --deepspeed default-zero2 \
    ...
```

## Common pitfalls

- **`Failed to automatically match model_type`**: add `--model_type llama`.
- **`Failed to automatically match template_type`**: add `--template chatml`.
- **Conflict with LLaMA-Factory in same env**: LLaMA-Factory pins `transformers==4.52`, ms-swift wants the latest (currently transformers ≥5.6). Use separate venvs, or set `PYTHONNOUSERSITE=1` to ignore user-site `transformers`.

## Reference

[`docs/finetune/ms_swift.md`](../../docs/finetune/ms_swift.md)
