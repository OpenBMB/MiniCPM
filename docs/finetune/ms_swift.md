# Fine-tune MiniCPM5-1B and MiniCPM5-2B with ms-swift

[ms-swift](https://github.com/modelscope/ms-swift) is the ModelScope team's official fine-tuning and serving toolkit. MiniCPM5-1B and MiniCPM5-2B use different registered templates:

| Model | ms-swift version | Arguments |
| --- | --- | --- |
| MiniCPM5-1B | PyPI `4.5.3` | `--model_type llama --template minicpm5` |
| MiniCPM5-2B | `>=4.6.0.dev0` | `--model_type llama --template minicpm5_2b` |

Both models require `transformers>=5.6`.

## Install

For MiniCPM5-1B, install the PyPI release directly:

```bash
pip install "ms-swift==4.5.3" "transformers>=5.6"
```

For MiniCPM5-2B, the `minicpm5_2b` template requires `ms-swift>=4.6.0.dev0`.
Install the pinned source revision:

```bash
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
git checkout 654e24f17b5d9f40ed4b9ee4c56a723320244db5
pip install -e .
pip install "transformers>=5.6,<5.17"
```

For Megatron-SWIFT, also install MCore-Bridge:

```bash
pip install mcore-bridge==1.6.4
```

## 1. Dataset format

ms-swift directly consumes the same **messages-style JSONL** that vLLM / SGLang / OpenAI use:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

No `dataset_info.json`. Just point `--dataset` at the file.

## 2. LoRA SFT command

The following example uses MiniCPM5-2B. For MiniCPM5-1B, change the model to `openbmb/MiniCPM5-1B` and the template to `minicpm5`.

```bash
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model openbmb/MiniCPM5-2B \
    --model_type llama \
    --template minicpm5_2b \
    --tuner_type lora \
    --dataset /path/to/my_chat_data.jsonl \
    --output_dir ./runs/minicpm5_swift \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --max_length 4096 \
    --warmup_ratio 0.03 \
    --bf16 true \
    --logging_steps 10 \
    --save_steps 200
```

> **Select the template that matches the model**:
> - `--model_type llama` — without it, ms-swift errors with `Multiple possible types found: ['codefuse_codellama', 'llama', 'openbuddy_llama', 'yi']`.
> - MiniCPM5-1B: `--template minicpm5`, available in the current PyPI release.
> - MiniCPM5-2B: `--template minicpm5_2b`, available in `ms-swift>=4.6.0.dev0`.
>
> Both errors are because MiniCPM5 shares its disk-level architecture and tokenizer with several llama-family models, and ms-swift refuses to guess.

## 3. Sample output

```
{'loss': 4.5170, 'token_acc': 0.2587, 'epoch': 0.04, 'memory(GiB)': 4.62}
{'loss': 4.2137, 'token_acc': 0.2879, 'epoch': 0.20}
{'loss': 3.8537, 'token_acc': 0.3155, 'epoch': 0.40}
{'loss': 3.6304, 'token_acc': 0.3500, 'epoch': 0.60}
{'loss': 3.6500, 'token_acc': 0.3429, 'epoch': 0.80}
{'loss': 3.5670, 'token_acc': 0.3536, 'epoch': 1.00}
{'train_runtime': 12.46, 'train_samples_per_second': 16.05, 'train_loss': 3.795}
```

Loss 4.52 → 3.57, token accuracy 0.26 → 0.35 — clean convergence.

## 4. Merge LoRA & inference

```bash
swift export \
    --model openbmb/MiniCPM5-2B \
    --adapters ./runs/minicpm5_swift/checkpoint-XXXX \
    --merge_lora true \
    --output_dir ./minicpm5-swift-merged
```

The merged model is a regular `LlamaForCausalLM`; serve it with any deployment backend.

## 5. Full SFT / DPO / RLHF

ms-swift exposes the same flag surface for full SFT, DPO, RLHF, ORPO, KTO. Switch the trainer:

```bash
# Full SFT
swift sft --tuner_type full ...

# DPO
swift rlhf --rlhf_type dpo --model ... --model_type llama --template minicpm5_2b  --dataset preference.jsonl ...

# Megatron-SWIFT SFT
megatron sft --model ... --model_type llama --template minicpm5_2b --dataset train.jsonl --finetune true --output_dir output ...
```

Use the matching template for each trainer: `minicpm5` for MiniCPM5-1B or `minicpm5_2b` for MiniCPM5-2B.

For complete training options and configuration details, see the
[ms-swift command-line parameters](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html)
and the
[Megatron-SWIFT quick start](https://swift.readthedocs.io/en/latest/Megatron-SWIFT/Quick-start.html).

## 6. Multi-GPU

```bash
NPROC_PER_NODE=8 swift sft \
    --model openbmb/MiniCPM5-2B \
    --model_type llama \
    --template minicpm5_2b \
    --tuner_type lora \
    --deepspeed default-zero2 \
    ...
```

ms-swift auto-launches `torchrun` when `NPROC_PER_NODE` is set, so you don't write your own `torchrun ...` invocation.

## Q&A

### `Failed to automatically match model_type`

Add `--model_type llama` (see "Two flags you MUST pass" above).

### `Failed to automatically match template_type`

Use `--template minicpm5` for MiniCPM5-1B. Use `--template minicpm5_2b` for MiniCPM5-2B and make sure the active environment uses `ms-swift>=4.6.0.dev0`.

### `minicpm5_2b is not registered`

Check the installed `ms-swift` version:

```bash
python -c "from importlib.metadata import version; print(version('ms-swift'))"
```

If the environment contains an older release such as `4.4.1`, install `ms-swift>=4.6.0.dev0`. The source installation above pins a known revision for reproducibility.

### Conflict with LLaMA-Factory in the same env

LLaMA-Factory 0.9.3 pulls in `transformers==4.52`, ms-swift's `swift sft` works with `transformers>=4.45` but is happiest on the latest. Use `PYTHONNOUSERSITE=1` if both are installed (LLaMA-Factory in `~/.local`, ms-swift in a conda env), or use separate conda envs.

## See also

- [`llamafactory.md`](./llamafactory.md) — community standard, similar capabilities
- [`trl.md`](./trl.md) — bare-metal TRL + PEFT recipe with assistant-only loss
