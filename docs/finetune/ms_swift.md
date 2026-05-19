# Fine-tune MiniCPM5-1B with ms-swift

[ms-swift](https://github.com/modelscope/ms-swift) is the ModelScope team's official fine-tuning + serving toolkit. MiniCPM5-1B works with the standard `llama` model_type and ChatML template — no model-code patch.

> 🔑 **Two flags are mandatory** for MiniCPM5: `--model_type llama --template chatml`. Without them ms-swift refuses to auto-detect the architecture / template (because the disk-level structure is shared with several other Llama-family models).

## Verified versions

| Component | Version | Result |
| --- | --- | --- |
| ms-swift | **4.1.0.dev0** | LoRA SFT ✅ loss 4.52 → 3.57 (200 samples / 1 epoch / H200) |
| `transformers` | 4.57.1 | |
| `peft` | 0.11.1 | |
| `trl` | 0.20.0 | |
| `torch` | 2.7.1 + cu126 | |

## Install

```bash
pip install "ms-swift>=3.0"
# or, for the dev branch we tested:
pip install git+https://github.com/modelscope/ms-swift.git
```

## 1. Dataset format

ms-swift directly consumes the same **messages-style JSONL** that vLLM / SGLang / OpenAI use:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

No `dataset_info.json`. Just point `--dataset` at the file.

## 2. LoRA SFT command

```bash
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model openbmb/MiniCPM5-1B \
    --model_type llama \
    --template chatml \
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

> 🔑 **Two flags you MUST pass for MiniCPM5**:
> - `--model_type llama` — without it, ms-swift errors with `Multiple possible types found: ['codefuse_codellama', 'llama', 'openbuddy_llama', 'yi']`.
> - `--template chatml` — without it, ms-swift errors with `Multiple possible types found: ['llama', 'atom', 'megrez', ...]`.
>
> Both errors are because MiniCPM5 shares its disk-level architecture and tokenizer with several llama-family models, and ms-swift refuses to guess.

## 3. Verified output

```
{'loss': 4.5170, 'token_acc': 0.2587, 'epoch': 0.04, 'memory(GiB)': 4.62}
{'loss': 4.2137, 'token_acc': 0.2879, 'epoch': 0.20}
{'loss': 3.8537, 'token_acc': 0.3155, 'epoch': 0.40}
{'loss': 3.6304, 'token_acc': 0.3500, 'epoch': 0.60}
{'loss': 3.6500, 'token_acc': 0.3429, 'epoch': 0.80}
{'loss': 3.5670, 'token_acc': 0.3536, 'epoch': 1.00}
{'train_runtime': 12.46, 'train_samples_per_second': 16.05, 'train_loss': 3.795}
```

Loss 4.52 → 3.57, token accuracy 0.26 → 0.35, peak GPU memory 9.8 GB on a single H200 — clean convergence.

## 4. Merge LoRA & inference

```bash
swift export \
    --model openbmb/MiniCPM5-1B \
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
swift rlhf --rlhf_type dpo --model ... --template chatml --dataset preference.jsonl ...
```

For the `chatml` template combined with MiniCPM5, all of `sft / dpo / kto / orpo / simpo` are tested and work.

## 6. Multi-GPU

```bash
NPROC_PER_NODE=8 swift sft \
    --model openbmb/MiniCPM5-1B \
    --model_type llama \
    --template chatml \
    --tuner_type lora \
    --deepspeed default-zero2 \
    ...
```

ms-swift auto-launches `torchrun` when `NPROC_PER_NODE` is set, so you don't write your own `torchrun ...` invocation.

## Q&A

### `Failed to automatically match model_type`

Add `--model_type llama` (see "Two flags you MUST pass" above).

### `Failed to automatically match template_type`

Add `--template chatml`.

### Output looks slightly off after training

The `chatml` template works for MiniCPM5's `<|im_start|>user / <|im_start|>assistant` layout, but it does **not** include MiniCPM5's `<think>` block tokens by default. If you want think-mode preservation, fine-tune through the model's native `tokenizer.chat_template` instead — see [`trl.md`](./trl.md), which patches the tokenizer's chat_template directly to keep think-mode behaviour.

### Conflict with LLaMA-Factory in the same env

LLaMA-Factory 0.9.3 pulls in `transformers==4.52`, ms-swift's `swift sft` works with `transformers>=4.45` but is happiest on the latest. Use `PYTHONNOUSERSITE=1` if both are installed (LLaMA-Factory in `~/.local`, ms-swift in a conda env), or use separate conda envs.

## See also

- [`llamafactory.md`](./llamafactory.md) — community standard, similar capabilities
- [`trl.md`](./trl.md) — bare-metal TRL + PEFT recipe with assistant-only loss
