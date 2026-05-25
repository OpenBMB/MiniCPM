# Fine-tune MiniCPM5-1B with LLaMA-Factory

[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) is the most-used community fine-tuning framework. MiniCPM5-1B is a vanilla `LlamaForCausalLM`, so it works out-of-the-box with LLaMA-Factory's standard recipe.

## Install

```bash
pip install "llamafactory==0.9.3"        # or just `pip install llamafactory` for latest
```

## 1. Dataset prep

LLaMA-Factory consumes a `dataset_info.json` registry. For the MiniCPM5 chat template, use `formatting: sharegpt` and the standard role tags:

```bash
mkdir -p ~/finetune_data
cd ~/finetune_data
```

`~/finetune_data/dataset_info.json`:

```json
{
  "my_chat_data": {
    "file_name": "my_chat_data.jsonl",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "system_tag": "system"
    }
  }
}
```

Each line of `my_chat_data.jsonl` is a chat:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

## 2. LoRA SFT config

Save as `lora_sft.yaml`:

```yaml
### model
model_name_or_path: openbmb/MiniCPM5-1B
trust_remote_code: false

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_target: all          # all linear layers; or "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

### dataset
dataset: my_chat_data
dataset_dir: /path/to/finetune_data
template: empty           # MiniCPM5 chat template auto-loads from the model's tokenizer_config.json
cutoff_len: 4096
max_samples: 100000
overwrite_cache: true
preprocessing_num_workers: 8

### output
output_dir: ./runs/minicpm5_lora
logging_steps: 10
save_steps: 200
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2.0e-4
num_train_epochs: 2.0
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true
ddp_timeout: 180000000
```

> 💡 `template: empty` tells LLaMA-Factory to **delegate to the tokenizer's built-in `chat_template.jinja`**, which is the MiniCPM5 ChatML-style template (with think / no-think / tools support). Do **not** set `template: llama3` or other built-ins — they will produce a broken token layout.

## 3. Train

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train lora_sft.yaml
```

Sample run (200 samples, 1 epoch, single GPU, bs=4, grad_acc=2, lr=2e-4):

```
{'loss': 4.1912, 'learning_rate': 0.000192, 'epoch': 0.2}
{'loss': 3.9248, 'learning_rate': 0.000150, 'epoch': 0.4}
{'loss': 3.8348, 'learning_rate': 0.000087, 'epoch': 0.6}
{'loss': 3.6936, 'learning_rate': 0.000029, 'epoch': 0.8}
{'loss': 3.6183, 'learning_rate': 0.000001, 'epoch': 1.0}
{'train_runtime': 11.45, 'train_samples_per_second': 17.5, 'train_loss': 3.85}
```

Loss decreases monotonically from 4.19 → 3.62 over the 25 optimizer steps — the framework + chat template + tokenizer combo all work end-to-end.

## 4. Inference with the LoRA adapter

The adapter is saved to `output_dir/checkpoint-XXXX/` and `output_dir/`. To merge for deployment:

```bash
llamafactory-cli export merge.yaml
```

`merge.yaml`:

```yaml
model_name_or_path: openbmb/MiniCPM5-1B
adapter_name_or_path: ./runs/minicpm5_lora
template: empty
finetuning_type: lora
export_dir: ./minicpm5-merged
export_size: 4
export_legacy_format: false
```

The merged model is a regular `LlamaForCausalLM` and can be served with **any** of the deployment paths (`vllm`, `sglang`, `transformers`, `llama.cpp`-after-GGUF) without changes.

## 5. Full SFT (no LoRA)

If you have enough GPU memory (~12 GB for bf16 + AdamW on a single GPU), drop `finetuning_type: lora` and the LoRA fields:

```yaml
finetuning_type: full
deepspeed: examples/deepspeed/ds_z2_config.json   # optional, for multi-GPU
```

## Q&A

### `template not found` / wrong tokens

You probably set `template: llama3` or similar. Use `template: empty` to delegate to the model's own `chat_template.jinja`.

### `transformers >= 4.55 required` (when also using vLLM in the same env)

LLaMA-Factory 0.9.3 wants `transformers==4.52`, vLLM 0.21 wants `>=5.6`. Use two virtual environments — one for fine-tuning, one for serving. The merged model is portable across them.

### Multi-GPU

Add `deepspeed: examples/deepspeed/ds_z2_config.json` to the YAML, and launch with:

```bash
FORCE_TORCHRUN=1 llamafactory-cli train lora_sft.yaml
```

LLaMA-Factory's `examples/deepspeed/` folder ships ZeRO-2 / ZeRO-3 / ZeRO-3-offload templates that all work directly with MiniCPM5.

## See also

- [`ms_swift.md`](./ms_swift.md) — alternative ModelScope-style framework (similar capabilities)
- [`trl.md`](./trl.md) — minimal TRL + PEFT recipe (closer to the metal, fewer abstractions)
