---
name: minicpm5-finetune-llamafactory
description: Fine-tune MiniCPM5-1B with LLaMA-Factory (YAML-driven SFT / DPO / WebUI). Use when the user wants to fine-tune via LLaMA-Factory, llamafactory-cli, mentions YAML configs, WebUI, or asks for the most-documented community framework.
---

# Fine-tune MiniCPM5-1B with LLaMA-Factory

YAML-driven SFT / DPO with WebUI. Most-documented community framework.

## Required input

| Var | Example | Default |
| --- | --- | --- |
| `BASE_MODEL` | `openbmb/MiniCPM5-1B` | required |
| `DATA_DIR` | dir containing `dataset_info.json` + jsonl | required |
| `DATASET_NAME` | name registered in `dataset_info.json` | required |
| `OUTPUT_DIR` | `./runs/minicpm5_lf` | required |
| `GPU_ID` | `0` | `0` |

## Steps

### 1. Install (once, in its own venv to avoid breaking vLLM)

```bash
python -m venv .venv-lf && source .venv-lf/bin/activate
# `template: minicpm5` landed after v0.9.5 (the latest PyPI release) — install from source.
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
pip install -e LLaMA-Factory
```

> ⚠️ LLaMA-Factory requires `transformers>=4.55.0,<=5.8.0` (excluding 4.57.0 and 5.6.0), which can clash with a serving stack such as vLLM. Always install it into its own venv.

### 2. Register the dataset (sharegpt / messages format)

`${DATA_DIR}/dataset_info.json`:

```json
{
  "${DATASET_NAME}": {
    "file_name": "your_data.jsonl",
    "formatting": "sharegpt",
    "columns": {"messages": "messages"},
    "tags": {
      "role_tag": "role", "content_tag": "content",
      "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"
    }
  }
}
```

Each line of `your_data.jsonl`:

```json
{"messages": [{"role":"system","content":"..."}, {"role":"user","content":"..."}, {"role":"assistant","content":"..."}]}
```

### 3. Write the training YAML

Save as `${OUTPUT_DIR}/lora_sft.yaml`:

```yaml
### model
model_name_or_path: ${BASE_MODEL}
trust_remote_code: false

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_target: all                      # all linear layers

### dataset
dataset: ${DATASET_NAME}
dataset_dir: ${DATA_DIR}
template: minicpm5                    # 🔑 MANDATORY for MiniCPM5 — ChatML + XML tool calling
cutoff_len: 4096
max_samples: 100000
overwrite_cache: true
preprocessing_num_workers: 8

### output
output_dir: ${OUTPUT_DIR}
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

> 🔑 **`template: minicpm5` is MANDATORY.** It reproduces the model's own `chat_template.jinja` byte-for-byte — ChatML with think / nothink, plus XML tool calling.
>
> Do NOT use `template: empty`. LLaMA-Factory does not delegate to the tokenizer's jinja; `empty` is a genuinely empty template (bare `{{content}}` slots, no role markers, no EOS replacement, ReAct-style tools) that trains on a layout the model has never seen. `llama3` / `qwen` / etc. are equally wrong. Requires LLaMA-Factory from source — see step 1.

### 4. Train

```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} llamafactory-cli train ${OUTPUT_DIR}/lora_sft.yaml
```

For multi-GPU: prepend `FORCE_TORCHRUN=1` and set `NPROC_PER_NODE=8`.

### 5. Validate

Loss should decrease monotonically; you should see lines like:

```
{'loss': 4.19, 'learning_rate': 0.000192, 'epoch': 0.2}
{'loss': 3.62, 'learning_rate': 0.000001, 'epoch': 1.0}
```

After training, the LoRA adapter is at `${OUTPUT_DIR}/`. Sanity-check inference:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained("${BASE_MODEL}", torch_dtype=torch.bfloat16, device_map="auto").eval()
model = PeftModel.from_pretrained(base, "${OUTPUT_DIR}").eval()
tok = AutoTokenizer.from_pretrained("${BASE_MODEL}")
inputs = tok.apply_chat_template([{"role":"user","content":"1+1=?"}], add_generation_prompt=True, enable_thinking=False, return_tensors="pt").to(model.device)
print(tok.decode(model.generate(inputs, max_new_tokens=32, do_sample=False)[0][inputs.shape[-1]:], skip_special_tokens=True))
```

Coherent answer ⇒ ✅. Gibberish ⇒ check `template: minicpm5` in the YAML. (A clean loss curve does **not** confirm the template is right — a mismatched template trains just as smoothly.)

## Merge LoRA for serving

```bash
cat > ${OUTPUT_DIR}/merge.yaml <<EOF
model_name_or_path: ${BASE_MODEL}
adapter_name_or_path: ${OUTPUT_DIR}
template: minicpm5
finetuning_type: lora
export_dir: ./minicpm5-merged
export_size: 4
EOF
llamafactory-cli export ${OUTPUT_DIR}/merge.yaml
```

The merged model is a regular `LlamaForCausalLM` and serves with any `minicpm5-deploy-*` skill.

## Full SFT (no LoRA)

Replace `finetuning_type: lora` and the LoRA fields with `finetuning_type: full`. Add `deepspeed: examples/deepspeed/ds_z2_config.json` for multi-GPU.

## Reference

[`docs/finetune/llamafactory.md`](../../docs/finetune/llamafactory.md)
