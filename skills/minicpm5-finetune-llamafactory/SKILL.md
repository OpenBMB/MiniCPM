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
pip install "llamafactory==0.9.3"
```

> ⚠️ LLaMA-Factory pins `transformers==4.52`. Do NOT install it into a vLLM env (vLLM 0.21 wants `transformers>=5.6`). Always use a separate venv.

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
template: empty                       # 🔑 MANDATORY for MiniCPM5 — delegates to model's own jinja
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

> 🔑 **`template: empty` is MANDATORY.** It delegates to the model's own `chat_template.jinja` (which is the MiniCPM5 ChatML template with think / nothink / tools support). Do NOT set `template: llama3` / `qwen` / etc. — those produce a corrupted token layout.

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

Coherent answer ⇒ ✅. Gibberish ⇒ check `template: empty` in the YAML.

## Merge LoRA for serving

```bash
cat > ${OUTPUT_DIR}/merge.yaml <<EOF
model_name_or_path: ${BASE_MODEL}
adapter_name_or_path: ${OUTPUT_DIR}
template: empty
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
