---
name: minicpm5-deploy-transformers
description: Run MiniCPM5-1B with Hugging Face Transformers for one-shot Python generation on GPU (bfloat16) or CPU (float32). Use when the user wants a quick Python script, no server, no extra deps, or asks for "Transformers", "AutoModelForCausalLM", "model.generate" with MiniCPM5.
---

# Deploy MiniCPM5-1B with HF Transformers

One-shot Python generation. No server. Works on a single GPU (bfloat16) or CPU only (fp32).

## Required input

| Var | Example | Default |
| --- | --- | --- |
| `MODEL_PATH` | `openbmb/MiniCPM5-1B` or local dir | required |
| `MODE` | `think` or `nothink` | `think` |

## Steps

### 1. Install (once)

```bash
pip install -U "transformers>=5.6,<6" "torch>=2.11" accelerate     # latest (CUDA 13.x driver hosts)
# pip install -U "transformers==4.57.3" "torch==2.7.1" accelerate  # fallback for CUDA 12.x driver hosts
```

### 2. Run

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "${MODEL_PATH}"      # ← replace
tok = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,    # CPU users: torch.float32 + device_map="cpu"
    device_map="auto",
).eval()

messages = [{"role": "user", "content": "用一句话解释什么是 GQA。"}]
inputs = tok.apply_chat_template(
    messages,
    add_generation_prompt=True,
    enable_thinking=True,          # set False for nothink mode
    return_tensors="pt",
).to(model.device)

with torch.no_grad():
    out = model.generate(
        inputs,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.9,           # nothink: 0.7
        top_p=0.95,
    )
print(tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True))
```

For CPU only: change `torch_dtype=torch.float32, device_map="cpu"` and drop `enable_thinking=True` (use False for latency).

## Sampling defaults

| Mode | `enable_thinking` | `temperature` | `top_p` |
| --- | --- | --- | --- |
| Think | `True` | 0.9 | 0.95 |
| No-think | `False` | 0.7 | 0.95 |

## Validate

A coherent answer to `1+1=?` (e.g. `"2"` or `"答案是 2"`).

## LoRA inference

```python
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, "/path/to/adapter").eval()
```

Adapters from any of the `minicpm5-finetune-*` skills load directly with no surgery.

## When NOT to use

- Need an OpenAI-compatible HTTP server → `minicpm5-deploy-vllm` or `minicpm5-deploy-sglang`
- Apple Silicon → `minicpm5-deploy-mlx` is faster
- CPU-only or low-VRAM laptop → `minicpm5-deploy-llama-cpp` with Q4_K_M is faster

## Reference

[`docs/deployment/transformers.md`](../../docs/deployment/transformers.md)
