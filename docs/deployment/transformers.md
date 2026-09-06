# Deploy MiniCPM5-1B and MiniCPM5-2B with 🤗 Transformers

MiniCPM5-1B and MiniCPM5-2B are standard `LlamaForCausalLM` models, so they load directly via `AutoModelForCausalLM` — no custom modeling code, no `trust_remote_code`.

## Install

```bash
pip install -U "transformers>=5.6,<6" "torch>=2.11" accelerate     # latest (CUDA 13.x driver hosts)
# pip install -U "transformers==4.57.3" "torch==2.7.1" accelerate  # fallback for CUDA 12.x driver hosts
```

## GPU inference (bfloat16)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "openbmb/MiniCPM5-2B"  # or openbmb/MiniCPM5-1B
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
).eval()

messages = [{"role": "user", "content": "用一句话解释什么是 GQA。"}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    enable_thinking=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
    )

prompt_len = inputs["input_ids"].shape[-1]
print(tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True))
```

## CPU-only inference

Both checkpoints can run on CPU only with sufficient system RAM (for example, on laptops, CI machines, or no-GPU sanity-check hosts):

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "openbmb/MiniCPM5-2B"  # or openbmb/MiniCPM5-1B
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,   # bf16 also works on AVX-512 BF16 / AMX hosts
    device_map="cpu",
).eval()

messages = [{"role": "user", "content": "用一句话解释什么是 GQA。"}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    enable_thinking=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
    )

prompt_len = inputs["input_ids"].shape[-1]
print(tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True))
```

## Generation defaults

| Mode | `enable_thinking` | `temperature` | `top_p` | When to use |
| --- | --- | --- | --- | --- |
| MiniCPM5-2B Think | `True` | 1.0 | 0.95 | only supported mode |
| MiniCPM5-1B Think | `True` | 0.9 | 0.95 | hard reasoning, math, code, multi-step |
| MiniCPM5-1B No-think | `False` | 0.7 | 0.95 | fast assistant, latency-bound |

`generation_config.json` is tuned for **think** mode by default.

## LoRA inference (PEFT)

```python
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained(
    "openbmb/MiniCPM5-2B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, "<your_lora_dir>").eval()
```

Adapters trained against this base load directly with no surgery.
