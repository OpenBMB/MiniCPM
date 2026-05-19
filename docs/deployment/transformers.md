# Deploy MiniCPM5-1B with 🤗 Transformers

MiniCPM5-1B is a standard `LlamaForCausalLM`, so it loads directly via `AutoModelForCausalLM` — no custom modeling code, no `trust_remote_code`.

## Verified versions

| Component | Version |
| --- | --- |
| `transformers` | 4.51 + |
| `torch` | 2.1 + |
| `peft` | 0.11 + (optional, for LoRA inference) |
| Python | 3.9 – 3.12 |

## Install

```bash
pip install -U "transformers>=4.51" "torch>=2.1" accelerate
```

## GPU inference (bfloat16)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "openbmb/MiniCPM5-1B"
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
    enable_thinking=True,        # set False for fast / no-think mode
    return_tensors="pt",
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
    )
print(tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True))
```

Verified on a single H200 (bf16): **load ≈ 7 s, throughput ≈ 16 tok/s** for short prompts. Throughput rises with longer outputs once CUDA graphs are warm.

## CPU-only inference

The whole 1.08B-param model is < 4.5 GB in fp32, so it can run on CPU only (laptops, CI machines, no-GPU sanity checks):

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "openbmb/MiniCPM5-1B"
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
    enable_thinking=False,        # nothink is recommended for CPU latency
    return_tensors="pt",
)

with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
    )
print(tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True))
```

Verified on a server-class CPU: **fp32 throughput ≈ 20 tok/s** for short prompts. For lower memory and higher throughput on commodity laptops, use the GGUF builds with `llama.cpp` / `Ollama` / `LM Studio` (see [`llama_cpp.md`](./llama_cpp.md)).

## Generation defaults

| Mode | `enable_thinking` | `temperature` | `top_p` | When to use |
| --- | --- | --- | --- | --- |
| Think | `True` | 0.6 | 0.95 | hard reasoning, math, code, multi-step |
| No-think | `False` | 0.7 | 0.8 | fast assistant, latency-bound |

`generation_config.json` is tuned for **think** mode by default.

## LoRA inference (PEFT)

```python
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained(
    "openbmb/MiniCPM5-1B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, "<your_lora_dir>").eval()
```

Adapters trained against this base load directly with no surgery.
