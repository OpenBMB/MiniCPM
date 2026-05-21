<div align="center">
<img src="./assets/minicpm_logo.png" width="500em" ></img> 
</div>

<h4 align="center">
    <p>
        <a href="https://github.com/OpenBMB/MiniCPM/blob/main/README-cn.md">中文</a> | <b>English</b>
    <p>
</h4>

<p align="center">
<a href="https://arxiv.org/pdf/2506.07900" target="_blank">MiniCPM Paper</a> |
<a href="https://modelbest.feishu.cn/wiki/D2tFw8Pcsi5CIzkaHNacLK64npg" target="_blank">MiniCPM Wiki (in Chinese)</a> |
<a href="https://github.com/OpenBMB/MiniCPM-V/" target="_blank">MiniCPM-V Repo</a> |
Join our <a href="https://discord.gg/3cGQn9b3YM" target="_blank">discord</a> and <a href="https://github.com/OpenBMB/MiniCPM/blob/main/assets/wechat.jpg" target="_blank">WeChat</a> |
<a href="https://mp.weixin.qq.com/s/KIhH2nCURBXuFXAtYRpuXg?poc_token=HBIsUWijxino8oJ5s6HcjcfXFRi0Xj2LJlxPYD9c">Join Us</a>
</p>

> [!NOTE]
> ### 🏆 2026 Sparse Operator Acceleration & Race (SOAR) is Now Live!
>
> **The MiniCPM-SALA architecture is just the beginning. Realizing its full potential requires deep system-level synergy and cross-layer compilation optimization.**
>
> OpenBMB, in collaboration with **SGLang** and **NVIDIA**, invites global geeks to tackle the limits of **9B-scale, 1M-token inference** on a dedicated **NVIDIA 6000D** environment.
>
> * 💰 **Prize Pool:** >$100,000 USD (Top Prize: **$89,000**)
> * 🚀 **Goal:** Optimize single and multi-batch performance via cross-layer compilation.
>
> 👉 **[Learn more and Register](https://soar.openbmb.cn/)**

## Changelog🔥
- 📌 [2026.05.19] **[MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B)** is released! A compact dense 1B-class LLM that **tops every public sub-2B leaderboard** across reasoning, knowledge, math, code, instruction-following and agentic use. It is built for on-device and resource-constrained deployment, and ships with [Agent Skills](./skills/) for one-prompt inference and fine-tuning. 🔥🔥🔥
- [2026.02.11] **[MiniCPM-SALA](https://huggingface.co/openbmb/MiniCPM-SALA)** is released! This is the first large-scale hybrid model effectively integrating sparse and linear attention for million-token context modeling. 🔥🔥🔥
- [2025.09.05] **[MiniCPM4.1 series](https://huggingface.co/collections/openbmb/minicpm-4-6841ab29d180257e940baa9b)** are released! This series is a hybrid reasoning model with trainable sparse attention, which can be used in both deep reasoning mode and non-reasoning mode. 🔥🔥🔥
- [2025.06.06] Released [**MiniCPM4**](https://huggingface.co/collections/openbmb/minicpm-4-6841ab29d180257e940baa9b)! This model achieves ultimate efficiency improvements while maintaining optimal performance at the same scale! It can achieve over 5x generation acceleration on typical end-side chips!

<details>
<summary>Older entries (2024 + InfLLM-V2 paper)</summary>

- [2025.09.29] **[InfLLM-V2 paper](https://arxiv.org/abs/2509.24663) is released!** We can train a sparse attention model with only 5B long-text tokens.
- [2024.09.05] We release [**MiniCPM3-4B**](https://huggingface.co/openbmb/MiniCPM3-4B)! This model outperforms Phi-3.5-mini-instruct and GPT-3.5-Turbo-0125 and is comparable to several models with 7B-9B parameters like Llama3.1-8B-Instruct, Qwen2-7B-Instruct, and GLM-4-9B-Chat.
- [2024.07.05] Released [**MiniCPM-S-1B**](https://huggingface.co/openbmb/MiniCPM-S-1B-sft)! This model achieves an average sparsity of 87.89% in the FFN layer, reducing FFN FLOPs by 84%, while maintaining downstream task performance.
- [2024.04.11] Released [**MiniCPM-2B-128k**](https://huggingface.co/openbmb/MiniCPM-2B-128k), [**MiniCPM-MoE-8x2B**](https://huggingface.co/openbmb/MiniCPM-MoE-8x2B) and [**MiniCPM-1B**](https://huggingface.co/openbmb/MiniCPM-1B-sft-bf16)! Click [here](https://openbmb.vercel.app/) to read our technical blog.
- [2024.02.01] Released [**MiniCPM-2B**](https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16)! This model performs similarly to Mistral-7B on public benchmarks (with better performance in Chinese, math, and code abilities) and overall outperforms models like Llama2-13B, MPT-30B, and Falcon-40B.

</details>

## Quick Links

- [Changelog🔥](#changelog)
- [Model Downloads](#model-downloads)
- [MiniCPM5 Series](#minicpm5-series)
  - [Highlights](#highlights)
  - [Introduction](#introduction)
  - [Training Recipe](#training-recipe)
  - [Evaluation Results](#evaluation-results)
    - [Standard Benchmarks](#standard-benchmarks)
    - [What Does RL Improve?](#what-does-rl-improve)
  - [How to Run MiniCPM5-1B in One Prompt?](#how-to-run-minicpm5-1b-in-one-prompt)
  - [Deployment and Fine-tuning Cookbooks](#deployment-and-fine-tuning-cookbooks)
  - [MiniCPM5 Applications](#minicpm5-applications)
    - [Desktop Pet](#desktop-pet)
    - [Persona LoRA Hub](#persona-lora-hub)
- [MiniCPM-SALA](#minicpm-sala)
- [MiniCPM4 & MiniCPM4.1 Series](#minicpm4-and-minicpm41-series)
- [Legacy topics →](./docs/README-legacy.md): BitCPM4 quantization, MiniCPM4 applications
- [LICENSE](#license) · [Institutions](#institutions) · [Citation](#citation)


## Model Downloads

**Current release: MiniCPM5-1B** (BF16, GGUF, MLX, AWQ, GPTQ):

| HuggingFace | ModelScope |
|---|---|
| [MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) | [MiniCPM5-1B](https://www.modelscope.cn/models/OpenBMB/MiniCPM5-1B) |
| [MiniCPM5-1B-GGUF](https://huggingface.co/openbmb/MiniCPM5-1B-GGUF) | [MiniCPM5-1B-GGUF](https://www.modelscope.cn/models/OpenBMB/MiniCPM5-1B-GGUF) |
| [MiniCPM5-1B-MLX](https://huggingface.co/openbmb/MiniCPM5-1B-MLX) | [MiniCPM5-1B-MLX](https://www.modelscope.cn/models/OpenBMB/MiniCPM5-1B-MLX) |
| [MiniCPM5-1B-AWQ](https://huggingface.co/openbmb/MiniCPM5-1B-AWQ) | [MiniCPM5-1B-AWQ](https://www.modelscope.cn/models/OpenBMB/MiniCPM5-1B-AWQ) |
| [MiniCPM5-1B-GPTQ](https://huggingface.co/openbmb/MiniCPM5-1B-GPTQ) | [MiniCPM5-1B-GPTQ](https://www.modelscope.cn/models/OpenBMB/MiniCPM5-1B-GPTQ) |

<details>
<summary>📋 Click to view earlier MiniCPM releases: SALA, 4.1 / 4, BitCPM, applications, MiniCPM3 / 2B / 1B</summary>

**Earlier flagships:**

| HuggingFace | ModelScope |
|---|---|
| [MiniCPM-SALA](https://huggingface.co/openbmb/MiniCPM-SALA) | [MiniCPM-SALA](https://www.modelscope.cn/models/OpenBMB/MiniCPM-SALA) |
| [MiniCPM4.1-8B](https://huggingface.co/openbmb/MiniCPM4.1-8B) | [MiniCPM4.1-8B](https://www.modelscope.cn/models/OpenBMB/MiniCPM4.1-8B) |
| [MiniCPM4-8B](https://huggingface.co/openbmb/MiniCPM4-8B) | [MiniCPM4-8B](https://www.modelscope.cn/models/OpenBMB/MiniCPM4-8B) |
| [MiniCPM4-0.5B](https://huggingface.co/openbmb/MiniCPM4-0.5B) | [MiniCPM4-0.5B](https://www.modelscope.cn/models/OpenBMB/MiniCPM4-0.5B) |

**MiniCPM4.1 quantized & speculative variants:**

| HuggingFace | ModelScope |
|---|---|
| [MiniCPM4.1-8B-GPTQ](https://huggingface.co/openbmb/MiniCPM4.1-8B-GPTQ) | [MiniCPM4.1-8B-GPTQ](https://www.modelscope.cn/openbmb/MiniCPM4.1-8B-GPTQ) |
| [MiniCPM4.1-8B-AutoAWQ](https://huggingface.co/openbmb/MiniCPM4.1-8B-AutoAWQ) | [MiniCPM4.1-8B-AutoAWQ](https://www.modelscope.cn/openbmb/MiniCPM4.1-8B-AutoAWQ) |
| [MiniCPM-4.1-8B-Marlin](https://huggingface.co/openbmb/MiniCPM-4.1-8B-Marlin) | [MiniCPM-4.1-8B-Marlin](https://www.modelscope.cn/openbmb/MiniCPM-4.1-8B-Marlin) |
| [MiniCPM4.1-8B-GGUF](https://huggingface.co/openbmb/MiniCPM4.1-8B-GGUF) | [MiniCPM4.1-8B-GGUF](https://www.modelscope.cn/openbmb/MiniCPM4.1-8B-GGUF) |
| [MiniCPM4.1-8B-MLX](https://huggingface.co/openbmb/MiniCPM4.1-8B-MLX) | [MiniCPM4.1-8B-MLX](https://www.modelscope.cn/openbmb/MiniCPM4.1-8B-MLX) |
| [MiniCPM4.1-8B-Eagle3](https://huggingface.co/openbmb/MiniCPM4.1-8B-Eagle3) | [MiniCPM4.1-8B-Eagle3](https://www.modelscope.cn/openbmb/MiniCPM4.1-8B-Eagle3) |

**BitCPM4 ternary-quantized + MiniCPM4 Applications:**

| HuggingFace | ModelScope |
|---|---|
| [BitCPM4-1B](https://huggingface.co/openbmb/BitCPM4-1B) | [BitCPM4-1B](https://www.modelscope.cn/models/OpenBMB/BitCPM4-1B) |
| [BitCPM4-0.5B](https://huggingface.co/openbmb/BitCPM4-0.5B) | [BitCPM4-0.5B](https://www.modelscope.cn/models/OpenBMB/BitCPM4-0.5B) |
| [MiniCPM4-Survey](https://huggingface.co/openbmb/MiniCPM4-Survey) | [MiniCPM4-Survey](https://www.modelscope.cn/models/OpenBMB/MiniCPM4-Survey) |
| [MiniCPM4-MCP](https://huggingface.co/openbmb/MiniCPM4-MCP) | [MiniCPM4-MCP](https://www.modelscope.cn/models/OpenBMB/MiniCPM4-MCP) |

**MiniCPM4 Eagle speculative decoding, QAT, and pre-2025 releases:**

| HuggingFace | ModelScope |
|---|---|
| [MiniCPM4-8B-Eagle-FRSpec](https://huggingface.co/openbmb/MiniCPM4-8B-Eagle-FRSpec) | [MiniCPM4-8B-Eagle-FRSpec](https://www.modelscope.cn/models/OpenBMB/MiniCPM4-8B-Eagle-FRSpec) |
| [MiniCPM4-8B-Eagle-FRSpec-QAT](https://huggingface.co/openbmb/MiniCPM4-8B-Eagle-FRSpec-QAT) | [MiniCPM4-8B-Eagle-FRSpec-QAT](https://www.modelscope.cn/models/OpenBMB/MiniCPM4-8B-Eagle-FRSpec-QAT) |
| [MiniCPM4-8B-Eagle-vLLM](https://huggingface.co/openbmb/MiniCPM4-8B-Eagle-vLLM) | [MiniCPM4-8B-Eagle-vLLM](https://www.modelscope.cn/models/OpenBMB/MiniCPM4-8B-Eagle-vLLM) |
| [MiniCPM4-8B-marlin-Eagle-vLLM](https://huggingface.co/openbmb/MiniCPM4-8B-marlin-Eagle-vLLM) | [MiniCPM4-8B-marlin-Eagle-vLLM](https://www.modelscope.cn/models/OpenBMB/MiniCPM4-8B-marlin-Eagle-vLLM) |
| [MiniCPM4-0.5B-QAT-Int4-unquantized](https://huggingface.co/openbmb/MiniCPM4-0.5B-QAT-Int4-unquantized) | [MiniCPM4-0.5B-QAT-Int4-unquantized](https://modelscope.cn/models/OpenBMB/MiniCPM4-0.5B-QAT-Int4-unquantized) |
| [MiniCPM4-0.5B-QAT-Int4-GPTQ-format](https://huggingface.co/openbmb/MiniCPM4-0.5B-QAT-Int4-GPTQ-format) | [MiniCPM4-0.5B-QAT-Int4-GPTQ-format](https://modelscope.cn/models/OpenBMB/MiniCPM4-0.5B-QAT-Int4-GPTQ-format) |
| [MiniCPM3-4B](https://huggingface.co/openbmb/MiniCPM3-4B) | [MiniCPM3-4B](https://www.modelscope.cn/models/OpenBMB/MiniCPM3-4B) |
| [MiniCPM-2B-sft](https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16) | [MiniCPM-2B-sft](https://modelscope.cn/models/OpenBMB/miniCPM-bf16) |
| [MiniCPM-2B-dpo](https://huggingface.co/openbmb/MiniCPM-2B-dpo-bf16) | [MiniCPM-2B-dpo](https://modelscope.cn/models/OpenBMB/MiniCPM-2B-dpo-bf16/summary) |
| [MiniCPM-2B-128k](https://huggingface.co/openbmb/MiniCPM-2B-128k) | [MiniCPM-2B-128k](https://modelscope.cn/models/openbmb/MiniCPM-2B-128k/summary) |
| [MiniCPM-MoE-8x2B](https://huggingface.co/openbmb/MiniCPM-MoE-8x2B) | [MiniCPM-MoE-8x2B](https://modelscope.cn/models/OpenBMB/MiniCPM-MoE-8x2B) |
| [MiniCPM-1B](https://huggingface.co/openbmb/MiniCPM-1B-sft-bf16) | [MiniCPM-1B](https://modelscope.cn/models/OpenBMB/MiniCPM-1B-sft-bf16) |
| [MiniCPM-S-1B](https://huggingface.co/openbmb/MiniCPM-S-1B-sft) | [MiniCPM-S-1B](https://modelscope.cn/models/OpenBMB/MiniCPM-S-1B-sft) |

</details>

## MiniCPM5 Series

MiniCPM5 is our next-generation end-side model family. The first release, **MiniCPM5-1B**, is a compact dense 1B Transformer designed to maximize quality per parameter at the 1B scale, with a heavy emphasis on RL training, single-page cookbooks, and an agent driven deployment story.

### Highlights

🏆 **Strongest 1B-class model on public leaderboards**: leads the **average score** (43.56) against LFM2.5-1.2B-Thinking, Qwen3-0.6B/think and Qwen3.5-0.8B/think across 23 reasoning, knowledge, code, instruction-following and agentic benchmarks, while also being the smallest by total parameters.

🧩 **Standard Architecture**: `LlamaForCausalLM` with **GQA (16 Q / 2 KV)** and **SwiGLU**. Runs on every mainstream engine without custom kernels.

📚 **Native 128 K Context**: `max_position_embeddings = 131,072`, `rope_theta = 5e6`, no RoPE scaling needed.

🧠 **Dual Mode Reasoning**: built-in `<think>` chat template, switch via `enable_thinking`. The same checkpoint serves as both a fast assistant and a deliberate reasoner.

🛠️ **Agent Skills and Cookbooks**: every inference and fine-tuning path in this repo ships with a single-page cookbook and a paired [Agent Skill](./skills/) for one-prompt deployment by any LLM coding agent.

🎮 **MiniCPM5 Applications**: reference apps showing what a 1B model can run on-device today: a desktop pet powered locally by MiniCPM5-1B and a community driven **Persona LoRA Hub** for personality LoRAs. See [MiniCPM5 Applications](#minicpm5-applications).

### Introduction

MiniCPM5-1B is a compact dense decoder-only Transformer trained to maximize quality per parameter. It keeps the standard `LlamaForCausalLM` architecture (24 layers, GQA 8:1, native 128K context, ~1.0 B total params) so it runs out-of-the-box on every mainstream inference engine (Transformers, vLLM, SGLang, llama.cpp, MLX, Ollama, LM Studio…) without custom kernels.

For full architecture details and per-component parameter breakdown see [`docs/deployment/transformers.md`](./docs/deployment/transformers.md).

### Training Recipe

The training of MiniCPM5-1B is a full-stack practice of the **UltraData hierarchical data governance system**, covering both staged pre-training and post-training.

During **pre-training**, the model goes through two stable-training stages with **1T tokens** each, followed by **200B tokens of decay training** and **200B tokens of mid-training** to further align capability targets and data distribution.

During **post-training**, we continue with **200B tokens of deep-thinking SFT** and **200B tokens of hybrid-thinking SFT** to strengthen reasoning and general chat abilities. On top of that, domain-specific Reinforcement Learning (RL) and **On-Policy Distillation (OPD)** integrate specialized RL teachers into the final release model, improving capability while merging multiple training branches into one checkpoint.

![MiniCPM5-1B Training Recipe](./assets/minicpm5/training_recipe.png)

### Evaluation Results

#### Standard Benchmarks

MiniCPM5-1B is benchmarked against the closest open-source 1B-class peers: **LFM2.5-1.2B-Thinking**, **Qwen3-0.6B/think** and **Qwen3.5-0.8B/think**. Across 23 public benchmarks covering general and domain knowledge, code, instruction-following, math, logical reasoning, subjective writing and agentic tool use, **MiniCPM5-1B is the smallest model by parameter count** and **wins the overall average by a wide margin (43.56 vs. next-best 34.52)**. It leads or ties on 18 of the 21 benchmarks where we report a score.

![MiniCPM-5 1B Public Leaderboard](./assets/minicpm5/public_leaderboard.png)

#### What Does RL Improve?

RL training delivers the largest single jump in MiniCPM5-1B's intelligence: it turns the SFT checkpoint into a usable assistant for reasoning and instruction-following workloads.

![MiniCPM5-1B RL Gains](./assets/minicpm5/rl_gains.png)

Across seven math, code, and instruction-following benchmarks, RL raises the average score by **↑16.0 points**. The largest gains come from math (AIME 2025 ↑20.2 / AIME 2026 ↑25.6 / HMMT ↑11.6) and instruction-following (IFBench ↑17.1, Multi-IF ↑15.1, IFEval ↑10.7); LCB-v6 also improves by ↑11.6.

RL also makes the model **dramatically less verbose** on reasoning tasks: the share of responses truncated at the max-tokens budget drops by **↓29 pp on average** (HMMT ↓40.5 pp, AIME 2025 ↓32.7 pp). In practice, RL raises scores while producing shorter, more focused reasoning traces for latency-sensitive on-device deployment.

![MiniCPM5-1B RL Overlong Response Rate Drop](./assets/minicpm5/rl_overlong.png)

### How to Run MiniCPM5-1B in One Prompt?

MiniCPM5-1B uses the **standard `LlamaForCausalLM` architecture** and runs out of the box on every mainstream engine: **no custom kernels, no model-code fork**. We adapted MiniCPM5-1B to **9 inference backends** and **5 fine-tuning frameworks**, and shipped two top-level [Cursor Agent Skills](https://docs.cursor.com/agent/skills) so any LLM coding agent (Cursor / Claude Code / Codex / opencode / …) can drive them **from a single natural-language prompt**.

| Top-level skill | What it does | Routes to |
| --- | --- | --- |
| **[`minicpm5-deploy`](./skills/minicpm5-deploy/SKILL.md)** | Inference router | `transformers` · `vllm` · `sglang` · `awq` · `gptq` · `llama-cpp` · `ollama` · `lmstudio` · `mlx` |
| **[`minicpm5-finetune`](./skills/minicpm5-finetune/SKILL.md)** | Fine tuning router | `trl` · `llamafactory` · `ms-swift` · `unsloth` · `xtuner` |

Drop a line like this into Cursor / Claude Code and the agent picks the right sub skill, sets up the env, runs the command, and reports back:

```
@minicpm5-deploy   serve openbmb/MiniCPM5-1B with vLLM on port 8000
@minicpm5-finetune use unsloth + LoRA on /data/my_chat.jsonl, write to ./out
```

Recommended chat template sampling:

| Mode | Recommended params | Enable |
| --- | --- | --- |
| **Think** | `temperature=0.6, top_p=0.95` | `enable_thinking=True` |
| **No Think** | `temperature=0.7, top_p=0.8` | `enable_thinking=False` |

### Deployment and Fine-tuning Cookbooks

Prefer to drive things by hand, or want to know exactly what each Agent Skill does under the hood? Every backend / framework has a single-page cookbook with the exact command and observed output, paired with a backend-specific sub skill.

**Deployment** (9 backends)

| Backend | Cookbook | Paired Agent Skill |
| --- | --- | --- |
| Transformers (GPU + CPU) | [`docs/deployment/transformers.md`](./docs/deployment/transformers.md) | [`minicpm5-deploy-transformers`](./skills/minicpm5-deploy-transformers/SKILL.md) |
| vLLM (OpenAI server) | [`docs/deployment/vllm.md`](./docs/deployment/vllm.md) | [`minicpm5-deploy-vllm`](./skills/minicpm5-deploy-vllm/SKILL.md) |
| SGLang (OpenAI server) | [`docs/deployment/sglang.md`](./docs/deployment/sglang.md) | [`minicpm5-deploy-sglang`](./skills/minicpm5-deploy-sglang/SKILL.md) |
| AWQ-Marlin Int4 (vLLM) | [`docs/deployment/awq.md`](./docs/deployment/awq.md) | [`minicpm5-deploy-awq`](./skills/minicpm5-deploy-awq/SKILL.md) |
| GPTQ-Marlin Int4 (vLLM) | [`docs/deployment/gptq.md`](./docs/deployment/gptq.md) | [`minicpm5-deploy-gptq`](./skills/minicpm5-deploy-gptq/SKILL.md) |
| llama.cpp (GGUF, CPU/GPU) | [`docs/deployment/llama_cpp.md`](./docs/deployment/llama_cpp.md) | [`minicpm5-deploy-llama-cpp`](./skills/minicpm5-deploy-llama-cpp/SKILL.md) |
| Ollama (GGUF, end-side) | [`docs/deployment/ollama.md`](./docs/deployment/ollama.md) | [`minicpm5-deploy-ollama`](./skills/minicpm5-deploy-ollama/SKILL.md) |
| LM Studio (Mac, OpenAI server) | [`docs/deployment/lmstudio.md`](./docs/deployment/lmstudio.md) | [`minicpm5-deploy-lmstudio`](./skills/minicpm5-deploy-lmstudio/SKILL.md) |
| MLX (Apple Silicon) | [`docs/deployment/mlx.md`](./docs/deployment/mlx.md) | [`minicpm5-deploy-mlx`](./skills/minicpm5-deploy-mlx/SKILL.md) |

**Fine tuning** (5 frameworks)

| Framework | Cookbook | Paired Agent Skill |
| --- | --- | --- |
| [TRL](https://github.com/huggingface/trl) + [PEFT](https://github.com/huggingface/peft) | [`docs/finetune/trl.md`](./docs/finetune/trl.md) | [`minicpm5-finetune-trl`](./skills/minicpm5-finetune-trl/SKILL.md) |
| [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | [`docs/finetune/llamafactory.md`](./docs/finetune/llamafactory.md) | [`minicpm5-finetune-llamafactory`](./skills/minicpm5-finetune-llamafactory/SKILL.md) |
| [ms-swift](https://github.com/modelscope/ms-swift) | [`docs/finetune/ms_swift.md`](./docs/finetune/ms_swift.md) | [`minicpm5-finetune-ms-swift`](./skills/minicpm5-finetune-ms-swift/SKILL.md) |
| [unsloth](https://github.com/unslothai/unsloth) | [`docs/finetune/unsloth.md`](./docs/finetune/unsloth.md) | [`minicpm5-finetune-unsloth`](./skills/minicpm5-finetune-unsloth/SKILL.md) |
| [xtuner](https://github.com/InternLM/xtuner) | [`docs/finetune/xtuner.md`](./docs/finetune/xtuner.md) | [`minicpm5-finetune-xtuner`](./skills/minicpm5-finetune-xtuner/SKILL.md) |

### MiniCPM5 Applications

Reference apps built on top of MiniCPM5-1B, showing what a 1B-class on-device model can power in real world scenarios. Both apps are open and accept community contributions.

#### Desktop Pet

MiniCPM5-1B is small enough and capable enough to be the **local LLM brain** for an interactive desktop pet. We provide a thin bridge service, **`minicpm-pet-bridge`**, that exposes the model as an OpenAI-compatible local endpoint for [Clawd on Desk](https://github.com/rullerzhou-afk/clawd-on-desk), a pixel desktop pet that reacts to your AI coding agent in real time.

> Thanks to [@rullerzhou-afk](https://github.com/rullerzhou-afk) for building Clawd on Desk. The pet runtime, animation packs, and multi-agent integrations are all upstream work; `minicpm-pet-bridge` is the local LLM adapter.

One-liner to try the pet with a locally served MiniCPM5-1B:

```bash
# 1) start MiniCPM5-1B via vLLM (any deploy backend works)
python -m vllm.entrypoints.openai.api_server \
    --model openbmb/MiniCPM5-1B --served-model-name MiniCPM5-1B \
    --dtype bfloat16 --port 8000

# 2) point Clawd on Desk at http://localhost:8000/v1 (OpenAI-compatible)
#    → see Clawd's settings → MiniCPM tab for the GUI version
```

#### Persona LoRA Hub

Beyond the base assistant, we are launching the **MiniCPM5 Persona LoRA Hub**: a community-driven space where anyone can upload a persona dataset (character / mascot / role-play / customer-service / …), get it **labeled and trained by us into a published LoRA**, and have their contribution credited on the hub.

- **Hub**: `openbmb/minicpm5-persona-lora-hub` on Hugging Face Spaces
- **How to contribute**: dataset format, submission steps, attribution policy → [`docs/PERSONA_LORA_HUB-en.md`](./docs/PERSONA_LORA_HUB-en.md) (中文版：[`docs/PERSONA_LORA_HUB-cn.md`](./docs/PERSONA_LORA_HUB-cn.md))
- **First example**: `lora_nekoqa_adapter`, a cat-girl chat persona

If your dataset is accepted and the resulting LoRA is published, you will be credited by name / handle on both the hub page and in the LoRA's `README.md`.

## MiniCPM-SALA

> First large-scale **Sparse + Linear Attention hybrid** for million-token context (2026-02). [HuggingFace](https://huggingface.co/openbmb/MiniCPM-SALA) · [ModelScope](https://www.modelscope.cn/models/OpenBMB/MiniCPM-SALA)

<details>
<summary>Click to expand: highlights, evaluation, inference setup</summary>

#### Highlights

MiniCPM-SALA (Sparse Attention and Linear Attention) is the first large-scale hybrid model effectively integrating sparse and linear attention for million-token context modeling

✅ Innovative Hybrid Architecture: Synergizes 25% Sparse Attention (InfLLM-v2) for high-fidelity long context modeling with 75% Linear Attention (Lightning Attention) for global efficiency.

✅ Shattering Efficiency Walls: Breaks the "Compute Wall" and the "Memory Wall," achieving 3.5× inference speed and significantly lower KV-cache overhead compared to dense baselines. 

✅ Million-Token Context: Empowered by HyPE (Hybrid Positional Embedding), it scales to 1M+ tokens while maintaining strong length generalization. 

✅ HALO Adaptation: Utilizes Hybrid Attention via Layer Optimization (HALO), a novel distillation recipe that effectively transfers dense attention capabilities to the hybrid architecture, avoiding the severe performance degradation typical of pure linear models.

#### Introduction

MiniCPM-SALA is an efficient hybrid model in which 25% of the layers adopt [InfLLM-V2](https://arxiv.org/abs/2509.24663) and the remaining 75% utilize Lightning Attention. This architecture enables inference of one million tokens on consumer GPUs such as the NVIDIA RTX 5090.

- **SALA Hybrid Attention Mechanism**
  - Integrates 25% InfLLM-V2 and 75% Lightning Attention, effectively leveraging the granular focus of sparse attention for local details and the high efficiency of linear attention for broad context.

- **Transformer-to-Hybrid Continue Training**
  - Circumvents the inefficiencies of cold-start training by performing an architectural transformation on the pre-trained weights, thereby reducing the total training budget to approximately 25% relative to training a comparable model from scratch.

- **[HyPE](https://arxiv.org/abs/2601.22156) (Hybrid Positional Encoding)**
  - Harmonizes the performance across both short and long contexts, which can maintain general capabilities (e.g., knowledge, mathematics, and coding) comparable to modern full-attention models like Qwen3-8B and achieve substantial advantages across multiple long-context benchmarks.

- **Efficient Inference on Long Sequences**
  - Achieves up to 3.5x the inference speed of Qwen3-8B at a sequence length of 256K tokens on A6000D, supports inference at context lengths of up to 1M tokens on both NVIDIA A6000D and 5090 GPUs, whereas Qwen3-8B fails at this length due to out-of-memory (OOM) errors.

### Evaluation Results

#### Efficiency Evaluation

We benchmarked MiniCPM-SALA (9B) against Qwen3-8B on NVIDIA A6000D and RTX 5090 GPUs to evaluate inference speed and memory efficiency. The results demonstrate a significant performance leap: MiniCPM-SALA not only achieves up to a 2.5x speedup in time-to-first-token (TTFT) but also overcomes the memory bottlenecks of full-attention architectures. While Qwen3-8B suffers from OOM errors at extended lengths, MiniCPM-SALA successfully scales to 1M-token contexts on a single consumer-grade RTX 5090, effectively democratizing ultra-long context inference on edge hardware.

![inference_speed_a6000d](./assets/minicpm_sala/inference_speed_a600d.png)

![inference_speed_5090](./assets/minicpm_sala/inference_speed_5090.png)

#### Long-Context Evaluation

MiniCPM-SALA consistently outperforms other open-source LLMs of similar scale across most involved long-context benchmarks. Specifically, it achieves the highest scores in the RULER and NoLiMa tests at all context lengths (up to 128K) and maintains the highest overall average score of 38.97, suggesting superior performance in handling long-context information processing.

![long_text_evaluation](./assets/minicpm_sala/long_text_evaluation.png)

#### Ultra-long Context Evaluation

The evaluation demonstrates that MiniCPM-SALA exhibits effective length extrapolation capabilities, maintaining a score of 81.6 at a 2048K context length despite being trained on only 520K tokens. The model achieves this without auxiliary techniques like YaRN, likely due to its NoPE configuration in sparse attention layers.

![ultra_long_text_evaluation](./assets/minicpm_sala/ultra_long_text_evaluation.png)

#### Standard Evaluation

MiniCPM-SALA achieves an average score of 76.53 across standard benchmarks, outperforming comparable models such as Qwen3-8B and Falcon-H1R-7B. The architecture maintains robust performance in Knowledge, Code, and Math.

![benchmark](./assets/minicpm_sala/benchmark.png)

### Inference

To achieve optimal performance, we recommend using the `Temperature=0.9`.

#### HuggingFace

Our model is readily compatible with 🤗 Hugging Face transformers. You can perform inference with our model as follows:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "openbmb/MiniCPM-SALA"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
model.eval()

prompts = ["My name is", "The capital of China is"]
with torch.no_grad():
    inputs = tokenizer(prompts, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs)
output_texts = tokenizer.batch_decode(outputs)
print(output_texts)
```

#### SGLang

##### Requirements

- CUDA 12.x or higher
- `gcc` / `g++` compiler
- `uv` package manager (script will check)

##### Installation

```bash
# Clone repository
git clone -b minicpm_sala https://github.com/OpenBMB/sglang.git
cd sglang

# One-click installation (creates venv and compiles all dependencies)
bash install_minicpm_sala.sh

# Or specify PyPI mirror
bash install_minicpm_sala.sh https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

The installation script performs the following steps:

1. Creates `sglang_minicpm_sala_env` virtual environment (Python 3.12)
2. Clones dependencies to `3rdparty/` (infllmv2) and initializes submodules (sparse_kernel)
3. Installs MiniCPM-SALA (current repo)
4. Compiles and installs `infllmv2_cuda_impl`
5. Compiles and installs `sparse_kernel`
6. Installs `tilelang` & `flash-linear-attention`

##### Usage

```bash
# Activate environment
source sglang_minicpm_sala_env/bin/activate

# Launch Inference Server (Replace MODEL_PATH with actual path)
MODEL_PATH=/path/to/your/MiniCPM-SALA

python3 -m sglang.launch_server \
    --model ${MODEL_PATH} \
    --trust-remote-code \
    --disable-radix-cache \
    --attention-backend minicpm_flashinfer \
    --chunked-prefill-size 8192 \
    --max-running-requests 32 \
    --skip-server-warmup \
    --port 31111 \
    --dense-as-sparse
```

| Parameter | Description |
|-----------|-------------|
| `--trust-remote-code` | Allow custom code in model |
| `--disable-radix-cache` | Disable RadixAttention prefix cache |
| `--attention-backend minicpm_flashinfer` | Use MiniCPM FlashInfer backend |
| `--chunked-prefill-size 8192` | Chunked prefill size |
| `--max-running-requests 32` | Max concurrent requests |
| `--skip-server-warmup` | Skip server warmup |
| `--port 31111` | Server port |
| `--dense-as-sparse` | Use dense-as-sparse mode |

##### Manual Installation

If the script doesn't work for you, follow these steps:

```bash
# 0. Ensure uv is installed
pip install uv

# 1. Create venv
uv venv --python 3.12 sglang_minicpm_sala_env
source sglang_minicpm_sala_env/bin/activate

# 2. Install SGLang
uv pip install --upgrade pip setuptools wheel
uv pip install -e ./python[all]

# 3. Compile CUDA Extensions
# (Ensure dependencies are cloned to 3rdparty/)
cd 3rdparty/infllmv2_cuda_impl && python setup.py install && cd ../..
cd 3rdparty/sparse_kernel && python setup.py install && cd ../..

# 4. Install extra deps
uv pip install tilelang flash-linear-attention
```

##### Q&A

**Q: CUDA extension compilation failed?**

- Ensure CUDA 12+ is installed (`nvcc --version`).
- Ensure `gcc` / `g++` are available.
- If `CXX` is set to `clang++ -pthread`, manually `export CXX=g++`.

</details>

## MiniCPM4 and MiniCPM4.1 Series

> 8B-scale **trainable sparse attention** with hybrid reasoning (2025-09 / 2025-06). [MiniCPM4.1-8B](https://huggingface.co/openbmb/MiniCPM4.1-8B) · [MiniCPM4-8B](https://huggingface.co/openbmb/MiniCPM4-8B) · [ModelScope](https://www.modelscope.cn/models/OpenBMB/MiniCPM4.1-8B)

<details>
<summary>Click to expand: highlights, evaluation, inference (HF / vLLM / SGLang / CPM.cu / llama.cpp / Ollama)</summary>

<div align="center">
  <a href="https://www.youtube.com/watch?v=VouXjLHKDUY"><img src="https://img.youtube.com/vi/VouXjLHKDUY/0.jpg", width=70%></a>
</div>

#### Highlights
MiniCPM 4.1-8B is the first open-source reasoning LLM with trainable sparse attention:

✅ Strong Reasoning Capability: Surpasses similar-sized models on 15 tasks!

✅ Fast Generation: 3x decoding speedup for reasoning

✅ Efficient Architecture: Trainable sparse attention, frequency-ranked speculative decoding

#### Introduction
MiniCPM4 and MiniCPM4.1 series are highly efficient large language models (LLMs) designed explicitly for end-side devices, which achieves this efficiency through systematic innovation in four key dimensions: model architecture, training data, training algorithms, and inference systems.

- 🏗️ **Efficient Model Architecture:**
  - InfLLM-V2 -- Trainable Sparse Attention Mechanism: Adopts a trainable sparse attention mechanism architecture where each token only needs to compute relevance with less than 5% of tokens in 128K long text processing, significantly reducing computational overhead for long texts ([InfLLM-V2 Training Kernels](https://github.com/OpenBMB/infllmv2_cuda_impl))

- 🧠 **Efficient Learning Algorithms:**
  - Model Wind Tunnel 2.0 -- Efficient Predictable Scaling: Introduces scaling prediction methods for performance of downstream tasks, enabling more precise model training configuration search
  - BitCPM -- Ultimate Ternary Quantization: Compresses model parameter bit-width to 3 values, achieving 90% extreme model bit-width reduction
  - Efficient Training Engineering Optimization: Adopts FP8 low-precision computing technology combined with Multi-token Prediction training strategy

- 📚 **High-Quality Training Data:**
  - UltraClean -- High-quality Pre-training Data Filtering and Generation: Builds iterative data cleaning strategies based on efficient data verification, open-sourcing high-quality Chinese and English pre-training dataset [UltraFinweb](https://huggingface.co/datasets/openbmb/Ultra-FineWeb)
  - UltraChat v2 -- High-quality Supervised Fine-tuning Data Generation: Constructs large-scale high-quality supervised fine-tuning datasets covering multiple dimensions including knowledge-intensive data, reasoning-intensive data, instruction-following data, long text understanding data, and tool calling data

- ⚡ **Efficient Inference and Deployment System:**
  - CPM.cu -- Lightweight and Efficient CUDA Inference Framework: Integrates sparse attention, model quantization, and speculative sampling to achieve efficient prefilling and decoding ([Inference Kernels and Framework](https://github.com/openbmb/cpm.cu))
  - ArkInfer -- Cross-platform Deployment System: Supports efficient deployment across multiple backend environments, providing flexible cross-platform adaptation capabilities

### Evaluation Results

#### Efficiency Evaluation
On two typical end-side chips, Jetson AGX Orin and RTX 4090, MiniCPM4 and MiniCPM4.1 demonstrate significantly faster processing speed compared to similar-size models in long text processing tasks. As text length increases, MiniCPM4 and MiniCPM4.1's efficiency advantage becomes more pronounced. On the Jetson AGX Orin platform, compared to Qwen3-8B, MiniCPM4 and MiniCPM4.1 achieves approximately 7x decoding speed improvement.

![benchmark](./assets/minicpm4/efficiency.png)

MiniCPM4.1 achieves 3x decoding speed improvement in reasoning.

![benchmark](./assets/minicpm4/minicpm4.1_speed.png)

#### Comprehensive Evaluation
MiniCPM4 launches end-side versions with 8B and 0.5B parameter scales, both achieving best-in-class performance in their respective categories.

![benchmark](./assets/minicpm4/benchmark.png)

MiniCPM4.1 launches end-side versions with 8B parameter scale, achieving best-in-class performance in deep reasoning mode.

![benchmark](./assets/minicpm4/benchmark4.1.png)

#### Long Text Evaluation
MiniCPM4 is pre-trained on 32K long texts and achieves length extension through YaRN technology. In the 128K long text needle-in-a-haystack task, MiniCPM4 demonstrates outstanding performance. MiniCPM4.1 is pre-trained on 64K long texts and achieves length extension through YaRN technology. In the 128K long text needle-in-a-haystack task, MiniCPM4.1 demonstrates outstanding performance.

![long-niah](./assets/minicpm4/128k-niah.png)

### Inference
MiniCPM 4.1 can be used with following frameworks: Huggingface Transformers, SGLang, vLLM, and CPM.cu. For the ultimate inference speed, we highly recommend CPM.cu.

MiniCPM4/MiniCPM4.1 supports both dense attention inference and sparse attention inference modes, where vLLM and SGLang currently only support dense inference mode. If you want to use sparse inference mode, please use Huggingface Transformers and CPM.cu.

- Dense attention inference: vLLM, SGLang, Huggingface Transformers
- Sparse attention inference: Huggingface Transformers, CPM.cu

#### Hybrid Reasoning Mode

MiniCPM4.1 supports hybrid reasoning mode, which can be used in both deep reasoning mode and non-reasoning mode. To enable hybrid reasoning mode. User can set `enable_thinking=True` in `tokenizer.apply_chat_template` to enable hybrid reasoning mode, and set `enable_thinking=False` to enable non-reasoning mode. Similarly, user can directly add `/no_think` at the end of the query to enable non-reasoning mode. If not add any special token or add `/think` at the end of the query, the model will enable reasoning mode.

```python
# Enable reasoning mode
prompt_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)
# Enable non-reasoning mode
prompt_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)
```

#### HuggingFace

- **Inference with Dense Attention**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
torch.manual_seed(0)

path = 'openbmb/MiniCPM4.1-8B'
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)

# User can directly use the chat interface
# responds, history = model.chat(tokenizer, "Write an article about Artificial Intelligence.", temperature=0.7, top_p=0.7)
# print(responds)

# User can also use the generate interface
messages = [
    {"role": "user", "content": "Write an article about Artificial Intelligence."},
]
prompt_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([prompt_text], return_tensors="pt").to(device)

model_outputs = model.generate(
    **model_inputs,
    max_new_tokens=32768,
    top_p=0.95,
    temperature=0.6
)
output_token_ids = [
    model_outputs[i][len(model_inputs[i]):] for i in range(len(model_inputs['input_ids']))
]

responses = tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0]
print(responses)
```

- **Inference with Sparse Attention**
This model supports InfLLM v2, a sparse attention mechanism designed for efficient long-sequence inference. It requires the [infllmv2_cuda_impl](https://github.com/OpenBMB/infllmv2_cuda_impl) library.

You can install it by running the following command:

```bash
git clone -b feature_infer https://github.com/OpenBMB/infllmv2_cuda_impl.git
cd infllmv2_cuda_impl
git submodule update --init --recursive
pip install -e . # or python setup.py install 
```

To enable InfLLM v2, you need to add the `sparse_config` field in `config.json`:

```json
{
    ...,
    "sparse_config": {
        "kernel_size": 32,
        "kernel_stride": 16,
        "init_blocks": 1,
        "block_size": 64,
        "window_size": 2048,
        "topk": 64,
        "use_nope": false,
        "dense_len": 8192
    }
}
```

These parameters control the behavior of InfLLM v2:

* `kernel_size` (default: 32): The size of semantic kernels.
* `kernel_stride` (default: 16): The stride between adjacent kernels.
* `init_blocks` (default: 1): The number of initial blocks that every query token attends to. This ensures attention to the beginning of the sequence.
* `block_size` (default: 64): The block size for key-value blocks.
* `window_size` (default: 2048): The size of the local sliding window. 
* `topk` (default: 64): The specifies that each token computes attention with only the top-k most relevant key-value blocks.
* `use_nope` (default: false): Whether to use the NOPE technique in block selection for improved performance.
* `dense_len` (default: 8192): Since Sparse Attention offers limited benefits for short sequences, the model can use standard (dense) attention for shorter texts. The model will use dense attention for sequences with a token length below `dense_len` and switch to sparse attention for sequences exceeding this length. Set this to `-1` to always use sparse attention regardless of sequence length.

- **Long Context Extension**
MiniCPM4.1 natively supports context lengths of up to 65,536(64k) tokens. For conversations where the total length (including both input and output) significantly exceeds this limit, we recommend using RoPE scaling techniques for effective handling of long texts. We have validated the model's performance on context lengths of up to 131,072 tokens by modifying the LongRoPE factor.

You can apply the LongRoPE factor modification by modifying the model files. Specifically, in the `config.json` file, adjust the `rope_scaling` fields.

```json
{
    ...,
    "rope_scaling": {
        "rope_type": "longrope", 
        "long_factor": [0.9982316082870437, 1.033048153422584, 1.0749920956484724, 1.1255096879436193, 1.1863348602111476, 1.259543828902579, 1.3476188888731149, 1.4535223827776373, 1.5807816745852985, 1.7335856049489526, 1.9168922912975785, 2.1365471404135326, 2.3994084200118646, 2.713475511863602, 3.0880118452194134, 3.533650295140154, 4.062463396503134, 4.687974098908333, 5.425075306704039, 6.289818967956352, 7.29902962722721, 8.6357018163639, 10.210822723989212, 12.053807765671676, 14.193944598909404, 16.65780676784363, 19.463620727694074, 22.628311203524586, 26.150106147261315, 30.02526691405111, 34.23183327975347, 38.73811934094828, 43.502489489729555, 48.47627117965394, 53.61139491762471, 58.857366522037935, 64.16798299215064, 69.51359464319125, 74.86555458220285, 80.21497790341579, 85.55322183307433, 90.89611806932027, 96.26245306514224, 101.68269304046481, 107.18619510219668, 112.82253283014026, 118.63764063163615, 119.88866203644656, 120.9462882391725, 121.837565139014, 122.58663780572562, 123.2147719894291, 123.74049454862576, 124.17980424685767, 124.54641761955492, 124.85202548028222, 125.10654406389756, 125.31835105170659, 125.49450117164764, 125.64091910903052, 125.76256945356558, 125.86360463815589, 125.94749252260765, 126.01712561287873],
        "short_factor": [0.9982316082870437, 1.033048153422584, 1.0749920956484724, 1.1255096879436193, 1.1863348602111476, 1.259543828902579, 1.3476188888731149, 1.4535223827776373, 1.5807816745852985, 1.7335856049489526, 1.9168922912975785, 2.1365471404135326, 2.3994084200118646, 2.713475511863602, 3.0880118452194134, 3.533650295140154, 4.062463396503134, 4.687974098908333, 5.425075306704039, 6.289818967956352, 7.29902962722721, 8.6357018163639, 10.210822723989212, 12.053807765671676, 14.193944598909404, 16.65780676784363, 19.463620727694074, 22.628311203524586, 26.150106147261315, 30.02526691405111, 34.23183327975347, 38.73811934094828, 43.502489489729555, 48.47627117965394, 53.61139491762471, 58.857366522037935, 64.16798299215064, 69.51359464319125, 74.86555458220285, 80.21497790341579, 85.55322183307433, 90.89611806932027, 96.26245306514224, 101.68269304046481, 107.18619510219668, 112.82253283014026, 118.63764063163615, 119.88866203644656, 120.9462882391725, 121.837565139014, 122.58663780572562, 123.2147719894291, 123.74049454862576, 124.17980424685767, 124.54641761955492, 124.85202548028222, 125.10654406389756, 125.31835105170659, 125.49450117164764, 125.64091910903052, 125.76256945356558, 125.86360463815589, 125.94749252260765, 126.01712561287873],
        "original_max_position_embeddings": 65536
    }
}
```

#### vLLM

##### Speculative Decoding

For accelerated inference with speculative decoding using vLLM, follow these steps:

###### 1. Download MiniCPM4.1 Draft Model

First, download the MiniCPM4.1 draft model:

```bash
cd /your_path
git clone https://huggingface.co/openbmb/MiniCPM4.1-8B-Eagle3
```

###### 2. Install EAGLE3-Compatible vLLM

The EAGLE3 vLLM PR has been submitted. For now, use our repository for installation:

```bash
git clone https://github.com/LDLINGLINGLING/vllm.git
cd vllm 
pip install -e .
```

###### 3. Launch vLLM Server with Speculative Decoding

Start the vLLM inference server with speculative decoding enabled. Make sure to update the model path in the speculative-config to point to your downloaded MiniCPM4_1-8B-Eagle3-bf16 folder:

```bash
VLLM_USE_V1=1 \
vllm serve openbmb/MiniCPM4.1-8B \
--seed 42 \
--trust-remote-code \
--speculative-config '{
  "model": "your/path/MiniCPM4_1-8B-Eagle3-bf16",
  "num_speculative_tokens": 3,
  "method": "eagle3",
  "draft_tensor_parallel_size": 1
}'
```

###### 4. Client Usage Example

The client usage remains the same for both standard and speculative decoding:

```python
import openai

client = openai.Client(base_url="http://localhost:8000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="openbmb/MiniCPM4.1-8B",
    messages=[
        {"role": "user", "content": "Write an article about Artificial Intelligence."},
    ],
    temperature=0.6,
    max_tokens=32768,
    extra_body=dict(add_special_tokens=True),  # Ensures special tokens are added for chat template
    
)

print(response.choices[0].message.content)
```

###### vLLM Configuration Parameters

- `VLLM_USE_V1=1`: Enables vLLM v1 API
- `--speculative-config`: JSON configuration for speculative decoding
  - `model`: Path to the draft model for speculation
  - `num_speculative_tokens`: Number of speculative tokens (default: 3)
  - `method`: Speculative decoding method (eagle3)
  - `draft_tensor_parallel_size`: Tensor parallel size for draft model (default: 1)
- `--seed`: Random seed for reproducibility
- `--trust-remote-code`: Allow execution of remote code for custom models

##### Standard Inference (Without Speculative Decoding)

For now, you need to install the latest version of vLLM.

```bash
pip install -U vllm \
    --pre \
    --extra-index-url https://wheels.vllm.ai/nightly
```

Then you can inference MiniCPM4.1-8B with vLLM:
```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

model_name = "openbmb/MiniCPM4.1-8B"
prompt = [{"role": "user", "content": "Write an article about Artificial Intelligence."}]

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
input_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

llm = LLM(
    model=model_name,
    trust_remote_code=True,
    max_num_batched_tokens=65536,
    dtype="bfloat16", 
    gpu_memory_utilization=0.8, 
)
sampling_params = SamplingParams(top_p=0.95, temperature=0.6, max_tokens=32768)

outputs = llm.generate(prompts=input_text, sampling_params=sampling_params)

print(outputs[0].outputs[0].text)
```

Also, you can start the inference server by running the following command:
> **Note**: In vLLM's chat API, `add_special_tokens` is `False` by default. This means important special tokens—such as the beginning-of-sequence (BOS) token—will not be added automatically. To ensure the input prompt is correctly formatted for the model, you should explicitly set `extra_body={"add_special_tokens": True}`.

```bash
vllm serve openbmb/MiniCPM4.1-8B --trust-remote-code
```

Then you can use the chat interface by running the following code:

```python
import openai

client = openai.Client(base_url="http://localhost:8000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="openbmb/MiniCPM4.1-8B",
    messages=[
        {"role": "user", "content": "Write an article about Artificial Intelligence."},
    ],
    temperature=0.6,
    max_tokens=32768,
    extra_body=dict(add_special_tokens=True),  # Ensures special tokens are added for chat template
)

print(response.choices[0].message.content)
```

#### SGLang

##### Speculative Decoding

For accelerated inference with speculative decoding, follow these steps:

###### 1. Download MiniCPM4.1 Draft Model

First, download the MiniCPM4.1 draft model:

```bash
cd /your_path
git clone https://huggingface.co/openbmb/MiniCPM4.1-8B-Eagle3
```

###### 2. Install EAGLE3-Compatible SGLang

The EAGLE3 adaptation PR has been submitted. For now, use our repository for installation:

```bash
git clone https://github.com/LDLINGLINGLING/sglang.git
cd sglang
pip install -e .
```

###### 3. Launch SGLang Server with Speculative Decoding

Start the SGLang server with speculative decoding enabled:

```bash
python -m sglang.launch_server \
  --model-path "openbmb/MiniCPM4.1-8B" \
  --host "127.0.0.1" \
  --port 30002 \
  --mem-fraction-static 0.9 \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path "your/path/MiniCPM4_1-8B-Eagle3-bf16" \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 32 \
  --temperature 0.7
```

###### 4. Client Usage

The client usage remains the same for both standard and speculative decoding:

```python
import openai

client = openai.Client(base_url=f"http://localhost:30002/v1", api_key="None")

response = client.chat.completions.create(
    model="openbmb/MiniCPM4.1-8B",
    messages=[
        {"role": "user", "content": "Write an article about Artificial Intelligence."},
    ],
    temperature=0.6,
    max_tokens=32768,
)

print(response.choices[0].message.content)
```

Note: Make sure to update the port number in the client code to match the server port (30002 in the speculative decoding example).

###### Configuration Parameters

- `--speculative-algorithm EAGLE3`: Enables EAGLE3 speculative decoding
- `--speculative-draft-model-path`: Path to the draft model for speculation
- `--speculative-num-steps`: Number of speculative steps (default: 3)
- `--speculative-eagle-topk`: Top-k parameter for EAGLE (default: 1)
- `--speculative-num-draft-tokens`: Number of draft tokens (default: 32)
- `--mem-fraction-static`: Memory fraction for static allocation (default: 0.9)

##### Standard Inference (Without Speculative Decoding)

For now, you need to install our forked version of SGLang.

```bash
git clone -b openbmb https://github.com/OpenBMB/sglang.git
cd sglang

pip install --upgrade pip
pip install -e "python[all]"
```

You can start the inference server by running the following command:

```bash
python -m sglang.launch_server --model openbmb/MiniCPM4.1-8B --trust-remote-code --port 30000 --chat-template chatml
```

Then you can use the chat interface by running the following command:

```python
import openai

client = openai.Client(base_url=f"http://localhost:30000/v1", api_key="None")

response = client.chat.completions.create(
    model="openbmb/MiniCPM4.1-8B",
    messages=[
        {"role": "user", "content": "Write an article about Artificial Intelligence."},
    ],
    temperature=0.6,
    max_tokens=32768,
)

print(response.choices[0].message.content)
```


#### CPM.cu

We **recommend** using [CPM.cu](https://github.com/OpenBMB/CPM.cu) for the inference of MiniCPM4 and MiniCPM4.1. CPM.cu is a CUDA inference framework developed by OpenBMB, which integrates efficient sparse, speculative sampling, and quantization techniques, fully leveraging the efficiency advantages of MiniCPM4 and MiniCPM4.1.

You can install CPM.cu by running the following command:

```bash
git clone https://github.com/OpenBMB/CPM.cu.git --recursive
cd CPM.cu
python3 setup.py install
```

You can run the following command to test the speed of the model.

```bash
python3 tests/long_prompt_gen.py # generate prompt.txt
python3 tests/test_generate.py --prompt-file prompt.txt
```

You can run the following command to infer with EAGLE3 speculative decoding algorithm.

```bash
python3 -m cpmcu.cli \
    --model-path $BASE_MODEL_PATH \
    --draft-model-path $EAGLE3_DRAFT_MODEL_PATH \
    --prompt-text "Tell me about Tsinghua University" \
    --use-eagle3 true
```

For more details about CPM.cu, please refer to the repo of [CPM.cu](https://github.com/OpenBMB/CPM.cu).


#### llama.cpp and Ollama

We also support inference with [llama.cpp](https://github.com/ggml-org/llama.cpp) and [Ollama](https://ollama.com/).

##### llama.cpp

You can download the GGUF format of MiniCPM4.1-8B model from [huggingface](https://huggingface.co/openbmb/MiniCPM4.1-8B-GGUF) and run it with llama.cpp for efficient CPU or GPU inference.
```
# case 1: main-cli
./build/bin/llama-cli -m MiniCPM4.1-8B-Q4_K_M.gguf -p "Write an article about Artificial Intelligence." -n 1500

# case 2: server
## launch server
./build/bin/llama-server -m MiniCPM4.1-8B-Q4_K_M.gguf --host 127.0.0.1 --port 8080 -c 4096 -fa on &

## send request
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Write an article about Artificial Intelligence."}],
    "max_tokens": 1500
  }'
```

##### Ollama
Please refer to [model hub](https://ollama.com/openbmb/minicpm4.1) for model download. After installing ollama package, you can use MiniCPM4.1 with following commands:
```
ollama run openbmb/minicpm4.1
```

</details>

> Quantization (**BitCPM4**) and MiniCPM4 applications (**Survey** / **MCP** / **Intel AIPC Client**): see [`docs/README-legacy.md`](./docs/README-legacy.md).


## LICENSE

#### Model LICENSE

* This repository and MiniCPM models are released under the [Apache-2.0](https://github.com/OpenBMB/MiniCPM/blob/main/LICENSE) License. 

#### Statement

* As a language model, MiniCPM generates content by learning from a vast amount of text. 
* However, it does not possess the ability to comprehend or express personal opinions or value judgments. 
* Any content generated by MiniCPM does not represent the viewpoints or positions of the model developers. 
* Therefore, when using content generated by MiniCPM, users should take full responsibility for evaluating and verifying it on their own.

## Institutions

This project is developed by the following institutions:

- <img src="assets/modelbest.png" width="28px"> [Modelbest Inc.](https://modelbest.cn/)
- <img src="assets/thunlp.png" width="28px"> [THUNLP](https://nlp.csai.tsinghua.edu.cn/)
- <img src="assets/RUC.png" width="28px"> [Gaoling School of Artificial Intelligence of RUC](https://linyankai.github.io/)

## Citation

* Please cite our paper: [MiniCPM4](https://arxiv.org/abs/2506.07900) if you find our work valuable.

```
@article{minicpm4,
  title={Minicpm4: Ultra-efficient llms on end devices},
  author={MiniCPM, Team},
  journal={arXiv preprint arXiv:2506.07900},
  year={2025}
}
```
