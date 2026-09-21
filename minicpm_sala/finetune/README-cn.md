# MiniCPM-SALA 微调指南

我们通过 [**Hugging Face Transformers Trainer**](https://huggingface.co/docs/transformers/en/main_classes/trainer) 和 [**LLaMA-Factory**](https://github.com/hiyouga/LlamaFactory) 支持灵活的微调工作流。无论您是在单个本地 GPU 上工作，还是在庞大的计算集群上，我们的设置都能满足您的需求：

* **单 GPU 训练：** 针对快速原型设计和较小模型进行了优化。
* **单节点多 GPU：** 使用 DeepSpeed ZeRO、Accelerate 多 GPU 或 FSDP 实现可扩展的性能。
* **多节点训练：** 由 Accelerate FSDP 驱动的大规模分布式训练。

---

## Hugging Face Transformers Trainer

我们提供官方脚本，以便在微调下游任务时轻松调整预训练的 **MiniCPM-SALA**。我们的微调脚本默认使用 `transformers.Trainer` 和 DeepSpeed。我们支持全参数微调和 LoRA 微调。

* **全参数微调**：全参数微调需要在整个训练过程中更新 LLM 的所有参数。
* **LoRA 微调**：LoRA 允许仅更新一小部分参数，实现轻量级的模型调优。我们提供了基于 `peft` 的 LoRA 实现。

### 前置要求

**1. 安装**
通过 pip 安装所有软件包：

```bash
pip install -r requirements.txt
```

**2. 多轮对话格式**
多轮对话微调示例采用了对话格式约定，为不同角色添加不同的 `loss_mask` 值，从而能够在单次前向传播中计算多个回复的损失。

**示例数据集格式：**

```json
[
  {
    "messages": [
      { "role": "system", "content": "<系统提示文本>" },
      { "role": "user", "content": "<用户提示文本>" },
      { "role": "assistant", "content": "<助手回复文本>" },
      { "role": "user", "content": "<用户提示文本>" },
      { "role": "assistant", "content": "<助手回复文本>" }
    ]
  }
]
```

**注意：** 微调代码现在包含验证集。一个完整的数据集必须包含训练集和验证集（测试集是可选的）。以下是单个示例的数据格式：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "类型#裙*裙长#半身裙"
    },
    {
      "role": "assistant",
      "content": "这款百搭时尚的仙女半身裙，整体设计非常的飘逸随性，穿上之后每个女孩子都能瞬间变成小仙女啦。料子非常的轻盈，透气性也很好，穿到夏天也很舒适。"
    }
  ]
}
```

### DeepSpeed ZeRO

**全参数微调**

```bash
formatted_time=$(date +"%Y%m%d%H%M%S")

deepspeed --include localhost:0,1,2,3,4,5,6,7 finetune.py \
    --model_name_or_path openbmb/MiniCPM-SALA \
    --output_dir output/AdvertiseGenSFT/$formatted_time/ \
    --train_data_path data/AdvertiseGenChatML/train.json \
    --eval_data_path data/AdvertiseGenChatML/dev.json \
    --learning_rate 5e-5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 32 \
    --bf16 \
    --gradient_accumulation_steps 4 \
    --warmup_steps 100 \
    --max_steps 3000 \
    --weight_decay 0.01 \
    --eval_steps 100 \
    --save_strategy steps \
    --save_steps 500 \
    --seed 42 \
    --log_level info \
    --logging_strategy steps \
    --logging_steps 10 \
    --deepspeed configs/ds_config_zero3.json
```

**LoRA 微调**
只需在上述脚本中添加 `--use_lora`。

### Accelerate

**全参数微调**

```bash
formatted_time=$(date +"%Y%m%d%H%M%S")

accelerate launch --config_file configs/accelerate/fsdp_config.yaml \
    finetune.py \
    --model_name_or_path openbmb/MiniCPM-SALA \
    --output_dir output/AdvertiseGenSFT/$formatted_time/ \
    --train_data_path data/AdvertiseGenChatML/train.json \
    --eval_data_path data/AdvertiseGenChatML/dev.json \
    --learning_rate 5e-5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 32 \
    --bf16 \
    --gradient_accumulation_steps 4 \
    --warmup_steps 100 \
    --max_steps 3000 \
    --weight_decay 0.01 \
    --eval_steps 100 \
    --save_strategy steps \
    --save_steps 500 \
    --seed 42 \
    --log_level info \
    --logging_strategy steps \
    --logging_steps 10
```

**LoRA 微调**
只需在上述脚本中添加 `--use_lora`。

---

## LLaMA-Factory

我们的 Hugging Face 检查点 `openbmb/MiniCPM-SALA` 可以直接在 LLaMA-Factory 中用于全参数和 LoRA 训练。我们提供 SFT、KTO、DPO 和持续预训练的训练指南。

### 前提条件

**1. 安装**
克隆并安装 LLaMA-Factory GitHub 仓库：

```bash
git clone https://github.com/hiyouga/LlamaFactory.git
cd LLaMA-Factory
pip install -r requirements.txt
```

**2. 数据准备**
由于我们支持四种训练模式（pretrain、sft、dpo、kto），您需要根据 `llama_factory/llama_factory_data/{mode}_demo.json` 中提供的格式准备数据。

然后，将数据集信息添加到 `LLaMA-Factory/data/dataset_info.json` 中，确保可以找到您的数据集。示例：

```json
{
  "identity": {
    "file_name": "identity.json"
  },
  "sft_zh_demo": {
    "file_name": "alpaca_zh_demo.json"
  },
  "kto_en_demo": {
    "file_name": "kto_en_demo.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "kto_tag": "label"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  },
  "dpo_en_demo": {
    "file_name": "dpo_en_demo.json",
    "ranking": true,
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "chosen": "chosen",
      "rejected": "rejected"
    }
  }
}
```

### 创建训练配置 YAML 文件

**全参数微调**
创建一个名为 `minicpm_sala_sft.yaml` 的完整训练配置文件，并将其放在 `LLaMA-Factory/examples/minicpm_config/` 中。对于其他配置，请参考 [`llama_factory/configs`](./llama_factory/configs)。

```yaml
### model
model_name_or_path: openbmb/MiniCPM-SALA
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: full

### ddp
ddp_timeout: 180000000
deepspeed: examples/deepspeed/ds_z3_config.json

### dataset
dataset: sft_zh_demo
template: cpm4
cutoff_len: 1800
max_samples: 500000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/minicpm/minicpm_sala_full
logging_steps: 10
save_strategy: epoch
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 0.0001
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true

### eval
val_size: 0.1
per_device_eval_batch_size: 4
eval_steps: 500
```

**LoRA 微调**
创建一个名为 `minicpm_sala_lora_sft.yaml` 的配置文件，并将其放在 `LLaMA-Factory/examples/minicpm_config/` 中。复制上面的 `minicpm_sala_sft.yaml` 的内容，只需将 `finetuning_type` 从 `full` 更改为 `lora`。

### 模型训练命令

#### **完整训练 (DeepSpeed)**

```bash
llamafactory-cli train examples/minicpm_config/minicpm_sala_sft.yaml
```

#### **完整训练 (Accelerate FSDP 多节点)**

```bash
accelerate launch \
    --config_file examples/accelerate/fsdp_config_multiple_nodes.yaml \
    src/train.py examples/minicpm_config/minicpm_sala_sft.yaml
```

#### **LoRA 训练 (DeepSpeed)**

```bash
llamafactory-cli train examples/minicpm_config/minicpm_sala_lora_sft.yaml
```
