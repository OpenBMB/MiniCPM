# Fine-Tuning Guide for MiniCPM-SALA

We support a flexible range of fine-tuning workflows via [**Hugging Face Transformers Trainer**](https://huggingface.co/docs/transformers/en/main_classes/trainer) and [**LLaMA-Factory**](https://github.com/hiyouga/LlamaFactory). Whether you are working on a single local GPU or a massive compute cluster, our setup scales to meet your needs:

* **Single-GPU Training:** Optimized for rapid prototyping and smaller models.
* **Single-Node Multi-GPU:** Scalable performance using DeepSpeed ZeRO, Accelerate Multi-GPU, or FSDP.
* **Multi-Node Training:** Large-scale distributed training powered by Accelerate FSDP.

---

## Hugging Face Transformers Trainer

We offer official scripts for easy fine-tuning of the pretrained **MiniCPM-SALA** on downstream tasks. Our finetune scripts use `transformers.Trainer` and DeepSpeed by default. We support full-parameter fine-tuning and LoRA fine-tuning.

* **Full-parameter fine-tuning:** Requires updating all parameters of the LLM throughout the entire training process.
* **LoRA fine-tuning:** Allows lightweight model tuning with only a small subset of parameters updated. We provide the LoRA implementation based on `peft`.

### Prerequisites

**1. Installation**
Install all packages via pip:

```bash
pip install -r requirements.txt
```

**2. Multi-Turn Dialogue Format**
The multi-turn dialogue fine-tuning example adopts a dialogue format convention, adding different `loss_mask` values for different roles, thus enabling loss computation for multiple replies in a single forward pass.

**Example Dataset Format:**

```json
[
  {
    "messages": [
      { "role": "system", "content": "<system prompt text>" },
      { "role": "user", "content": "<user prompt text>" },
      { "role": "assistant", "content": "<assistant response text>" },
      { "role": "user", "content": "<user prompt text>" },
      { "role": "assistant", "content": "<assistant response text>" }
    ]
  }
]

```

**Note:** The fine-tuning code now includes a validation set. A complete dataset must contain training and validation sets (test set is optional). Below is the data format for a single example:

```json
{
  "messages": [
    {
      "role": "user", 
      "content": "Category#Skirt*Skirt Length#Half-length skirt"
    }, 
    {
      "role": "assistant", 
      "content": "This versatile and fashionable fairy skirt features an overall design that is incredibly flowy and free-spirited; every girl can instantly transform into a little fairy when wearing it. The fabric is very lightweight with excellent breathability, making it comfortable to wear even in the summer."
    }
  ]
}

```

### DeepSpeed ZeRO

**Full-Parameter Fine-Tuning**

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

**LoRA Fine-Tuning**
simply add `--use_lora` to the script above.

### Accelerate

**Full-Parameter Fine-Tuning**

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

**LoRA Fine-Tuning**
Simply add `--use_lora` to the script above.

---

## LLaMA-Factory

Our Hugging Face checkpoint `openbmb/MiniCPM-SALA` can be directly used in LLaMA-Factory for both full-parameter and LoRA training. We provide training guidelines for SFT, KTO, DPO, and continual pre-training.

### Prerequisites

**1. Installation**
Clone and install the LLaMA-Factory GitHub repository:

```bash
git clone https://github.com/hiyouga/LlamaFactory.git
cd LLaMA-Factory
pip install -r requirements.txt

```

**2. Data Preparation**
Since we support four training modes (pretrain, sft, dpo, kto), you need to prepare the data according to the format provided in `./data_example/{mode}_demo.json`.

Then, add the dataset information to `LLaMA-Factory/data/dataset_info.json`, ensuring that your dataset can be found. Example:

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

### Create Training Configuration YAML Files

**Full-Parameter Fine-Tuning**
Create a full training configuration file named `minicpm_sala_sft.yaml` and place it in `LLaMA-Factory/examples/minicpm_config/`. For other configurations, refer to [`llama_factory/configs`](./llama_factory/configs).

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

**LoRA Fine-Tuning**
Create a configuration file named `minicpm_sala_lora_sft.yaml` and place it in `LLaMA-Factory/examples/minicpm_config/`. Copy the content of `minicpm_sala_sft.yaml` above, and simply change `finetuning_type` from `full` to `lora`.

### Model Training Commands

#### **Full Training (DeepSpeed)**

```bash
llamafactory-cli train examples/minicpm_config/minicpm_sala_sft.yaml
```

#### **Full Training (Accelerate FSDP on Multi-Node)**

```bash
accelerate launch \
    --config_file examples/accelerate/fsdp_config_multiple_nodes.yaml \
    src/train.py examples/minicpm_config/minicpm_sala_sft.yaml
```

#### **LoRA Training (DeepSpeed)**
```bash
llamafactory-cli train examples/minicpm_config/minicpm_sala_lora_sft.yaml
```
