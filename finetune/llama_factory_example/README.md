# MiniCPM_llama_factory 微调
MiniCPM已经支持llama_factory微调，llama_factory支持continue_pretrain,sft,ppo,dpo,kto,orpo等等微调方式。
由于llama_factory功能强大，但初学者较难上手，我们录制了微调教程

**我们提供了 llama_factory_example文件夹，用来微调minicpm1b，minicpm2b模型。**
1.首先安装llama_factory依赖。
```bash
git clone https://github.com/hiyouga/LLaMA-Factory
cd LLaMA-Factory
pip install -r requirements.txt
```
2.将数据集处理成Minicpm/finetune/llama_factory_example/llama_factory_data文件夹中的格式,示例包括dpo,kto,sft三种微调方式并放置到llama_factory/data目录下.以dpo为例：
```json
  [
    {
      "conversations": [
        {
          "from": "human",
          "value": "Hi! I'd like to create a new language game simulating the first person perspective of a character named Angela."
        }
      ],
      "chosen": {
        "from": "gpt",
        "value": "That sounds like a fun and engaging idea! Here are some tips to help you create the game:\n1. ......"
      },
      "rejected": {
        "from": "gpt",
        "value": "Hello! I'd be happy to help you create a language game simulating the first-person perspective ....."
      }
    }
  ]
```
3.在llama_factory/data/dataset_info.json中添加数据集信息,保证dataset_info.json中能找到你的数据集，如下例：
``` json
  {"identity": {
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
4.将MiniCPM/finetune/llama_factory_example中文件复制到LLaMA-Factory/examples目录下。
  ```bash
    cd LLaMA-Factory/examples
    mkdir minicpm
    #以下代码中的/your/path要改成你的MiniCPM代码和LLaMA-Factory路径
    cp -r /your/path/MiniCPM/finetune/llama_factory_example/*  /your/path/LLaMA-Factory/examples/minicpm
  ```
5.以dpo为例，首先修改minicpm_dpo.yaml,需要修改的：
```yaml
  model_name_or_path: openbmb/MiniCPM-2B-sft-bf16 #或者你本地保存的地址
  dataset: dpo_en_demo #这里写dataset_info.json中的键名
  output_dir: your/finetune_minicpm/save/path
  bf16: true #如果你的设备支持bf16，否则false
  deepspeed: examples/deepspeed/ds_z2_config.json #如果显存不够可以改成ds_z3_config.json
```
6.修改single_node.sh文件中：

  - 1.如果是a100以及更高端服务器，删除以下两行
  ```bash
    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1 
  ```
  - 2.设置你希望参与微调的卡，以下示例为第1张到第8张卡都参与微调
  ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  ```
  - 3.将以下代码src/train.py空格后方参数改为llama_facoty中minicpm_dpo.yaml的绝对路径
  ```bash
    src/train.py /root/ld/ld_project/LLaMA-Factory/examples/minicpm/minicpm_sft.yaml
  ```
7.执行：
```bash
  cd LLaMA-Factory
  bash single_node.sh
```
