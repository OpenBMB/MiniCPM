# MiniCPM 微调

（部分代码函数的文档由[RepoAgent](https://github.com/OpenBMB/RepoAgent)自动产生)

[English Version](https://github.com/OpenBMB/MiniCPM/blob/main/finetune/README_en.md)

本目录提供 MiniCPM-2B 模型的微调示例，包括全量微调和 PEFT。格式上，提供多轮对话微调样例和输入输出格式微调样例。

如果将模型下载到了本地，本文和代码中的 `OpenBMB/MiniCPM-2B` 字段均应替换为相应地址以从本地加载模型。

运行示例需要 `python>=3.10`，除基础的 `torch` 依赖外，示例代码运行还需要依赖。

**我们提供了 [示例notebook](lora_finetune.ipynb) 用于演示如何以 AdvertiseGen 为例处理数据和使用微调脚本。**

```bash
pip install -r requirements.txt
```

## 测试硬件标准

我们仅提供了单机多卡/多机多卡的运行示例，因此您需要至少一台具有多个 GPU 的机器。本仓库中的**默认配置文件**中，我们记录了显存的占用情况：

+ SFT 全量微调: 4张显卡平均分配，每张显卡占用 `30245MiB` 显存。
+ LORA 微调: 1张显卡，占用 `10619MiB` 显存。

> 请注意，该结果仅供参考，对于不同的参数，显存占用可能会有所不同。请结合你的硬件情况进行调整。

## 多轮对话格式

多轮对话微调示例采用 ChatGLM3 对话格式约定，对不同角色添加不同 `loss_mask` 从而在一遍计算中为多轮回复计算 `loss`。

对于数据文件，样例采用如下格式

```json
[
  {
    "messages": [
      {
        "role": "system",
        "content": "<system prompt text>"
      },
      {
        "role": "user",
        "content": "<user prompt text>"
      },
      {
        "role": "assistant",
        "content": "<assistant response text>"
      },
      // ... Muti Turn
      {
        "role": "user",
        "content": "<user prompt text>"
      },
      {
        "role": "assistant",
        "content": "<assistant response text>"
      }
    ]
  }
  // ...
]
```

## 数据集格式示例

> 请注意，现在的微调代码中加入了验证集，因此，对于一组完整的微调数据集，必须包含训练数据集和验证数据集，测试数据集可以不填写。或者直接用验证数据集代替。

```
{"messages": [{"role": "user", "content": "类型#裙*裙长#半身裙"}, {"role": "assistant", "content": "这款百搭时尚的仙女半身裙，整体设计非常的飘逸随性，穿上之后每个女孩子都能瞬间变成小仙女啦。料子非常的轻盈，透气性也很好，穿到夏天也很舒适。"}]}
```

## 开始微调

通过以下代码执行 **单机多卡/多机多卡** 运行。

```bash
cd finetune
bash sft_finetune.sh
```

通过以下代码执行 **单机单卡** 运行。

```angular2html
cd finetune
bash lora_finetune.sh
```
