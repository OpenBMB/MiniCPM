<div align="center">
  <img src="./assets/logo.png" alt="MiniCPM-4-MCP 标志" width="400em"></img>
</div>

<p align="center">
    【中文 | <a href="README_en.md">English</a>】
</p>

<p align="center">
  <a href="#概述">概述</a> •
  <a href="#安装">安装</a> •
  <a href="#模型训练">模型训练</a> •
  <a href="https://huggingface.co/openbmb/MiniCPM4-MCP">模型下载</a> •
  <a href="#推理">推理</a> •
  <a href="#模型评估">模型评估</a>
</p>

## 最新消息

* [2025-06-05] 🚀🚀🚀我们开源了基于MiniCPM4-8B构建的MiniCPM4-MCP，能够调用多种MCP工具，性能比肩更大模型。

## 概述

**MiniCPM-4-MCP** 是由[清华大学自然语言处理实验室（THUNLP）](https://nlp.csai.tsinghua.edu.cn)、中国人民大学与 [ModelBest](https://modelbest.cn/en) 联合开发的开源本地大语言模型智能体，基于 [MiniCPM-4](https://huggingface.co/openbmb/MiniCPM4-8B)，拥有 80 亿参数。它能够通过 MCP 与各种工具和数据资源交互，解决多种真实世界任务。截至目前，MiniCPM4-MCP 已支持：

- 涵盖 16 个 MCP Servers 中工具的使用：这些servers横跨了办公类、生活类、通讯类、资讯类、工作管理类等.

- 单工具使用的能力：可使用符合 MCP 协议的工具进行单一工具的一步或多步调用。

- 跨工具组合使用的能力：可组合使用符合 MCP 协议的不同工具。


演示案例（1 倍速）：



https://github.com/user-attachments/assets/b60b3ed6-f175-4ab7-ace2-4488ef82cb69




## 安装

所需依赖包版本在 `./requirements.txt` 中列出，以确保兼容性。

```
pip install -r requirements.txt
```


## 模型训练

我们主要采用“示范学习”的方法训练模型。示范数据由大语言模型与 MCP 环境持续交互生成。MiniCPM 通过监督微调（SFT）从这些示范中学习。我们使用 LLaMa-Factory 作为 SFT 框架训练 MiniCPM，适配过后的 LLaMa-Factory 版本会进行发布。

### 模型下载

通过[此链接](https://huggingface.co/openbmb/MiniCPM4-MCP)下载模型。

### 数据格式

```json
{
  "conversations": [
    {"from": "human", "value": "你好，我想把500美元换成欧元，你能帮我吗？"}, 
    {"from": "gpt", "value": "<|thought_start|>\n我将调用 get_currency_exchange_rate 函数将 500 美元兑换成欧元。\n<|thought_end|>\n<|tool_call_start|>\n```python\nget_currency_exchange_rate(from_currency=\"USD\",to_currency=\"EUR\",amount=500)\n```\n<|tool_call_end|>\n"}, 
    {"from": "tool", "value": "{\"converted_amount\": 425.50, \"exchange_rate\": 0.851}"}, 
    {"from": "gpt", "value": "<|thought_start|>\n助手认为用户想进行货币兑换，这是可以通过 get_currency_exchange_rate 函数完成的，并已获得执行该函数所需的参数。\n<|thought_end|>\n当然，500 美元大约可兑换为 425.50 欧元。当前汇率为 0.851。"}
  ], 
  "tools": "[{\"name\": \"get_currency_exchange_rate\", \"description\": \"获取两种货币之间的汇率\", \"parameters\": {\"type\": \"object\", \"properties\": {\"from_currency\": {\"type\": \"string\", \"description\": \"原始货币\"}, \"to_currency\": {\"type\": \"string\", \"description\": \"目标货币\"}, \"amount\": {\"type\": \"number\", \"description\": \"兑换金额\"}}, \"required\": [\"from_currency\", \"to_currency\", \"amount\"]}}, {\"name\": \"generate_random_password\", \"description\": \"生成符合特定要求的随机密码\", \"parameters\": {\"type\": \"object\", \"properties\": {\"length\": {\"type\": \"integer\", \"description\": \"密码长度\"}, \"include_numbers\": {\"type\": \"boolean\", \"description\": \"是否包含数字\"}, \"include_symbols\": {\"type\": \"boolean\", \"description\": \"是否包含符号\"}}, \"required\": [\"length\"]}}]", 
  "system": "你是一个具有函数调用能力的智能助手。请在需要时使用这些函数。"
}
```

### 单机训练

在单台机器上执行训练：

```bash
llamafactory-cli train /path/to/config.yaml
```

示例：

```bash
llamafactory-cli train ./LLaMA-Factory/examples/train_full/minicpm4/sft.yaml
```

### 分布式训练

所有节点必须满足以下条件：

- 软件环境一致

- 网络可连接（如开放 TCP 端口 29500）

- 可访问相同的训练数据

步骤如下：

1. 获取主节点IP（节点0）：

```bash
hostname -I | awk '{print $1}'
```

2. 启动训练：


主节点（节点0）：

```bash
export MASTER_ADDR=$(hostname -I | awk '{print $1}')

FORCE_TORCHRUN=1 NNODES=2 RANK=0 MASTER_ADDR=$MASTER_ADDR MASTER_PORT=29500 \
llamafactory-cli train /path/to/config.yaml
```

工作节点（节点1）：

```bash
FORCE_TORCHRUN=1 NNODES=2 RANK=1 MASTER_ADDR=$MASTER_ADDR MASTER_PORT=29500 \
llamafactory-cli train /path/to/config.yaml
```

## 推理

### MCP Servers 部署
MiniCPM4-MCP 所支持的 MCP Servers 具体包含
[Airbnb](https://github.com/openbnb-org/mcp-server-airbnb), 
[Amap-Maps](https://github.com/zxypro1/amap-maps-mcp-server),
[Arxiv-MCP-Server](https://github.com/blazickjp/arxiv-mcp-server),
[Calculator](https://github.com/githejie/mcp-server-calculator),
[Computer-Control-MCP](https://github.com/AB498/computer-control-mcp),
[Desktop-commander](https://github.com/wonderwhy-er/DesktopCommanderMCP),
[Filesystem](https://github.com/mark3labs/mcp-filesystem-server),
[Github](https://github.com/modelcontextprotocol/servers-archived/tree/main/src/github),
[Gaode](https://github.com/perMAIN/gaode),
[MCP-Code-Executor](https://github.com/bazinga012/mcp_code_executor),
[MCP-DOCx](https://github.com/MeterLong/MCP-Doc),
[PPT](https://github.com/GongRzhe/Office-PowerPoint-MCP-Server),
[PPTx](https://github.com/supercurses/powerpoint),
[Simple-Time-Server](https://github.com/andybrandt/mcp-simple-timeserver),
[Slack](https://github.com/modelcontextprotocol/servers-archived/tree/main/src/slack),
[Whisper](https://github.com/arcaputo3/mcp-server-whisper)。
根据这些servers的仓库指引即可成功部署。需要注意的是，这些servers中包含的工具并非全部都可以在环境中顺利跑通，有一些工具的波动性较大，会返回例如timeout、http error等报错。在训练数据构造的过程中，失败率过高的工具（例如LLM在上百次尝试后仍无法为该工具构建出一条能将其成功调用的query）会被过滤掉。

### MCP Client 搭建
我们基于 [mcp-cli](https://github.com/chrishayuk/mcp-cli) 仓库已经实现的 MCP Client 进行修改，由此实现 MiniCPM 和 MCP Server 的交互。MCP Client与 Server 进行握手后所获得的server工具列表内容样例如`available_tool_example.json`所示。获取到available tools以及用户query之后，可按照以下脚本中的逻辑使用指定模型生成结果：

```bash
python generate_example.py \
--tokenizer_path {minicpm4 tokenizer的路径} \
--base_url {vllm部署的url} \
--model {vllm部署时的模型名} \
--output_path {结果保存路径}
```

其中，MiniCPM4 以如下格式生成工具调用：

```
    <|tool_call_start|>
    ```python 
    read_file(path="/path/to/file")
    ```
    <|tool_call_end|>
```

可依据此逻辑为 MiniCPM4 工具调用实现自定义解析器，解析逻辑相关代码位于：`generate_example.py`。

由于 [mcp-cli](https://github.com/chrishayuk/mcp-cli) 仓库支持 vllm 推理框架，因此也可以通过修改vllm从而令MiniCPM4-MCP直接适配mcp-cli的运行逻辑。具体而言，可按照[此链接](https://github.com/OpenBMB/MiniCPM/tree/main/demo/minicpm3/function_call)所述方式修改vllm从而实现搭载着MiniCPM4-MCP模型的client与server的交互通信。


## 模型评估

生成结束后，运行以下脚本进行评估：

```bash
python eval_scripts.py \
--input_path {generate生成结果的保存路径}
```

该脚本用于评估模型在单轮工具调用中函数名预测的表现。多轮调用情况下，给定之前步骤的ground-truth信息即可评测模型在当前步骤生成的工具调用指令的准确性，每个步骤的评测逻辑与单轮相同。

### 评估结果

| MCP 服务器             |          | gpt-4o   |          |          | qwen3    |          |          | minicpm4 |          |
| -------------------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
|                      | 函数名正确率   | 参数名正确率    | 数值正确率    | 函数名正确率   | 参数名正确率    | 数值正确率    | 函数名正确率   | 参数名正确率    | 数值正确率    |
| Airbnb                | 89.3           | 67.9         | 53.6         | 92.8          | 60.7         | 50.0         | 96.4           | 67.9         | 50.0         |
| Amap-Maps             | 79.8           | 77.5         | 50.0         | 74.4          | 72.0         | 41.0         | 89.3           | 85.7         | 39.9         |
| Arxiv-MCP-Server      | 85.7           | 85.7         | 85.7         | 81.8          | 54.5         | 50.0         | 57.1           | 57.1         | 52.4         |
| Calculator            | 100.0          | 100.0        | 20.0         | 80.0          | 80.0         | 13.3         | 100.0          | 100.0        | 6.67         |
| Computor-Control-MCP  | 90.0           | 90.0         | 90.0         | 90.0          | 90.0         | 90.0         | 90.0           | 90.0         | 86.7         |
| Desktop-Commander     | 100.0          | 100.0        | 100.0        | 100.0         | 100.0        | 100.0        | 100.0          | 100.0        | 100.0        |
| Filesystem            | 63.5           | 63.5         | 31.3         | 69.7          | 69.7         | 26.0         | 83.3           | 83.3         | 42.7         |
|Github | 92.0 | 80.0 | 58.0 | 80.5 | 50.0 | 27.7 | 62.8 | 25.7 | 17.1 |
| Gaode                 | 71.1           | 55.6         | 17.8         | 68.8          | 46.6         | 24.4         | 68.9           | 46.7         | 15.6         |
| MCP-Code-Executor     | 85.0           | 80.0         | 70.0         | 80.0          | 80.0         | 70.0         | 90.0           | 90.0         | 65.0         |
| MCP-Docx              | 95.8           | 86.7         | 67.1         | 94.9          | 81.6         | 60.1         | 95.1           | 86.6         | 76.1         |
| PPT                   | 72.6           | 49.8         | 40.9         | 85.9          | 50.7         | 37.5         | 91.2           | 72.1         | 56.7         |
| PPTx                  | 64.2           | 53.7         | 13.4         | 91.0          | 68.6         | 20.9         | 91.0           | 58.2         | 26.9         |
| Simple-Time-Server    | 90.0           | 70.0         | 70.0         | 90.0          | 90.0         | 90.0         | 90.0           | 60.0         | 60.0         |
| Slack                 | 100.0          | 90.0         | 70.0         | 100.0         | 100.0        | 65.0         | 100.0          | 100.0        | 100.0        |
| Whisper               | 90.0           | 90.0         | 90.0         | 90.0          | 90.0         | 90.0         | 90.0           | 90.0         | 30.0         |
| **平均值**              | **80.2**       | **70.2**     | **49.1**     | **83.5**      | **67.7**     | **43.8**     | **88.3**       | **76.1**     | **51.2**     |


