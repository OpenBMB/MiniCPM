<div align="center">
  <img src="./assets/logo.png" alt="MiniCPM-4-MCP Logo" width="400em"></img>
</div>

<p align="center">
    [English | <a href="README.md">中文</a>]
</p>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#%EF%B8%8F-training">Training</a> •
  <a href="https://huggingface.co/openbmb/MiniCPM4-MCP">Model</a> •
  <a href="#-inference">Inference</a> •
  <a href="#-evaluation">Evaluation</a>
</p>

## News
* [2025-06-05] 🚀🚀🚀 We have open-sourced MiniCPM4-MCP, built on MiniCPM4-8B, which is capable of calling a variety of MCP tools and delivers performance comparable to larger models.

## 🚩 Overview

**MiniCPM4-MCP** is an open-source on-device LLM agent model jointly developed by [THUNLP](https://nlp.csai.tsinghua.edu.cn), Renmin University of China and [ModelBest](https://modelbest.cn/en), built on [MiniCPM-4](https://huggingface.co/openbmb/MiniCPM4-8B) with 8 billion parameters. It is capable of solving a wide range of real-world tasks by interacting with various tool and data resources through MCP. As of now, MiniCPM4-MCP supports the following:

- Utilization of tools across 16 MCP servers: These servers span various categories, including office, lifestyle, communication, information, and work management.

- Single-tool-calling capability: It can perform single- or multi-step tool calls using a single tool that complies with the MCP.

- Cross-tool-calling capability: It can perform single- or multi-step tool calls using different tools that complies with the MCP.


Demo Case (1 x speed):


https://github.com/user-attachments/assets/f8fe481f-899d-4fe7-8f40-64a31c0bddd4




## 🛠️ Installation


The required package versions are listed in `./requirements.txt` to ensure compatibility.

```
pip install -r requirements.txt
```


## 📽️ Training

We primarily adopt a learning-from-demonstration approach to train our model. The demonstrations are generated through continuous interactions between an LLM and the MCP environment. MiniCPM learns from these demonstrations through Supervised Fine-Tuning (SFT). We employ LLaMa-Factory as our SFT framework, with an adapted version for MiniCPM.

### Model Download

Download the model in this [link](https://huggingface.co/openbmb/MiniCPM4-MCP).

### Data Format

```json
{
  "conversations": [
    {"from": "human", "value": "Hi, I need to convert 500 US dollars to Euros. Can you help me with that?"}, 
    {"from": "gpt", "value": "<|thought_start|>\nI will call the get_currency_exchange_rate function to convert 500 US dollars to Euros.\n<|thought_end|>\n<|tool_call_start|>\n```python\nget_currency_exchange_rate(from_currency=\"USD\",to_currency=\"EUR\",amount=500)\n```\n<|tool_call_end|>\n"}, 
    {"from": "tool", "value": "{\"converted_amount\": 425.50, \"exchange_rate\": 0.851}"}, 
    {"from": "gpt", "value": "<|thought_start|>\nThe assistant thought that the user asked for a currency conversion, which is a task that can be handled by the 'get_currency_exchange_rate' function, and the assistant has received the necessary parameters to execute this function.\n<|thought_end|>\nSure, 500 US dollars will convert to approximately 425.50 Euros. The current exchange rate is 0.851."}
  ], 
  "tools": "[{\"name\": \"get_currency_exchange_rate\", \"description\": \"Get the exchange rate between two currencies\", \"parameters\": {\"type\": \"object\", \"properties\": {\"from_currency\": {\"type\": \"string\", \"description\": \"The currency to convert from\"}, \"to_currency\": {\"type\": \"string\", \"description\": \"The currency to convert to\"}, \"amount\": {\"type\": \"number\", \"description\": \"The amount to convert\"}}, \"required\": [\"from_currency\", \"to_currency\", \"amount\"]}}, {\"name\": \"generate_random_password\", \"description\": \"Generate a random password with specified requirements\", \"parameters\": {\"type\": \"object\", \"properties\": {\"length\": {\"type\": \"integer\", \"description\": \"The length of the password\"}, \"include_numbers\": {\"type\": \"boolean\", \"description\": \"Include numbers in the password\"}, \"include_symbols\": {\"type\": \"boolean\", \"description\": \"Include symbols in the password\"}}, \"required\": [\"length\"]}}]", 
  "system": "You are a helpful assistant with access to some functions. Use them if required."}
```

### Single-Node Training
To run training on a single machine, simply use the following command:
```bash
llamafactory-cli train /path/to/config.yaml
```
Example:
```bash
llamafactory-cli train ./LLaMA-Factory/examples/train_full/minicpm4/sft.yaml
```


### Distributed Training Setup (2 Nodes Example)

All nodes must have:
  - Identical software environments
  - Network connectivity (e.g., TCP port 29500 open)
  - Access to the same training data

To run training across 2 machines, follow these steps:

1. Determine Master Node IP
On your master node (node 0), run:
```bash
hostname -I | awk '{print $1}'
```

2. Launch Training

On master node (node 0):
```
export MASTER_ADDR=$(hostname -I | awk '{print $1}')

FORCE_TORCHRUN=1 NNODES=2 RANK=0 MASTER_ADDR=$MASTER_ADDR MASTER_PORT=29500 \
llamafactory-cli train /path/to/config.yaml
```

On worker node (node 1):
```bash
FORCE_TORCHRUN=1 NNODES=2 RANK=1 MASTER_ADDR=$MASTER_ADDR MASTER_PORT=29500 \
llamafactory-cli train /path/to/config.yaml
```


## 📖 Inference

### MCP Servers Deployment

The MCP Servers supported by MiniCPM4-MCP include
[Airbnb](https://github.com/openbnb-org/mcp-server-airbnb), 
[Amap-Maps](https://github.com/zxypro1/amap-maps-mcp-server),
[Arxiv-MCP-Server](https://github.com/blazickjp/arxiv-mcp-server),
[Calculator](https://github.com/githejie/mcp-server-calculator),
[Computer-Control-MCP](https://github.com/AB498/computer-control-mcp),
[Desktop-commander](https://github.com/wonderwhy-er/DesktopCommanderMCP),
[Filesystem](https://github.com/mark3labs/mcp-filesystem-server),
[Github](https://github.com/modelcontextprotocol/servers/tree/main/src/github),
[Gaode](https://github.com/perMAIN/gaode),
[MCP-Code-Executor](https://github.com/bazinga012/mcp_code_executor),
[MCP-DOCx](https://github.com/MeterLong/MCP-Doc),
[PPT](https://github.com/GongRzhe/Office-PowerPoint-MCP-Server),
[PPTx](https://github.com/supercurses/powerpoint),
[Simple-Time-Server](https://github.com/andybrandt/mcp-simple-timeserver),
[Slack](https://github.com/modelcontextprotocol/servers/tree/main/src/slack), and
[Whisper](https://github.com/arcaputo3/mcp-server-whisper). Follow the instructions provided in each server's repository for successful deployment. Note that not all tools in these servers will function properly in every environment. Some tools are unstable and may return errors such as timeouts or HTTP errors. During training data construction, tools with consistently high failure rates (e.g., those for which the LLM fails to produce a successful query even after hundreds of attempts) are filtered out.

### MCP Client Setup

We modified the existing MCP Client from the [mcp-cli](https://github.com/chrishayuk/mcp-cli) repository to enable interaction between MiniCPM and MCP Servers.  
After the MCP Client performs a handshake with a Server, it retrieves a list of available tools. An example of tool information contained in this list is provided in `available_tool_example.json`.  

Once the available tools and user query are obtained, results can be generated using the following script logic:

```bash
python generate_example.py \
--tokenizer_path {path to MiniCPM4 tokenizer} \
--base_url {vllm deployment URL} \
--model {model name used in vllm deployment} \
--output_path {path to save results}
```
where MiniCPM4 generates tool calls in the following format:

```
    <|tool_call_start|>
    ```python 
    read_file(path="/path/to/file")
    ```
    <|tool_call_end|>
```
You can build a custom parser for MiniCPM4 tool calls based on this format. The relevant parsing logic is located in `generate_example.py`.

Since the [mcp-cli](https://github.com/chrishayuk/mcp-cli) repository supports the vLLM inference framework, MiniCPM4-MCP can also be integrated into `mcp-cli` by modifying vLLM accordingly.  
Specifically, follow the instructions in [this link](https://github.com/OpenBMB/MiniCPM/tree/main/demo/minicpm3/function_call) to enable interaction between a client running the MiniCPM4-MCP model and the MCP Server.


## 📈 Evaluation
Once generation is complete, run the following example evaluation script:
```bash
python eval_scripts.py \
--input_path {path where the results generated by `generate` are saved}
```
This script is used to evaluate the model's performance in predicting function names during single-turn tool calls. In multi-turn scenarios, the accuracy of the tool call generated at the current step can be evaluated by providing the ground-truth information from previous steps. The evaluation logic for each step is the same as that of the single-turn setting.



### Evaluation Results

| MCP Server                  |          | gpt-4o             |              |          | qwen3             |              |      |      minicpm4         |              |
|-----------------------|----------------|--------------|--------------|---------------|--------------|--------------|----------------|--------------|--------------|
|                       | func           | param        | value        | func          | param        | value        | func           | param        | value        |
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
| **Average**              | **80.2**       | **70.2**     | **49.1**     | **83.5**      | **67.7**     | **43.8**     | **88.3**       | **76.1**     | **51.2**     |

