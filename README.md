<div align="center">
<h1>
  MiniCPM: 揭示端侧大语言模型的无限潜力
</h1>
</div>

<h4 align="center">
    <p>
        <b>中文</b> | <a href="https://github.com/OpenBMB/MiniCPM/blob/main/README-en.md">English</a>
    <p>
</h4>


<p align="center">
<a href="https://shengdinghu.notion.site/MiniCPM-c805a17c5c8046398914e47f0542095a" target="_blank">MiniCPM 技术博客</a> |
<a href="https://github.com/OpenBMB/OmniLMM/" target="_blank">OmniLMM 多模态模型</a> |
<a href="https://luca.cn/" target="_blank">CPM-C 千亿模型试用</a> |
加入我们的 <a href="https://discord.gg/3cGQn9b3YM" target="_blank">discord</a> 和 <a href="https://github.com/OpenBMB/MiniCPM/blob/main/assets/wechat.jpg" target="_blank">wechat</a>
 
</p>

MiniCPM 是面壁智能与清华大学自然语言处理实验室共同开源的系列端侧大模型，主体语言模型 MiniCPM-2B 仅有 24亿（2.4B）的非词嵌入参数量, 总计2.7B参数量。
- 经过 SFT 后，MiniCPM 在公开综合性评测集上，MiniCPM 与 Mistral-7B相近（中文、数学、代码能力更优），整体性能超越 Llama2-13B、MPT-30B、Falcon-40B 等模型。
- 经过 DPO 后，MiniCPM 在当前最接近用户体感的评测集 MTBench上，MiniCPM-2B 也超越了 Llama2-70B-Chat、Vicuna-33B、Mistral-7B-Instruct-v0.1、Zephyr-7B-alpha 等众多代表性开源大模型。
- 以 MiniCPM-2B 为基础构建端侧多模态大模型 MiniCPM-V，整体性能在同规模模型中实现最佳，超越基于 Phi-2 构建的现有多模态大模型，在部分评测集上达到与 9.6B Qwen-VL-Chat 相当甚至更好的性能。
- 经过 Int4 量化后，MiniCPM 可在手机上进行部署推理，流式输出速度略高于人类说话速度。MiniCPM-V 也直接跑通了多模态大模型在手机上的部署。
- 一张1080/2080可高效参数微调，一张3090/4090可全参数微调，一台机器可持续训练 MiniCPM，二次开发成本较低。

我们完全开源MiniCPM-2B的模型参数供学术研究和有限商用.
具体而言，我们目前已公开以下模型，地址详见 [模型下载](#1) 部分
- 基于MiniCPM-2B的指令微调与人类偏好对**MiniCPM-2B-SFT/DPO**。
- 基于MiniCPM-2B的多模态模型**MiniCPM-V**，能力超越基于Phi-2的同参数级别多模态模型。
- MiniCPM-2B-SFT/DPO的Int4量化版**MiniCPM-2B-SFT/DPO-Int4**。
- 基于MLC-LLM、LLMFarm开发的MiniCPM手机端程序，**文本及多模态模型均可在手机端进行推理**。
- 训练过程中的[30个Checkpoints](https://huggingface.co/openbmb/MiniCPM-2B-history)供模型机理研究。

### 局限性：

- 受限于模型规模，模型可能出现**幻觉性问题**。其中由于DPO模型生成的回复内容更长，更容易出现幻觉。我们也将持续进行MiniCPM模型的迭代改进。
- 为了保证在学术研究用途上模型的通用性，我们**未对模型进行任何身份认同训练**。同时由于我们用ShareGPT开源语料作为部分训练数据，模型可能会输出类似GPT系列模型的身份认同信息。
- 受限于模型规模，模型的**输出受到提示词（prompt）的影响较大**，可能多次尝试产生不一致的结果。
- 受限于模型容量，模型的**知识记忆较不准确**，后续我们将结合RAG方法来增强模型的知识记忆能力。

## 目录

- [更新日志](#0)
- [模型下载](#1)
- [快速上手](#2)
- [开源社区](#community)
- [评测结果](#3)
- [手机部署](#4)
- [Demo & API 部署](#5)
- [二次开发](#6)
- [开源协议](#7)
- [工作引用](#8)
- [典型示例](#9)

<p id="0"></p>

## 更新日志
- 2024/03/16 minicpm-2b 的30余个中间检查点开放了！[huggingface链接](https://huggingface.co/openbmb/MiniCPM-2B-history)
- 2024/02/13 支持了llama.cpp
- 2024/02/09 我们在readme里加入了一个[开源社区](#community)章节，用来收集开源社区对MiniCPM的支持案例。
- 2024/02/08 我们更新了[llama-format的模型权重](#llamaformat)，方便大家更加快捷地使用我们的模型。
- 2024/02/01 初始发布。

<p id="1"></p>

## 模型下载

* 语言模型
 
  | HuggingFace | ModelScope | WiseModel | Replicate |
  |-------------|------------|-----------|-----------|
  |[MiniCPM-2B-sft-bf16](https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16)|[MiniCPM-2B-sft-bf16](https://modelscope.cn/models/OpenBMB/miniCPM-bf16)|[MiniCPM-2B-sft-bf16](https://wisemodel.cn/models/OpenBMB/miniCPM-bf16)|
  |[MiniCPM-2B-sft-fp32](https://huggingface.co/openbmb/MiniCPM-2B-sft-fp32)|[MiniCPM-2B-sft-fp32](https://modelscope.cn/models/OpenBMB/MiniCPM-2B-sft-fp32)|[MiniCPM-2B-sft-fp32](https://wisemodel.cn/models/OpenBMB/miniCPM-dpo-fp32)|
  |[MiniCPM-2B-dpo-bf16](https://huggingface.co/openbmb/MiniCPM-2B-dpo-bf16)|[MiniCPM-2B-dpo-bf16](https://modelscope.cn/models/OpenBMB/MiniCPM-2B-dpo-bf16/summary)|[MiniCPM-2B-dpo-bf16](https://wisemodel.cn/models/OpenBMB/MiniCPM-2B-dpo-bf16)|[MiniCPM-2B-dpo-bf16](https://replicate.com/tuantuanzhang/minicpm)
  |[MiniCPM-2B-dpo-fp16](https://huggingface.co/openbmb/MiniCPM-2B-dpo-fp16)|[MiniCPM-2B-dpo-fp16](https://modelscope.cn/models/OpenBMB/MiniCPM-2B-dpo-fp16/)|[MiniCPM-2B-dpo-fp16](https://wisemodel.cn/models/OpenBMB/MiniCPM-2B-dpo-fp16)|
  |[MiniCPM-2B-dpo-fp32](https://huggingface.co/openbmb/MiniCPM-2B-dpo-fp32)|[MiniCPM-2B-dpo-fp32](https://modelscope.cn/models/OpenBMB/MiniCPM-2B-dpo-fp32)|[MiniCPM-2B-dpo-fp32](https://wisemodel.cn/models/OpenBMB/miniCPM-dpo-fp32)|
  |[MiniCPM-2B-sft-fp32-llama-format](https://huggingface.co/openbmb/MiniCPM-2B-sft-fp32-llama-format)|
  |[MiniCPM-2B-sft-bf16-llama-format](https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16-llama-format)|
  |[MiniCPM-2B-dpo-bf16-llama-format](https://huggingface.co/openbmb/MiniCPM-2B-dpo-bf16-llama-format)|
  |[MiniCPM-2B-dpo-fp16-gguf](https://huggingface.co/runfuture/MiniCPM-2B-dpo-fp16-gguf) |
  |[MiniCPM-2B-dpo-q4km-gguf](https://huggingface.co/runfuture/MiniCPM-2B-dpo-q4km-gguf) |

  注: 
  1. 模型训练为bf16训练，因此用bf16进行推理将取得最好的效果，其他的格式会由于精度问题造成一点的性能下降。
  2. -llama-format后缀的模型是我们将MiniCPM结构的模型转化成了Llama结构的（主要将mup的参数化方案融合进了模型本身的参数）。使得Llama模型的使用者可以零成本尝试MiniCPM。[详见这里](#llamaformat)
  3. 感谢[@runfuture](https://github.com/runfuture)对MiniCPM进行了[llama.cpp](https://github.com/ggerganov/llama.cpp)和[ollama](https://github.com/ollama/ollama)的适配


* 多模态模型
  | HuggingFace | ModelScope | WiseModel |
  |-------------|------------|-----------|
  | [MiniCPM-V](https://huggingface.co/openbmb/MiniCPM-V) | [MiniCPM-V](https://modelscope.cn/models/OpenBMB/MiniCPM-V/) | [MiniCPM-V](https://wisemodel.cn/models/OpenBMB/MiniCPM-V) |
  | [OmniLMM](https://huggingface.co/openbmb/OmniLMM-12B) | [OmniLMM](https://modelscope.cn/models/OpenBMB/OmniLMM-12B) | [OmniLMM](https://wisemodel.cn/models/OpenBMB/OmniLMM-12B) |

  


<p id="2"></p>

## 快速上手

#### 在线体验

- [Colab](https://colab.research.google.com/drive/1tJcfPyWGWA5HezO7GKLeyeIso0HyOc0l?usp=sharing)

#### Huggingface 模型

##### MiniCPM-2B
* 安装`transformers>=4.36.0`以及`accelerate`后，运行以下代码
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
torch.manual_seed(0)

path = 'openbmb/MiniCPM-2B-dpo-bf16'
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map='cuda', trust_remote_code=True)

responds, history = model.chat(tokenizer, "山东省最高的山是哪座山, 它比黄山高还是矮？差距多少？", temperature=0.5, top_p=0.8, repetition_penalty=1.02)
print(responds)
```

* 期望输出
```shell
山东省最高的山是泰山，海拔1545米。

相对于黄山（海拔1864米），泰山海拔较低，相差约319米。
```

<p id="llamaformat"></p>

##### MiniCPM-2B （Llama Format）
我们将MiniCPM的模型权重转化成了Llama代码可以直接调用的形式，以便大家尝试:
```python
import torch
from transformers import LlamaTokenizerFast, LlamaForCausalLM
model_path = "openbmb/MiniCPM-2B-dpo-bf16-llama-format"
tokenizer = LlamaTokenizerFast.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='cuda', trust_remote_code=True)

prompt="Now you act like a terminal situated within a beginner's C++ practice repository folder, please provide the output for the command: `ls -l`"
input_ids = tokenizer.encode("<用户>{}<AI>".format(prompt), return_tensors='pt', add_special_tokens=True).cuda()
responds = model.generate(input_ids, temperature=0.3, top_p=0.8, repetition_penalty=1.02, max_length=1024)
responds = tokenizer.decode(responds[0], skip_special_tokens=True)
print(responds)
```

##### MiniCPM-V

```python
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('openbmb/MiniCPM-V', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V', trust_remote_code=True)
model.eval().cuda()

image = Image.open('xx.jpg').convert('RGB')
question = 'What is in the image?'
msgs = [{'role': 'user', 'content': question}]

res, context, _ = model.chat(
    image=image,
    msgs=msgs,
    context=None,
    tokenizer=tokenizer,
    sampling=True,
    temperature=0.7
)
print(res)
```


#### vLLM 推理

* 安装支持 MiniCPM 的 vLLM
  - 因为 MiniCPM 采用 MUP 结构，在矩阵乘法中存在一定的放缩计算，与Llama类模型结构有细微差别。
  - 我们基于版本为 0.2.2 的 vLLM 实现了 MiniCPM 的推理，代码位于仓库[inference](https://github.com/OpenBMB/MiniCPM/tree/main/inference)文件夹下，未来将会支持更新的vLLM 版本。

* 安装支持 MiniCPM 的 vLLM 版本
```shell
pip install inference/vllm
```

* 将Huggingface Transformers仓库转为vLLM-MiniCPM支持的格式，其中`<hf_repo_path>`, `<vllmcpm_repo_path>`均为本地路径
```shell
python inference/convert_hf_to_vllmcpm.py --load <hf_repo_path> --save <vllmcpm_repo_path>
```

* 测试样例
```shell
cd inference/vllm/examples/infer_cpm
python inference.py --model_path <vllmcpm_repo_path> --prompt_path prompts/prompt_demo.txt
```

* 期望输出
```shell
<用户>: Which city is the capital of China?
<AI>:
 The capital city of China is Beijing. Beijing is a major political, cultural, and economic center in China, and it is known for its rich history, beautiful architecture, and vibrant nightlife. It is also home to many of China's most important cultural and historical sites, including the Forbidden City, the Great Wall of China, and the Temple of Heaven. Beijing is a popular destination for tourists from around the world, and it is an important hub for international business and trade.
```

#### llama.cpp、Ollama、fastllm推理
我们支持了[llama.cpp](https://github.com/ggerganov/llama.cpp/) 推理、[ollama](https://github.com/ollama/ollama)推理、[fastllm](https://github.com/ztxz16/fastllm)推理.

**llama.cpp**
1. [安装llama.cpp](https://github.com/ggerganov/llama.cpp?tab=readme-ov-file#build)
2. 下载gguf形式的模型。[下载链接-fp16格式](https://huggingface.co/runfuture/MiniCPM-2B-dpo-fp16-gguf) [下载链接-q4km格式](https://huggingface.co/runfuture/MiniCPM-2B-dpo-q4km-gguf)
3. 在命令行运行示例代码:
```
./main -m ../../model_ckpts/download_from_hf/MiniCPM-2B-dpo-fp16-gguf.gguf --prompt "<用户>写藏头诗，藏头是龙年大吉<AI>" --temp 0.3 --top-p 0.8 --repeat-penalty 1.05
```
更多参数调整[详见](https://github.com/ggerganov/llama.cpp/blob/master/examples/main/README.md)

**ollama**
正在解决[这个问题](https://github.com/ollama/ollama/issues/2383)

**fastllm**
1. [编译安装fastllm](https://github.com/ztxz16/fastllm)
2. 模型推理
```
import torch
from transformers import AutoTokenizer, LlamaTokenizerFast, AutoModelForCausalLM
path = 'openbmb/MiniCPM-2B-dpo-fp16'
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map='cuda', trust_remote_code=True)
from fastllm_pytools import llm
llm.set_device_map("cpu")
model = llm.from_hf(model, tokenizer, dtype = "float16") # dtype支持 "float16", "int8", "int4"
print(model.response("<用户>山东省最高的山是哪座山, 它比黄山高还是矮？差距多少？<AI>", top_p=0.8, temperature=0.5, repeat_penalty=1.02))
```
 


<p id="community"></p>

## 开源社区

- [ChatLLM框架](https://github.com/foldl/chatllm.cpp):[在CPU上跑MiniCPM](https://huggingface.co/openbmb/MiniCPM-2B-dpo-bf16/discussions/2#65c59c4f27b8c11e43fc8796)



<p id="3"></p>

## 评测结果

#### 评测设置

* 由于大模型评测难以统一，且大量评测也没有公开的prompt和测试代码，对于具体评测方式，我们只能尽量做到适合各类模型。
* 整体而言，我们测试时采用统一的prompt输入，并按照各模型对应的模板进行输入调整。
* **评测脚本及prompt已开源在我们的Github仓库中，也欢迎更多开发者来不断改进我们的评测方式。**
  * 文本评测部分，采用了我们的开源大模型能力评测框架[UltraEval](https://github.com/OpenBMB/UltraEval)。以下为开源模型复现流程：
    * 安装UltraEval
      ```shell
      git clone https://github.com/OpenBMB/UltraEval.git
      cd UltraEval
      pip install -e .
      ```
    * 下载相关数据并解压处理
      ```shell
      wget -O RawData.zip "https://cloud.tsinghua.edu.cn/f/71b5232264ae4833a4d0/?dl=1"
      unzip RawData.zip
      python data_process.py
      ```
    * 执行评测脚本(提供了模板，可自定义)
      ```shell
      bash run_eval.sh
      ```

#### 部署模式

* 因为MiniCPM采用Mup的结构，与现有模型在具体计算上有细微差别，我们是基于vllm=0.2.2版本进行了我们模型的实现。
* **对于非MiniCPM模型，我们采用了vllm=0.2.7的最新版本进行推理。**

#### 评测度量

* 对于QA任务（选择题任务），我们选用两种方式进行测试：
  * PPL：将选项作为题目生成的延续，并根据各个选项的PPL来进行答案选择；
  * 第二种是直接生成答案选项。
* 对于不同模型，这两种方式得到的结果差异较大。MiniCPM两种模式上的结果较为接近，而Mistral-7B-v0.1等模型在PPL上表现较好，直接生成上效果较差。
* 在具体评测时，我们以两种评测方式得分的最高者为最终结果，以此保证对比的公平性(以下表格中*号表示采用PPL)。

#### 文本模型评测

**越级比较:**
|模型|平均分|英文均分|中文均分|C-Eval|CMMLU|MMLU|HumanEval|MBPP|GSM8K|MATH|BBH|ARC-E|ARC-C|HellaSwag|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|Llama2-7B|35.40|36.21|31.765|32.42|31.11|44.32|12.2|27.17|13.57|1.8|33.23|75.25|42.75|75.62*|
|Qwen-7B|49.46|47.19|59.655|58.96|60.35|57.65|17.07|42.15|41.24|5.34|37.75|83.42|64.76|75.32*|
|Deepseek-7B|39.96|39.15|43.64|42.82|44.45|47.82|20.12|41.45|15.85|1.53|33.38|74.58*|42.15*|75.45*|
|Mistral-7B|48.97|49.96|44.54|46.12|42.96|62.69|27.44|45.2|33.13|5.0|41.06|83.92|70.73|80.43*|
|Llama2-13B|41.48|42.44|37.19|37.32|37.06|54.71|17.07|32.55|21.15|2.25|37.92|78.87*|58.19|79.23*|
|MPT-30B|38.17|39.82|30.72|29.34|32.09|46.56|21.95|35.36|10.31|1.56|38.22|78.66*|46.08*|79.72*|
|Falcon-40B|43.62|44.21|40.93|40.29|41.57|53.53|24.39|36.53|22.44|1.92|36.24|81.94*|57.68|83.26*|
|MiniCPM-2B|52.33|52.6|51.1|51.13|51.07|53.46|50.00|47.31|53.83|10.24|36.87|85.44|68.00|68.25|

**同级比较：**
|模型|平均分|英文均分|中文均分|C-Eval|CMMLU|MMLU|HumanEval|MBPP|GSM8K|MATH|BBH|ARC-E|ARC-C|HellaSwag|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|TinyLlama-1.1B|25.36|25.55|24.525|25.02|24.03|24.3|6.71|19.91|2.27|0.74|28.78|60.77*|28.15*|58.33*|Qwen-1.8B|34.72|31.87|47.565|49.81|45.32|43.37|7.93|17.8|19.26|2.42|29.07|63.97*|43.69|59.28*|
|Qwen-1.8B|34.72|31.87|47.57|49.81|45.32|43.37|7.93|17.80|19.26|2.42|29.07|63.97*|43.69|59.28*|
|Gemini Nano-3B|-|-|-|-|-|-|-|27.2(report)|22.8(report)|-|42.4(report)|-|-|-|
|StableLM-Zephyr-3B|43.46|46.31|30.62|30.34|30.89|45.9|35.37|31.85|52.54|12.49|37.68|73.78|55.38|71.87*|
|Phi-2-2B|48.84|54.41|23.78|23.37|24.18|52.66|47.56|55.04|57.16|3.5|43.39|86.11|71.25|73.07*|
|MiniCPM-2B|52.33|52.6|51.10|51.13|51.07|53.46|50.00|47.31|53.83|10.24|36.87|85.44|68.00|68.25|

**Chat模型比较：**
|模型|平均分|英文均分|中文均分|C-Eval|CMMLU|MMLU|HumanEval|MBPP|GSM8K|MATH|BBH|ARC-E|ARC-C|HellaSwag|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|ChatGLM2-6B|37.98|35.17|50.63|52.05|49.21|45.77|10.37|9.38|22.74|5.96|32.6|74.45|56.82|58.48*|
|Mistral-7B-Instruct-v0.1|44.36|45.89|37.51|38.06|36.96|53.56|29.27|39.34|28.73|3.48|39.52|81.61|63.99|73.47*|
|Mistral-7B-Instruct-v0.2|50.91|52.83|42.235|42.55|41.92|60.51|36.59|48.95|40.49|4.95|39.81|86.28|73.38|84.55*|
|Qwen-7B-Chat|44.93|42.05|57.9|58.57|57.23|56.03|15.85|40.52|42.23|8.3|37.34|64.44*|39.25*|74.52*|
|Yi-6B-Chat|50.46|45.89|70.995|70.88|71.11|62.95|14.02|28.34|36.54|3.88|37.43|84.89|70.39|74.6*|
|Baichuan2-7B-Chat|44.68|42.74|53.39|53.28|53.5|53|21.34|32.32|25.25|6.32|37.46|79.63|60.15|69.23*|
|Deepseek-7B-chat|49.34|49.56|48.335|46.95|49.72|51.67|40.85|48.48|48.52|4.26|35.7|76.85|63.05|76.68*|
|Llama2-7B-Chat|38.16|39.17|33.59|34.54|32.64|47.64|14.02|27.4|21.15|2.08|35.54|74.28|54.78|75.65*|
|MiniCPM-2B|52.33|52.6|51.10|51.13|51.07|53.46|50.00|47.31|53.83|10.24|36.87|85.44|68.00|68.25|

**DPO后模型比较：**

|模型|MT-bench|
|---|---|
|GPT-4-turbo|9.32|
|GPT-3.5-turbo|8.39|
|Mistral-8*7b-Instruct-v0.1|8.30|
|Claude-2.1|8.18|
|Zephyr-7B-beta|7.34|
|**MiniCPM-2B**|**7.25**|
|Vicuna-33B|7.12|
|Zephyr-7B-alpha|6.88|
|LLaMA-2-70B-chat|6.86|
|Mistral-7B-Instruct-v0.1|6.84|
|MPT-34B-instruct|6.39|

#### 多模态模型评测

<div align="left">

<table style="margin: 0px auto;">
<thead>
  <tr>
    <th align="left">Model</th>
    <th>Size</th>
    <th nowrap="nowrap" >Visual Tokens</th>
    <th>MME</th>
    <th nowrap="nowrap" >MMB dev (en)</th>
    <th nowrap="nowrap" >MMB dev (zh)</th>
    <th nowrap="nowrap" >MMMU val</th>
    <th nowrap="nowrap" >CMMMU val</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td align="left">LLaVA-Phi</td>
    <td align="right">3B</td>
    <td>576</td>
    <td>1335</td>
    <td>59.8</td>
    <td>- </td>
    <td>- </td>
    <td>- </td>
  </tr>
  <tr>
    <td nowrap="nowrap" align="left">MobileVLM</td>
    <td align="right">3B</td>
    <td>144</td>
    <td>1289</td>
    <td>59.6</td>
    <td>- </td>
    <td>- </td>
    <td>- </td>
  </tr>
  <tr>
    <td nowrap="nowrap" align="left" >Imp-v1</td>
    <td align="right">3B</td>
    <td>576</td>
    <td>1434</td>
    <td>66.5</td>
    <td>- </td>
    <td>- </td>
    <td>- </td>
  </tr>
  <tr>
    <td  nowrap="nowrap" align="left" >Qwen-VL-Chat</td>
    <td align="right" >9.6B</td>
    <td>256</td>
    <td>1487</td>
    <td>60.6 </td>
    <td>56.7 </td>
    <td>35.9 </td>
    <td>30.7 </td>
  </tr>
  <tr>
    <td nowrap="nowrap" align="left" >CogVLM</td>
    <td align="right">17.4B </td>
    <td>1225</td>
    <td>1438 </td>
    <td>63.7 </td>
    <td>53.8 </td>
    <td>32.1 </td>
    <td>- </td>
  </tr>
  <tr>
    <td nowrap="nowrap" align="left" ><b>MiniCPM-V(3B)</b></td>
    <td align="right">3B </td>
    <td>64</td>
    <td>1452 </td>
    <td>67.3 </td>
    <td>61.9 </td>
    <td>34.7 </td>
    <td>32.1 </td>
  </tr>
</tbody>
</table>

</div>



<p id="4"></p>

## 手机部署

#### 部署步骤

* 进行Int4量化后，MiniCPM只占2GB空间，具备在端侧手机进行模型部署的条件。
* 对于不同的操作系统，我们进行了不同的适配。
* **注意：当前开源框架对手机支持还在完善，并非所有芯片与操作系统版本均能成功运行MLC-LLM或LLMFarm。**
* Android、HarmonyOS
  * 使用开源框架MLC-LLM进行模型适配。
  * 支持文本模型、多模态模型。
  * 适用于MiniCPM-2B-SFT-INT4、MiniCPM-2B-DPO-INT4、MiniCPM-V。
  * [编译安装MiniCPM指南](https://github.com/OpenBMB/mlc-MiniCPM) 
* iOS
  * 使用开源框架LLMFarm进行模型适配。
  * 支持文本模型。
  * 适用于MiniCPM-2B-SFT-INT4、MiniCPM-2B-DPO-INT4。
  * [编译安装MiniCPM指南](https://github.com/OpenBMB/LLMFarm)

#### 部署性能

* 我们未针对手机推理模型进行深度优化和系统测试，仅验证MiniCPM使用手机芯片进行推理的可行性。
* 【更正】在本工作之前已有初步的基于llama.cpp进行手机部署多模态大模型的[努力](https://github.com/ggerganov/llama.cpp/blob/master/examples/llava/MobileVLM-README.md)，我们此次在MLC-LLM上验证了手机部署MiniCPM-V的可行性，能够正常输入输出，但也存在图片处理时间较长的问题，需要进一步优化，兼容性问题也需要进一步解决 :)。
* **我们也欢迎更多开发者进一步调优并更新下面的测试列表，不断提升端侧大模型在手机上的推理性能**。

|手机型号|操作系统|处理器|Memory（GB）|文本吞吐（token/s）|
|-|-|-|-|-|
|OPPO Find N3|Android 13|snapdragon 8 Gen2|12|6.5|
|Samsung S23 Ultra|Android 14|snapdragon 8 Gen2|12|6.4|
|Meizu M182Q|Android 11|snapdragon 888Plus|8|3.7|
|Xiaomi 12 Pro|Android 13|snapdragon 8 Gen1|8+3|3.7|
|Xiaomi Redmi K40|Android 11|snapdragon 870|8|3.5|
|Oneplus LE 2100|Android 13|snapdragon 870|12|3.5|
|Oneplus HD1900|Android 11|snapdragon 865|8|3.2|
|Oneplus HD1900|Android 11|snapdragon 855|8|3.0|
|Oneplus HD1905|Android 10|snapdragon 855|8|3.0|
|Oneplus HD1900|Android 11|snapdragon 855|8|3.0|
|Xiaomi MI 8|Android 9|snapdragon 845|6|2.3|
|Huawei Nova 11SE|HarmonyOS 4.0.0|snapdragon 778|12|1.9|
|Xiaomi MIX 2|Android 9|snapdragon 835|6|1.3|
|iPhone 15 Pro|iOS 17.2.1|A17 pro|8|18.0|
|iPhone 15|iOS 17.2.1|A16|6|15.0|
|iPhone 12 Pro|iOS 16.5.1|A14|6|5.8|
|iPhone 12|iOS 17.2.1|A14|4|5.8|
|iPhone 11|iOS 16.6|A13|4|4.6|
|Xiaomi Redmi K50|HyperOS 1.0.2|MediaTek Dimensity 8100|12|3.5

![多模态样例](https://github.com/OpenBMB/OmniLMM/blob/main/assets/Snake_cn_Mushroom_en.gif)


<p id="5"></p>

## Demo & API 部署

#### 基于Gradio的网页版Demo

* 使用如下命令启动基于Gradio的网页版demo：

```shell
# generation powered by vllm
python demo/vllm_based_demo.py --model_path <vllmcpm_repo_path>
# generation powered by huggingface
python demo/hf_based_demo.py --model_path <hf_repo_path>
```


<p id="6"></p>

## 二次开发

* 高效参数微调
  * 一张1080/2080可实现高效参数微调
  * [高效参数微调代码](https://github.com/OpenBMB/MiniCPM/tree/main/finetune)
  
* 全参数微调 or 持续训练
  * 使用[BMTrain](https://github.com/OpenBMB/BMTrain)，借助重计算和ZeRO-3，一张3090/4090可实现全参数微调，一台机器可实现持续训练
  * 相关代码也将陆续推出



<p id="9"></p>

## 典型示例

#### 文本生成

![内容创作-case1](./assets/creation.case1.png)

![内容创作-case2](./assets/creation.case2.png)

![内容创作-case3](./assets/creation.case3.png)

#### 代码生成

![代码生成-case1](./assets/code.case1.gif)

![代码生成-case2](./assets/code.case2.gif)

#### 数理逻辑

![数理逻辑-case1](./assets/math.case1.png)

![数理逻辑-case1](./assets/math.case2.png)

#### 文本翻译

![文本翻译-case1](./assets/translation.case1.png)

![文本翻译-case2](./assets/translation.case2.png)

#### 指令跟随

![指令跟随-case1](./assets/instruction_following.case1.png)

![指令跟随-case1](./assets/instruction_following.case2.png)

#### 特殊字符

![特殊字符-case1](./assets/special_char.case1.png)

![特殊字符-case2](./assets/special_char.case2.png)


<p id="7"></p>

## 开源协议

#### 模型协议

* 本仓库中代码依照 [Apache-2.0](https://github.com/OpenBMB/MiniCPM/blob/main/LICENSE) 协议开源
* MiniCPM 模型权重的使用则需要遵循 [“通用模型许可协议-来源说明-宣传限制-商业授权”](https://github.com/OpenBMB/General-Model-License/blob/main/%E9%80%9A%E7%94%A8%E6%A8%A1%E5%9E%8B%E8%AE%B8%E5%8F%AF%E5%8D%8F%E8%AE%AE-%E6%9D%A5%E6%BA%90%E8%AF%B4%E6%98%8E-%E5%AE%A3%E4%BC%A0%E9%99%90%E5%88%B6-%E5%95%86%E4%B8%9A%E6%8E%88%E6%9D%83.md)。
* MiniCPM 模型权重对学术研究完全开放。
* 如需将模型用于商业用途，请联系cpm@modelbest.cn来获取书面授权，在登记后亦允许免费商业使用。

#### 声明

* 作为一个语言模型，MiniCPM 通过学习大量的文本来生成内容，但它无法理解、表达个人观点或价值判断，它所输出的任何内容都不代表模型开发者的观点和立场。
* 因此用户在使用 MiniCPM 生成的内容时，应自行负责对其进行评估和验证。
* 如果由于使用 MiniCPM 开源模型而导致的任何问题，包括但不限于数据安全问题、公共舆论风险，或模型被误导、滥用、传播或不当利用所带来的任何风险和问题，我们将不承担任何责任。

<p id="8"></p>

## 工作引用

* 如果觉得MiniCPM有助于您的工作，请引用我们的[技术报告](https://shengdinghu.notion.site/MiniCPM-c805a17c5c8046398914e47f0542095a?pvs=4)

```
@misc{minicpm2024,
	title={MiniCPM: Unveiling the Potential of End-side Large Language Models},
	booktitle={OpenBMB Blog},
	year={2024}
}
```
