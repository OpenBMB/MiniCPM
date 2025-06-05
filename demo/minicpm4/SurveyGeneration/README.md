<div align="center">
  <img src="./assets/main.png" alt="MiniCPM4-Survey MAIN" width="400em"></img>
</div>
<p align="center">
    【<a href="README-en.md">English</a> | 中文】
</p>

<p align="center">
  <a href="https://huggingface.co/openbmb/MiniCPM4-Survey">模型</a> •
  <a href="https://arxiv.org/abs/????">技术报告</a>
</p>

## News

* [2025-06-05] 🚀🚀🚀我们开源了基于MiniCPM4-8B构建的MiniCPM4-Survey，能够生成可信的长篇调查报告，性能比肩更大模型。

## 概览

MiniCPM4-Survey是由[THUNLP](https://nlp.csai.tsinghua.edu.cn)、中国人民大学和[ModelBest](https://modelbest.cn)联合开发的开源大语言模型智能体。它基于[MiniCPM4](https://github.com/OpenBMB/MiniCPM4) 80亿参数基座模型，接受用户质量作为输入，自主生成可信的长篇综述论文。

主要特性包括：
- 计划-检索-写作生成框架 — 我们提出了一个多智能体生成框架，包含三个核心阶段：计划（定义综述的整体结构）、检索（生成合适的检索关键词）和写作（利用检索到的信息，生成连贯的段落）。
- 高质量数据集构建——我们收集并处理大量人类专家写作的综述论文，构建高质量训练集。同时，我们收集大量研究论文，构建检索数据库。
- 多方面奖励设计 — 我们精心设计了包含结构、内容和引用的奖励，用于评估综述的质量，在强化学习训练阶段作奖励函数。
- 多步强化学习训练策略 — 我们提出了一个上下文管理器，以确保在促进有效推理的同时保留必要的信息，并构建了并行环境，维持强化学习训练高效。

**Demo**:

https://github.com/user-attachments/assets/b47b1567-8af1-4be0-bec4-8f86f2207d60

## 使用

### 下载模型
从 Hugging Face 下载[MiniCPM4-Survey](https://huggingface.co/openbmb/MiniCPM4-Survey)并将其放在model/MiniCPM4-Survey中。
我们建议使用[MiniCPM-Embedding-Light](https://huggingface.co/openbmb/MiniCPM-Embedding-Light)作为表征模型，放在model/MiniCPM-Embedding-Light中。


### 准备环境
从 Kaggle 下载论文数据，然后解压。运行`python dataset_process.py`，处理数据并生成检索数据库。然后运行`python build_index.py`，构建检索数据库。
``` bash
curl -L -o ~/Downloads/arxiv.zip\
   https://www.kaggle.com/api/v1/datasets/download/Cornell-University/arxiv
unzip ~/Downloads/arxiv.zip -d .
mkdir data
python ./src/preprocess/dataset_process.py
mkdir index
python ./src/preprocess/build_index.py
```

### 模型推理
运行以下命令来构建检索环境并开始推理：
``` bash
python ./src/retriever.py
bash ./scripts/run.sh
```
如果您想使用前端运行，可以运行以下命令：
``` bash
python ./src/retriever.py
bash ./scripts/run_with_frontend.sh
cd frontend/minicpm4-survey
npm install
npm run dev
```
然后你可以在浏览器中访问`http://localhost:5173`使用。

## 性能

| Method                                      | Relevance | Coverage | Depth | Novelty | Avg.  | Fact Score |
|---------------------------------------------|-----------|----------|-------|---------|-------|------------|
| Naive RAG (driven by G2FT)                  | 3.25      | 2.95     | 3.35  | 2.60    | 3.04  | 43.68      |
| AutoSurvey (driven by G2FT)                 | 3.10      | 3.25     | 3.15  | **3.15**| 3.16  | 46.56      |
| Webthinker (driven by WTR1-7B)              | 3.30      | 3.00     | 2.75  | 2.50    | 2.89  | --         |
| Webthinker (driven by QwQ-32B)              | 3.40      | 3.30     | 3.30  | 2.50    | 3.13  | --         |
| OpenAI Deep Research (driven by GPT-4o)     | 3.50      |**3.95**  | 3.55  | 3.00    | **3.50**  | --         |
| MiniCPM4-Survey                            | 3.45      | 3.70     | **3.85** | 3.00    | **3.50**  | **68.73**  |
| &nbsp;&nbsp;&nbsp;*w/o* RL                  | **3.55**  | 3.35     | 3.30  | 2.25    | 3.11  | 50.24      |

*GPT-4o对综述生成系统的性能比较。“G2FT”代表Gemini-2.0-Flash-Thinking，“WTR1-7B”代表Webthinker-R1-7B。由于Webthinker不包括引用功能，OpenAI Deep Research在导出结果时不提供引用，因此省略了对它们的FactScore评估。我们的技术报告中包含评测的详细信息。*
