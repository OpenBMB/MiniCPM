<div align="center">
<h1>
  MiniCPM
</h1>
</div>

<p align="center">
<a href="https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16" target="_blank">Hugging Face</a> |
<a href="https://modelscope.cn/models/OpenBMB/miniCPM-bf16" target="_blank">ModelScope</a> |
<a href="https://wisemodel.cn/models/OpenBMB/miniCPM-bf16" target="_blank">WiseModel</a> |
<a href="XXXX" target="_blank">技术报告</a> 
</p>


<div align="center">

XXXXXX
XXXXXX

Experience models with larger scale at [Luca](https://luca.cn/).

<h4 align="center">
    <p>
        <b>中文</b> |
        <a href="XXXX">English</a>
    <p>
</h4>

</div>

## Quick Links

- [Downloading](#1)
- [Quick Start](#2)
- [Benchmark](#3)
- [Deployment on Mobile Phones](#4)
- [Demo & API](#5)
- [Fine-tuning Models](#6)
- [LICENSE](#7)
- [Citation](#8)
- [Show Cases](#9)

<p id="1"></p>

## Downloading

  | HuggingFace | ModelScope | WiseModel |
  |-------------|------------|-----------|
  |[sft-bf16](https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16)|[sft-bf16](https://modelscope.cn/models/OpenBMB/miniCPM-bf16)|[sft-bf16](https://wisemodel.cn/models/OpenBMB/miniCPM-bf16)
  |[sft-fp32](https://huggingface.co/openbmb/MiniCPM-2B-sft-fp32)|[sft-fp32](https://modelscope.cn/models/OpenBMB/MiniCPM-2B-sft-fp32)|[sft-fp32](https://wisemodel.cn/models/OpenBMB/miniCPM-dpo-fp32)
  |[dpo-bf16](https://huggingface.co/openbmb/MiniCPM-2B-dpo-bf16)|[dpo-bf16](https://modelscope.cn/models/OpenBMB/MiniCPM-2B-dpo-bf16/summary)|[dpo-bf16](https://wisemodel.cn/models/OpenBMB/MiniCPM-2B-dpo-bf16)
  |[dpo-fp16](https://huggingface.co/openbmb/MiniCPM-2B-dpo-fp16)|[dpo-fp16](https://modelscope.cn/models/OpenBMB/MiniCPM-2B-dpo-fp16/)|[dpo-fp16](https://wisemodel.cn/models/OpenBMB/MiniCPM-2B-dpo-fp16)
  |[dpo-fp32](https://huggingface.co/openbmb/MiniCPM-2B-dpo-fp32)|[dpo-fp32](https://modelscope.cn/models/OpenBMB/MiniCPM-2B-dpo-fp32)|[dpo-fp32](https://wisemodel.cn/models/OpenBMB/miniCPM-dpo-fp32)

<p id="2"></p>

## Quick Start

<p id="3"></p>

#### vLLM 

* Install vLLM supporting MiniCPM.
  - MiniCPM adopts the MUP structure, and this structure introduces some extra scaling operations to make the training process stable. And the MUP structure is little different from the structure used by Llama and other LLMs.
  - vLLM 0.2.2 is adapted to MiniCPM in the folder [inference](https://github.com/OpenBMB/MiniCPM/tree/main/inference). More vLLM versions will be supported in the future.
```shell
pip install inference/vllm
```

* Transfer Huggingface Transformers repo to vLLM-MiniCPM repo, where `<hf_repo_path>`, `<vllmcpm_repo_path>` are local paths.
```shell
python inference/convert_hf_to_vllmcpm.py --load <hf_repo_path> --save <vllmcpm_repo_path>
```

* Examples
```shell
cd inference/vllm/examples/infer_cpm
python inference.py --model_path <vllmcpm_repo_path> --prompt_path prompts/prompt_final.txt
```

#### Huggingface 

* Install `transformers>=4.36.0` and `accelerate`，run the following python code.
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
torch.manual_seed(0)

path = 'openbmb/MiniCPM-2B-dpo-bf16'
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map='cuda', trust_remote_code=True)

responds, history = model.chat(tokenizer, "Which city is the capital of China?", temperature=0.8, top_p=0.8)
print(responds)
```

* Examples
```shell
The capital city of China is Beijing. Beijing is not only the political center of China but also a cultural and economic hub. It is known for its rich history and numerous landmarks, such as the Great Wall, the Forbidden City, and the Temple of Heaven. The city is also home to the National Stadium, also known as the "Bird's Nest," and the National Aquatics Center, or "Water Cube." Beijing is a significant city in China, with a population of over 21 million people.
```


# Benchmark 

  | HuggingFace | ModelScope | WiseModel |
  |-------------|------------|-----------|
  |[sft-bf16](https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16)|[sft-bf16](https://modelscope.cn/models/OpenBMB/miniCPM-bf16)|[sft-bf16](https://wisemodel.cn/models/OpenBMB/miniCPM-bf16)
  |[sft-fp32](https://huggingface.co/openbmb/MiniCPM-2B-sft-fp32)|[sft-fp32](https://modelscope.cn/models/OpenBMB/MiniCPM-2B-sft-fp32)|[sft-fp32](https://wisemodel.cn/models/OpenBMB/miniCPM-dpo-fp32)
  |[dpo-bf16](https://huggingface.co/openbmb/MiniCPM-2B-dpo-bf16)|[dpo-bf16](https://modelscope.cn/models/OpenBMB/MiniCPM-2B-dpo-bf16/summary)|[dpo-bf16](https://wisemodel.cn/models/OpenBMB/MiniCPM-2B-dpo-bf16)
  |[dpo-fp16](https://huggingface.co/openbmb/MiniCPM-2B-dpo-fp16)|[dpo-fp16](https://modelscope.cn/models/OpenBMB/MiniCPM-2B-dpo-fp16/)|[dpo-fp16](https://wisemodel.cn/models/OpenBMB/MiniCPM-2B-dpo-fp16)
  |[dpo-fp32](https://huggingface.co/openbmb/MiniCPM-2B-dpo-fp32)|[dpo-fp32](https://modelscope.cn/models/OpenBMB/MiniCPM-2B-dpo-fp32)|[dpo-fp32](https://wisemodel.cn/models/OpenBMB/miniCPM-dpo-fp32)

## Multi-modal

|Models|MME(P)|MMB-dev(en)|MMB-dev(zh)|MMMU-val|CMMMU-val|
|-|-|-|-|-|-|
|LLaVA-Phi|1335.1|59.8|/|/|/|
|MobileVLM|1288.9|59.6|/|/|/|
|Imp-v1|1434.0|66.5|/|/|/|
|Qwen-VL-Chat|**1487**|60.6|56.7|**35.9**|30.7
|**MiniCPM-V**|1446|**67.3**|**61.9**|34.7|**32.1**|

## DPO

|Models|MT-bench|
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
|LLaMA-2-13B-chat|6.65|
|Vicuna-13B|6.57|
|MPT-34B-instruct|6.39|
|LLaMA-2-7B-chat|6.27|
|Vicuna-7B|6.17|
|MPT-7B-chat|5.42|


## Deployment on mobile phones

<!-- 进行Int4量化后，MiniCPM只占2GB空间，具备在端侧手机进行模型部署的条件。
对此，我们针对Android和Harmony系统使用开源框架MLC-LLM进行模型适配，针对iPhone系统使用开源框架LLMFarm进行模型适配，并分别选取了部分端侧手机设备进行了测试。 -->
After INT4 quantization, MiniCPM only occupies 2GB of space, meeting the requirements of inference on edge devices. 

We utilize the open-source framework [MLC-LLM](https://github.com/mlc-ai/mlc-llm) for deployment on Android and Harmony OS. For deployment on IOS, we adapt MiniCPM using [LLMFarm](https://github.com/guinmoon/LLMFarm). We select some mobile phones for testing respectively.

### Tutorial

  #### Android
<!-- android编译安装MiniCPM指南 [EN](https://github.com/OpenBMB/mlc-MiniCPM/blob/main/README.md)  -->
[Compilation and installation on Android](https://github.com/OpenBMB/mlc-MiniCPM/blob/main/README.md)

  #### IOS
<!-- [ios编译安装MiniCPM指南](https://github.com/OpenBMB/LLMFarm) -->
[Compilation and installation on IOS](https://github.com/OpenBMB/LLMFarm)

  #### Multimodal

### Performance

<!-- 我们并为针对手机部署进行深度优化，仅验证MiniCPM使用手机芯片进行推理的可行性。
**我们也欢迎更多开发者进一步调优并更新下面的测试列表，不断提升端侧大模型在手机上的推理性能。** -->
Instead of conducting in-depth optimization for deployment on mobile phones, we only verify the feasibility of MiniCPM using mobile chips for inference.

**We welcome more developers to continuously improve the inference performance of LLMs on mobile phones and update the test results below.**

|Mobile Phones|OS|Processor|Memory（GB）|Inference Throughput（token/s）|
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
|Huawei Nova 11SE|Harmony 4.0.0|snapdragon 778|12|1.9|
|Xiaomi MIX 2|Android 9|snapdragon 835|6|1.3|
|iPhone 15 Pro|iOS 17.2.1|A16|8|18.0|
|iPhone 15|iOS 17.2.1|A16|6|15.0|
|iPhone 12 Pro|iOS 16.5.1|A14|6|5.8|
|iPhone 12|iOS 17.2.1|A14|4|5.8|
|iPhone 11|iOS 16.6|A13|4|4.6|

![多模态样例](https://github.com/OpenBMB/OmniLMM/blob/main/assets/Snake_cn_Mushroom_en.gif)
  
## Demo & API

#### Web-demo based on Gradio

Using the following command can launch the gradio-based demo. 
```shell
python demo/gradio_based_demo.py
```

## Fine-tuning

* Parameter-efficient Tuning
  * With parameter-efficient tuning, we can tune MiniCPM using one piece of NVIDIA GeForce GTX 1080/2080.
  * [Code for Parameter-efficient Tuning](https://github.com/OpenBMB/MiniCPM/tree/main/finetune)
  
* Full-parameter Tuning
  * Using [BMTrain](https://github.com/OpenBMB/BMTrain)，as well as checkpointing and ZeRO-3 (zero redundancy optimizer)，we can tune all parameters of MiniCPM using one piece of NVIDIA GeForce GTX 3090/4090.
  * This code will be available soon.


## LICENSE

#### Model LICENSE

* This repository is released under the [Apache-2.0](https://github.com/OpenBMB/MiniCPM/blob/main/LICENSE) License. 
* The usage of MiniCPM model weights must strictly follow [the General Model License (GML)](https://github.com/OpenBMB/General-Model-License/blob/main/%E9%80%9A%E7%94%A8%E6%A8%A1%E5%9E%8B%E8%AE%B8%E5%8F%AF%E5%8D%8F%E8%AE%AE-%E6%9D%A5%E6%BA%90%E8%AF%B4%E6%98%8E-%E5%AE%A3%E4%BC%A0%E9%99%90%E5%88%B6-%E5%95%86%E4%B8%9A%E6%8E%88%E6%9D%83.md).
* The models and weights of MiniCPM are completely free for academic research.
* If you intend to utilize the model for commercial purposes, please reach out to cpm@modelbest.cn to obtain the certificate of authorization.

#### Statement

* As a language model, MiniCPM generates content by learning from a vast amount of text. 
* However, it does not possess the ability to comprehend or express personal opinions or value judgments. 
* Any content generated by MiniCPM does not represent the viewpoints or positions of the model developers. 
* Therefore, when using content generated by MiniCPM, users should take full responsibility for evaluating and verifying it on their own.


## Citation

* Please cite our [techinical report]() if you find our work valuable.
```
@inproceedings{minicpm2024,
	title={MiniCPM：Unveiling the Potential of End-side Large Language Models},
	booktitle={OpenBMB Blog},
	year={2024}
}
```

## Show Cases

#### Code
Case 1:
![代码生成-case1](./assets/code.case1.gif)

Case 2:
![代码生成-case2](./assets/code.case2.gif)

#### Reasoning
Case 1:
![数理逻辑-case1](./assets/math.case1.png)

Case 2:
![数理逻辑-case1](./assets/math.case2.png)


#### World-Knowledge
Case 1:
![知识推理-case1](./assets/knowledge.case1.png)

#### Content Creation
Case 1:
![内容创作-case1](./assets/creation.case1.png)

#### Translation
Case 1:
![文本翻译-case1](./assets/translation.case1.png)

Case 2:
![文本翻译-case1](./assets/translation.case2.png)

#### Instruction Following
Case 1:
![指令跟随-case1](./assets/instruction_following.case1.png)

Case 2:
![指令跟随-case1](./assets/instruction_following.case2.png)

#### Special characters
Case 1:
![指令跟随-case1](./assets/instruction_following.case3.png)

Case 2:
![指令跟随-case1](./assets/instruction_following.case4.png)
