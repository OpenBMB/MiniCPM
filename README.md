<div align="center">
<h1>
  MiniCPM
</h1>
</div>

<p align="center">
<a href="XXXX" target="_blank">Hugging Face</a> |
<a href="XXXX" target="_blank">ModelScope</a> |
<a href="XXXX" target="_blank">Hugging Face</a> |
<a href="XXXX" target="_blank">技术报告</a> 
</p>

<div align="center">

XXXXXX
XXXXXX

在[面壁露卡](https://luca.cn/)体验更大规模的模型。

<h4 align="center">
    <p>
        <b>中文</b> |
        <a href="XXXX">English</a>
    <p>
</h4>

</div>

# 目录

- [模型介绍]()
- [模型下载]()
- [评测结果]()
    - [中文]()
    - [英文]()
    - [代码]()
    - [逻辑]()
    - [多模态]()
- [手机部署]()
- [Demo & API]()
- [高效参数微调]()
- [开源协议]()
- [工作引用]()
- [典型示例]()

# 模型介绍

# 模型下载

  [HuggingFace仓库]()
  [ModelScope仓库]()
  [XX仓库]()

# 评测结果


     
## 多模态

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


# 端侧部署

进行Int4量化后，MiniCPM只占2GB空间，具备在端侧手机进行模型部署的条件。
对此，我们针对Android和Harmony系统使用开源框架MLC-LLM进行模型适配，针对iPhone系统使用开源框架LLMFarm进行模型适配，并分别选取了部分端侧手机设备进行了测试。



## 部署步骤

  ### 安卓
android编译安装MiniCPM指南 [EN](https://github.com/OpenBMB/mlc-MiniCPM/blob/main/README.md) [ZH](https://github.com/OpenBMB/mlc-MiniCPM/blob/main/README-ZH.md)

  ### IOS
[ios编译安装MiniCPM指南](https://github.com/OpenBMB/LLMFarm)


## 部署性能

我们并为针对手机部署进行深度优化，仅验证MiniCPM使用手机芯片进行推理的可行性。
**我们也欢迎更多开发者进一步调优并更新下面的测试列表，不断提升端侧大模型在手机上的推理性能。**

|手机型号|操作系统|处理器|Memory（GB）|推理吞吐（token/s）|
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
  
# Demo & API

## 基于Gradio的网页版Demo
使用如下命令启动基于Gradio的网页版demo：
```shell
python demo/gradio_based_demo.py
```

# 高效参数微调

# 开源协议

## 模型协议

本仓库中代码依照 [Apache-2.0](https://github.com/OpenBMB/MiniCPM/blob/main/LICENSE) 协议开源，MiniCPM 模型权重的使用则需要遵循 [“通用模型许可协议-来源说明-宣传限制-商业授权”](https://github.com/OpenBMB/General-Model-License/blob/main/%E9%80%9A%E7%94%A8%E6%A8%A1%E5%9E%8B%E8%AE%B8%E5%8F%AF%E5%8D%8F%E8%AE%AE-%E6%9D%A5%E6%BA%90%E8%AF%B4%E6%98%8E-%E5%AE%A3%E4%BC%A0%E9%99%90%E5%88%B6-%E5%95%86%E4%B8%9A%E6%8E%88%E6%9D%83.md)。
MiniCPM 模型权重对学术研究完全开放。如需将模型用于商业用途，请联系cpm@modelbest.cn来获取书面授权，在登记后亦允许免费商业使用。

## 声明

作为一个语言模型，MiniCPM 通过学习大量的文本来生成内容，但它无法理解、表达个人观点或价值判断，它所输出的任何内容都不代表模型开发者的观点和立场。
因此用户在使用 MiniCPM 生成的内容时，应自行负责对其进行评估和验证。

# 工作引用

如果觉得MiniCPM有助于您的工作，请考虑引用下列[技术报告]()

```
@inproceedings{han2022bminf,
	title={MiniCPM: todo},
	booktitle={OpenBMB Blog},
	year={2024}
}
```

# 典型示例

#### 代码生成
Case 1:
![代码生成-case1](./assets/code.case1.gif)

Case 2:
![代码生成-case2](./assets/code.case2.gif)

#### 数理逻辑
Case 1:
![数理逻辑-case1](./assets/math.case1.png)

Case 2:
![数理逻辑-case1](./assets/math.case2.png)


#### 知识推理
Case 1:
![知识推理-case1](./assets/knowledge.case1.png)

#### 内容创作
Case 1:
![内容创作-case1](./assets/creation.case1.png)

#### 文本翻译
Case 1:
![文本翻译-case1](./assets/translation.case1.png)

Case 2:
![文本翻译-case1](./assets/translation.case2.png)

#### 指令跟随
Case 1:
![指令跟随-case1](./assets/instruction_following.case1.png)

Case 2:
![指令跟随-case1](./assets/instruction_following.case2.png)

#### 特殊字符
Case 1:
![指令跟随-case1](./assets/instruction_following.case3.png)

Case 2:
![指令跟随-case1](./assets/instruction_following.case4.png)
