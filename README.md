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

## 目录

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

## 评测结果


     
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


## 端侧部署

进行Int4量化后，MiniCPM只占2GB空间，具备在端侧手机进行模型部署的条件。
对此，我们针对Android和Harmony系统使用开源框架MLC-LLM进行模型适配，针对iPhone系统使用开源框架LLMFarm进行模型适配，并分别选取了部分端侧手机设备进行了测试。



### 部署步骤

  #### 安卓
android编译安装MiniCPM指南 [EN](https://github.com/OpenBMB/mlc-MiniCPM/blob/main/README.md) [ZH](https://github.com/OpenBMB/mlc-MiniCPM/blob/main/README-ZH.md)

  #### IOS
[ios编译安装MiniCPM指南](https://github.com/OpenBMB/LLMFarm)

  #### 多模态

### 部署性能

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
  
## Demo & API

#### 基于Gradio的网页版Demo
使用如下命令启动基于Gradio的网页版demo：
```shell
python demo/gradio_based_demo.py
```

## 高效参数微调

## 开源协议

#### 模型协议

本仓库中代码依照 [Apache-2.0](https://github.com/OpenBMB/MiniCPM/blob/main/LICENSE) 协议开源，MiniCPM 模型权重的使用则需要遵循 [“通用模型许可协议-来源说明-宣传限制-商业授权”](https://github.com/OpenBMB/General-Model-License/blob/main/%E9%80%9A%E7%94%A8%E6%A8%A1%E5%9E%8B%E8%AE%B8%E5%8F%AF%E5%8D%8F%E8%AE%AE-%E6%9D%A5%E6%BA%90%E8%AF%B4%E6%98%8E-%E5%AE%A3%E4%BC%A0%E9%99%90%E5%88%B6-%E5%95%86%E4%B8%9A%E6%8E%88%E6%9D%83.md)。
MiniCPM 模型权重对学术研究完全开放。如需将模型用于商业用途，请联系cpm@modelbest.cn来获取书面授权，在登记后亦允许免费商业使用。

#### 声明

作为一个语言模型，MiniCPM 通过学习大量的文本来生成内容，但它无法理解、表达个人观点或价值判断，它所输出的任何内容都不代表模型开发者的观点和立场。
因此用户在使用 MiniCPM 生成的内容时，应自行负责对其进行评估和验证。

## 工作引用

如果觉得MiniCPM有助于您的工作，请考虑引用下列[技术报告]()

```
@inproceedings{han2022bminf,
	title={MiniCPM: todo},
	booktitle={OpenBMB Blog},
	year={2024}
}
```

## 典型示例

#### 代码生成
Case 1:
```
	<用户>: 给这段代码逐行加上注释\n#include <stdio.h>\nint rabbit(int n) {\nif (n == 1 || n == 2) {\nreturn 1;\n} else {\nreturn rabbit(n - 1) + rabbit(n - 2);\n}\n}\nint main() {\nint n;\nfor(n=1;n<41;n++){\n    printf(\"第%d个月时有%d对兔子\\n\", n, rabbit(n));\n}\nreturn 0;\n}
	<AI>:
	 以下是包含详细注释的代码：
	
	```c
	// 引入标准输入输出库
	#include <stdio.h>
	
	// 定义一个递归函数，计算第n个月有多少对兔子
	int rabbit(int n) {
	    // 如果n等于1或2，说明是第一个月或第二个月，此时有1对兔子
	    if (n == 1 || n == 2) {
	        return 1;
	    }
	    // 否则，递归调用自身，计算前一个月和上个月兔子数量之和
	    else {
	        return rabbit(n - 1) + rabbit(n - 2);
	    }
	}
	
	// 主函数
	int main() {
	    int n;
	
	    // 从1开始循环到第40个月
	    for(n=1; n<41; n++) {
	        // 输出当前月的编号和对应的兔子对数
	        printf("第%d个月时有%d对兔子\n", n, rabbit(n));
	    }
	
	    return 0;
	}
	```
	
	这段代码首先定义了一个名为`rabbit`的递归函数，用于计算第n个月有多少对兔子。在主函数中，我们使用一个for循环从1遍历到第40个月，并调用`rabbit`函数来计算每个月的兔子对数，并使用`printf`函数输出结果。
```
Case 2:
```
	<用户>: 编写一个Python程序来实现一个MiniCPM模型（Transformer结构, 40层，每层维度为2304, 词表大小为122753）。
	<AI>:
	 Mini-CPM（Mini-Contrastive Pretraining）模型是一种基于Transformer结构的自然语言处理模型。它使用自注意力机制和多层编码器-解码器结构来学习序列到序列的转换任务。在实现Mini-CPM模型时，我们需要遵循以下步骤：
	
	1. 导入所需库
	2. 定义模型结构
	3. 训练模型
	4. 测试模型
	
	以下是实现Mini-CPM模型的Python代码：
	
	```python
	import torch
	import torch.nn as nn
	from torch.autograd import Variable
	
	class MiniCpmModel(nn.Module):
	    def __init__(self, vocab_size, d_model, num_layers, dropout=0.1):
	        super(MiniCpmModel, self).__init__()
	        self.vocab_size = vocab_size
	        self.d_model = d_model
	        self.num_layers = num_layers
	        self.dropout = dropout
	
	        # 自注意力层
	        self.self_attn = nn.SelfAttention(d_model, dropout=dropout)
	        self.attn_layer = nn.TransformerEncoderLayer(d_model, num_layers, dropout=dropout)
	        self.transformer_encoder = nn.TransformerEncoder(self.attn_layer, num_layers)
	
	        # 全连接层
	        self.fc = nn.Linear(d_model * 8, vocab_size)
	
	    def forward(self, x):
	        x = self.self_attn(x)
	        x = x.view(x.size(0), -1)
	        x = self.transformer_encoder(x)
	        x = self.fc(x)
	        return x
	
	def train_mini_cpm(model, device, train_loader, optimizer, epochs):
	    model.train()
	    for epoch in range(epochs):
	        for i, (text, target) in enumerate(train_loader):
	            text = text.to(device)
	            target = target.to(device)
	
	            optimizer.zero_grad()
	            output = model(text)
	            loss = nn.CrossEntropyLoss()(output, target)
	            loss.backward()
	            optimizer.step()
	
	def test_mini_cpm(model, device, test_loader):
	    model.eval()
	    correct = 0
	    total = 0
	    with torch.no_grad():
	        for text, target in test_loader:
	            text = text.to(device)
	            target = target.to(device)
	            output = model(text)
	            _, predicted = torch.max(output, 1)
	            total += target.size(0)
	            correct += (predicted == target).sum().item()
	
	    print(f'Test Accuracy: {100 * correct / total}%')
	
	if __name__ == "__main__":
	    # 参数设置
	    vocab_size = 122753
	    d_model = 2304
	    num_layers = 40
	    dropout = 0.1
	
	    # 初始化模型
	    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	    model = MiniCpmModel(vocab_size, d_model, num_layers, dropout).to(device)
	
	    # 训练模型
	    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
	    epochs = 10
	    train_mini_cpm(model, device, train_data, optimizer, epochs)
	
	    # 测试模型
	    test_mini_cpm(model, device, test_data)
	```
	
	请注意，您需要将`train_data`和`test_data`替换为您的训练和测试数据。同时，您需要准备包含文本和目标标签的数据集。
```

#### 数理逻辑
Case 1:
```
	<用户>: 直接给出以下算式的计算结果15 + 25 + 33 + 22 + 9=
	<AI>:
	 15 + 25 + 33 + 22 + 9 = 104
	
	结果是104。
```
Case 2:
```
	<用户>: 如果以下算式中减号均表示加号，而加号实际上是减号，给出算式的计算结果15 - 25 - 12 + 34=
	<AI>:
	 如果在算式中将减号表示为加号，并且加号实际上表示减号，那么算式将变为：
	15 + 25 + 12 - 34
	
	按照数学运算的优先级（先乘除后加减），我们首先计算括号内的加法：
	15 + 25 = 40
	40 + 12 = 52
	
	然后，我们进行减法操作：
	52 - 34 = 18
	
	所以，原算式的计算结果仍然是 18。
```

#### 知识推理
Case 1:
```
	<用户>: 山东省最高的山是哪座山, 它比黄山高还是矮？差距多少？
	<AI>:
	山东省最高的山是泰山，海拔1545米。黄山位于中国安徽省，海拔1864米。从海拔上来看，黄山比泰山高。两者之间的差距为：
	
	黄山 - 泰山 = 319米
	
	请注意，这个数据可能随着气候变化和测量方法的变化而有所变化。
```
Case 2:

#### 内容创作
Case 1:
```
	<用户>: 晚上睡不着，可以写哪些简短高级的朋友圈文案？
	<AI>:
	1. "夜深人静，思绪万千。愿此刻的宁静，能驱散心中的纷扰。"
	2. "夜幕降临，星光闪烁。在这宁静的时刻，愿我的心灵得到片刻的慰藉。"
	3. "月光如水，洒满窗台。此刻的我，只想静静聆听内心的声音。"
	4. "繁星点点，照亮了黑暗。今晚的星空，让我感受到无尽的希望与力量。"
	5. "夜深人静，独享这宁静的时光。愿所有的烦恼都能随风而逝，留下美好的回忆。"
	6. "夜色渐浓，思念渐深。愿这份思念，能温暖我孤独的心房。"
	7. "月光如水，洒满心间。在这宁静的夜晚，愿我能找到内心的安宁与平静。"
	8. "夜深人静，独自品味生活的酸甜苦辣。愿每一个经历，都能成为成长的养分。"
	9. "月光如诗，诗意如画。在这宁静的夜晚，愿我能感受到生活的美好与温馨。"
	10. "夜深人静，倾听内心的呼唤。愿我能在这宁静的时刻，找到心灵的归宿。"
```
Case 2:
```

```

#### 文本翻译
Case 1:
```
	<用户>: Translate this sentence into Russian: '如何用Python创建一个简单的网页爬虫？'.
	<AI>:
	 'Как создать простый скрипт для собирания страниц в скрипте Python?'
```
Case 2:
```
	<用户>: Translate this sentence into French: "I am a fresh man on Chinese, do you know how this sentence is translated: 如何用Python创建一个简单的网页爬虫？"
	<AI>:
	 Je suis un nouveau-né en chinois, sais-tu comment on crée un simple web scraper en Python ?
	
	Note : Voici la traduction en français de la phrase originale : "How to create a simple web crawler in Python".
```

#### 指令跟随
Case 1:
```
	<用户>: 请帮小明写一份东京旅游的游记，要求每一句话以“哇太好玩了！”开头。
	<AI>:
	 哇太好玩了！第一天，我们抵达了东京的成田国际机场。哇太壮观了！机场内的设施和服务都让我惊叹不已。
	
	哇太兴奋了！我们搭乘高速列车前往东京站，准备开始我们的旅程。哇太刺激了！列车穿行在城市的上空，让我们感受到了日本的现代科技。
	
	哇太兴奋了！我们抵达了东京塔，这座标志性的建筑给我们留下了深刻的印象。哇太壮观了！从塔顶俯瞰整个城市，景色美不胜收。
	
	哇太好玩了！我们在浅草寺参观，感受到了日本传统文化的魅力。哇太神奇了！寺庙内的神像和佛像让我们感受到了宗教的力量。
	
	哇太好玩了！我们去了东京迪士尼乐园，这里充满了欢声笑语和快乐。哇太梦幻了！我们乘坐了各种游乐设施，体验了不同的游戏和表演。
	
	哇太好玩了！我们还去了东京迪士尼海洋，这里有更多的海洋主题景点和表演。哇太精彩了！我们欣赏了精彩的烟花表演，结束了这个美好的夜晚。
	
	哇太好玩了！我们的东京之旅结束了，但留下了许多美好的回忆。我们学到了很多关于日本文化和历史的知识，也体验了日本的美食和娱乐。这次旅行真是太棒了！
```
Case 2:
	```
	<用户>: 给我一个藏头(“雪碧”)的春联, 分别按顺序给出上联（七个字），下联（七个字），横批。
	<AI>:
	 以下是符合要求的藏头春联：
	
	上联：雪消冰释瑞气来
	下联：碧海青天春意浓
	横批：瑞雪兆丰年
```

#### 特殊字符
Case 1:
```
	<用户>: 请给以下句子加上emoji让它更生动: 亲爱的妻子：今天是你的生日，请收下我的蛋糕和玫瑰。
	<AI>:
	 亲爱的妻子：🎉今天是你的生日，💖请收下我的蛋糕🎂和玫瑰花🌹。
	
	（注：emoji需要根据具体语言环境进行替换，以下仅作示例展示）
```
Case 2:
```
	<用户>: 请给我一个json, 每个key是一个emoji，每个value是这个emoji对应的意思，给出10个左右
	<AI>:
	 以下是一个包含10个emoji及其对应意义的JSON对象：
	```json
	{
	  "😊": "表示满意和喜悦",
	  "😡": "表示愤怒或不满",
	  "🤣": "表示搞笑的跌倒动作，代表轻松和幽默",
	  "🚀": "表示兴奋和期待，通常与太空探索相关",
	  "💡": "表示光亮和智慧，代表新想法或创意",
	  "💤": "表示休息和放松，通常用于结束对话",
	  "🌟": "表示成就和优秀，代表成功和激励",
	  "🌱": "表示成长和发展，代表进步和积极变化",
	  "🦋": "表示自然和生命的循环，代表希望和重生",
	  "👥": "表示人与人之间的联系和团结，代表友谊和支持"
	}
	```
```

