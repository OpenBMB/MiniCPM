# VLLM 推理 MiniCPM | MiniCPM inference on VLLM

### 中文

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

### English


* Install vLLM which supports MiniCPM
 - The structure of MiniCPM is not completely same as Llama, since MiniCPM uses the structure of MUP and scaling is applied in matrix multiplications.
 - We implemented the inference of MiniCPM in vLLM 0.2.2, and the code is located at [inference](https://github.com/OpenBMB/MiniCPM/tree/main/inference). Newer vLLM versions will be supported in the future.

* Install vLLM which supports MiniCPM
```shell
pip install inference/vllm
```

* Convert Huggingface repo to vllm-cpm repo，where `<hf_repo_path>`, `<vllmcpm_repo_path>` are local paths
```shell
python inference/convert_hf_to_vllmcpm.py --load <hf_repo_path> --save <vllmcpm_repo_path>
```

* Test cases
```shell
cd inference/vllm/examples/infer_cpm
python inference.py --model_path <vllmcpm_repo_path> --prompt_path prompts/prompt_demo.txt
```

* Expected Output
```shell
<用户>: Which city is the capital of China?
<AI>:
 The capital city of China is Beijing. Beijing is a major political, cultural, and economic center in China, and it is known for its rich history, beautiful architecture, and vibrant nightlife. It is also home to many of China's most important cultural and historical sites, including the Forbidden City, the Great Wall of China, and the Temple of Heaven. Beijing is a popular destination for tourists from around the world, and it is an important hub for international business and trade.
```