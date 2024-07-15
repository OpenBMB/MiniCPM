## 模型量化
<p id="gptq"></p>

**gptq量化**
1. 首先git获取[minicpm_gptqd代码](https://github.com/LDLINGLINGLING/AutoGPTQ/tree/minicpm_gptq)
2. 进入minicpm_gptqd主目录./AutoGPTQ，命令行输入：
    ```
    pip install e .
    ```
3. 前往[模型下载](#1)下载未量化的MiniCPM仓库下所有文件放至本地同一文件夹下,1b、2b模型均可,训练后模型亦可。
4. 命令行输入以下命令，其中no_quantized_model_path是第3步模型下载路径，save_path是量化模型保存路径，--bits 为量化位数可以选择输入4或者8
    ```
    cd Minicpm/quantize
    python gptq_quantize.py --pretrained_model_dir no_quant_model_path --quantized_model_dir quant_save_path --bits 4
    ```
5. 可以使用./AutoGPTQ/examples/quantization/inference.py进行推理，也可以参考前文使用vllm对量化后的模型，单卡4090下minicpm-1b-int4模型vllm推理在2000token/s左右。

<p id="awq"></p>

**awq量化**
1. 在quantize/awq_quantize.py 文件中修改根据注释修改配置参数：
  ```python
  model_path = '/root/ld/ld_model_pretrained/MiniCPM-1B-sft-bf16' # model_path or model_id
  quant_path = '/root/ld/ld_project/pull_request/MiniCPM/quantize/awq_cpm_1b_4bit' # quant_save_path
  quant_data_path='/root/ld/ld_project/pull_request/MiniCPM/quantize/quantize_data/wikitext'# 写入自带量化数据集，data下的alpaca或者wikitext
  quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" } # "w_bit":4 or 8
  quant_samples=512 # how many samples to use for calibration
  custom_data=[{'question':'你叫什么名字。','answer':'我是openmbmb开源的小钢炮minicpm。'}, # 自定义数据集可用
                 {'question':'你有什么特色。','answer':'我很小，但是我很强。'}]
  ```
2. 在quantize/quantize_data文件下已经提供了alpaca和wiki_text两个数据集作为量化校准集，修改上述quant_data_path为其中一个文件夹的路径
3. 如果需要自定义数据集，修改quantize/awq_quantize.py中的custom_data变量，如：
    ```python
    custom_data=[{'question':'过敏性鼻炎有什么症状？','answer':'过敏性鼻炎可能鼻塞，流鼻涕，头痛等症状反复发作，严重时建议及时就医。'},
                 {'question':'1+1等于多少？','answer':'等于2'}]
    ```
4. 根据选择的数据集，选择以下某一行代码替换 quantize/awq_quantize.py 中第三十八行：
  ```python
    #使用wikitext进行量化
    model.quantize(tokenizer, quant_config=quant_config, calib_data=load_wikitext(quant_data_path=quant_data_path))
    #使用alpaca进行量化
    model.quantize(tokenizer, quant_config=quant_config, calib_data=load_alpaca(quant_data_path=quant_data_path))
    #使用自定义数据集进行量化
    model.quantize(tokenizer, quant_config=quant_config, calib_data=load_cust_data(quant_data_path=quant_data_path))
    
  ```
5. 运行quantize/awq_quantize.py文件,在设置的quan_path目录下可得awq量化后的模型。

<p id="bnb"></p>

**bnb量化**
1. 在quantize/bnb_quantize.py 文件中修改根据注释修改配置参数：
```python
model_path = "/root/ld/ld_model_pretrain/MiniCPM-1B-sft-bf16"  # 模型地址
save_path = "/root/ld/ld_model_pretrain/MiniCPM-1B-sft-bf16_int4"  # 量化模型保存地址
```
2. 更多量化参数可根据注释以及llm.int8()算法进行修改：
```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 是否进行4bit量化
    load_in_8bit=False,  # 是否进行8bit量化
    bnb_4bit_compute_dtype=torch.float16,  # 计算精度设置
    bnb_4bit_quant_storage=torch.uint8,  # 量化权重的储存格式
    bnb_4bit_quant_type="nf4",  # 量化格式，这里用的是正太分布的int4
    bnb_4bit_use_double_quant=True,  # 是否采用双量化，即对zeropoint和scaling参数进行量化
    llm_int8_enable_fp32_cpu_offload=False,  # 是否llm使用int8，cpu上保存的参数使用fp32
    llm_int8_has_fp16_weight=False,  # 是否启用混合精度
    #llm_int8_skip_modules=["out_proj", "kv_proj", "lm_head"],  # 不进行量化的模块
    llm_int8_threshold=6.0,  # llm.int8()算法中的离群值，根据这个值区分是否进行量化
)
```
3. 运行quantize/bnb_quantize.py文件,在设置的save_path目录下可得bnb量化后的模型。
```python
cd MiniCPM/quantize
python bnb_quantize.py
```
<p id="quantize_test"></p>

**量化测试**
1. 命令行进入到 MiniCPM/quantize 目录下
2. 修改quantize_eval.sh文件中awq_path,gptq_path,awq_path,bnb_path,如果不需要测试的类型保持为空字符串，如下示例表示仅测试awq模型：
  ```
    awq_path="/root/ld/ld_project/AutoAWQ/examples/awq_cpm_1b_4bit"
    gptq_path=""
    model_path=""
    bnb_path=""
  ```
3. 在MiniCPM/quantize路径下命令行输入：
  ```
    bash quantize_eval.sh
  ```
4. 窗口将输出该模型的内存占用情况、困惑度。