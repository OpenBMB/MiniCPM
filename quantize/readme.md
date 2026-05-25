## 模型量化

> 本目录保留早期 MiniCPM 1B / 2B 系列的历史量化脚本。当前 MiniCPM5-1B 发布包不再通过这里的脚本生成量化权重；端侧部署请优先使用 [`docs/deployment/llama_cpp.md`](../docs/deployment/llama_cpp.md) 和 [`docs/deployment/mlx.md`](../docs/deployment/mlx.md) 中的 GGUF / MLX 路径。

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
2. 修改 `quantize_eval.sh` 文件中的模型路径；如果不需要测试的类型保持为空字符串。
  ```
    model_path=""
    bnb_path=""
  ```
3. 在MiniCPM/quantize路径下命令行输入：
  ```
    bash quantize_eval.sh
  ```
4. 窗口将输出该模型的内存占用情况、困惑度。