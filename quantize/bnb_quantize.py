"""
the script will use bitandbytes to quantize the MiniCPM language model.
the be quantized model can be finetuned by MiniCPM or not.
you only need to set the model_path 、save_path and run bash code 

cd MiniCPM
python quantize/bnb_quantize.py

you will get the quantized model in save_path、quantized_model test time and gpu usage
"""


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
import torch
import GPUtil
import os

model_path = "/root/ld/ld_model_pretrain/MiniCPM-1B-sft-bf16"  # 模型下载地址
save_path = "/root/ld/ld_model_pretrain/MiniCPM-1B-sft-bf16_int4"  # 量化模型保存地址
device = "cuda" if torch.cuda.is_available() else "cpu"

# 创建一个配置对象来指定量化参数
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

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=device,  # 分配模型到device
    quantization_config=quantization_config,
    trust_remote_code=True,
)

gpu_usage = GPUtil.getGPUs()[0].memoryUsed
start = time.time()
response =  model.chat(tokenizer, "<用户>给我讲一个故事<AI>",history=[], temperature=0.5, top_p=0.8, repetition_penalty=1.02)  # 模型推理
print("量化后输出", response)
print("量化后推理用时", time.time() - start)
print(f"量化后显存占用: {round(gpu_usage/1024,2)}GB")


# 保存模型和分词器
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path, safe_serialization=True)
tokenizer.save_pretrained(save_path)
