

from datasets import load_dataset
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import os

model_path = '/root/ld/ld_model_pretrained/MiniCPM-1B-sft-bf16' # model_path or model_id
quant_path = '/root/ld/ld_project/pull_request/MiniCPM/quantize/awq_cpm_1b_4bit' # quant_save_path
quant_data_path='/root/ld/ld_project/pull_request/MiniCPM/quantize/quantize_data/wikitext'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" } # "w_bit":4 or 8
quant_samples=512 # how many samples to use for calibration

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,device_map={"": "cuda:0"})

# Define data loading methods
def load_alpaca(quant_data_path):
    data = load_dataset(quant_data_path, split="train") # Set the absolute path to alpaca or huggingface id

    # concatenate data
    def concatenate_data(x):
        return {"text": '<s><用户>'+x['instruction'] + '<AI>' + x['input'] + '\n' + x['output']}
    
    concatenated = data.map(concatenate_data)[:quant_samples]
    return [text for text in concatenated["text"]]

def load_wikitext(quant_data_path):
    data = load_dataset(quant_data_path,  split="train")
    return [text for text in data["text"] if text.strip() != '' and len(text.split(' ')) > 20][:quant_samples]

# Quantize
model.quantize(tokenizer, quant_config=quant_config, calib_data=load_wikitext(quant_data_path=quant_data_path))

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')