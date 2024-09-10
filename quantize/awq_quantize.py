from datasets import load_dataset	
from awq import AutoAWQForCausalLM	
from transformers import AutoTokenizer	
import os	

model_path = '/root/ld/ld_model_pretrained/minicpm3' # model_path or model_id	
quant_path = '/root/ld/ld_model_pretrained/minicpm3_awq' # quant_save_path	
quant_data_path='/Users/liudan/ai/pull_request/MiniCPM/quantize/quantize_data/wikitext'# 写入自带数据集地址	
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" } # "w_bit":4 or 8	
quant_samples=512 # how many samples to use for calibration	
custom_data=[ # first custom data
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "assistant", "content": "你好，有什么我可以帮助你的吗？"},
                    {"role": "user", "content": "我想了解如何编写Python代码。"},
                ], # second custom data
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "assistant", "content": "你好，有什么我可以帮助你的吗？"},
                    {"role": "user", "content": "我想了解如何编写Python代码。"},
                ]
                 #....more custom data
            ]	
# Load model	
model = AutoAWQForCausalLM.from_pretrained(model_path,safetensors=False)	
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,device_map={"": "cuda:0"})	

# Define data loading methods	
def load_alpaca(quant_data_path):	
    data = load_dataset(quant_data_path, split="train") # Set the absolute path to alpaca or huggingface id	

    # concatenate data	
    def concatenate_data(x):
        if x['input'] and x['instruction']:
            line = [
                        {"role": "system", "content": x['instruction']},
                        {"role": "user", "content": x['input']},
                        {"role": "assistant", "content": x['output']},
                    ]
        elif x['input']:
            line = [    
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": x['input']},
                        {"role": "assistant", "content": x['output']},
                    ]
        else:
            line = [    
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": x['instruction']},
                        {"role": "assistant", "content": x['output']},
                    ]
        if model.config.architectures == ["MiniCPM3ForCausalLM"]:
            print(tokenizer.decode(tokenizer.apply_chat_template(line)))
            return {"text":tokenizer.decode(tokenizer.apply_chat_template(line))}
        else:
            return {"text": '<用户>'+x['instruction']  + x['input'] + '<AI>' + '\n' + x['output']}	

    concatenated = data.map(concatenate_data)[:quant_samples]	
    return [text for text in concatenated["text"]]	

def load_wikitext(quant_data_path):	
    data = load_dataset(quant_data_path,  split="train")	
    return [text for text in data["text"] if text.strip() != '' and len(text.split(' ')) > 20][:quant_samples]	

def load_cust_data(custom_data):
    quant_data=[tokenizer.decode(tokenizer.apply_chat_template(i)) for i in custom_data]	
    return quant_data[:quant_samples]	
# Quantize	
model.quantize(tokenizer, quant_config=quant_config, calib_data=load_wikitext(quant_data_path=quant_data_path))	

# Save quantized model	
model.save_quantized(quant_path)	
tokenizer.save_pretrained(quant_path)	

print(f'Model is quantized and saved at "{quant_path}"')