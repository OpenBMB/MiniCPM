import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoConfig
import GPUtil
import argparse

parser = argparse.ArgumentParser(description="========量化困惑度测试========")
parser.add_argument(
    "--model_path",  
    type=str,  
    default='',  
    help="未量化前的模型路径。"  
)
parser.add_argument(
    "--bnb_path",  
    type=str,  
    default='',  
    help="bnb量化后的模型路径。"  
)
parser.add_argument(
    "--awq_path",  
    type=str,  
    default='',  
    help="awq量化后的模型保存路径。"  
)
#we will support gptq later
parser.add_argument(
    "--gptq_path",  
    type=str,  
    default='',  
    help="gptq量化后的模型保存路径。"  
)
parser.add_argument(
    "--data_path",  
    type=str,  
    default='quantize_data/wikitext',  
    help="可以是以后的量化数据集，示例中默认为wiki_text"  
)

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda:0"
    else:
        return "cpu"

def evaluate_perplexity(model, tokenizer,data_path):
    def _perplexity(nlls, n_samples, seqlen):
        return torch.exp(torch.stack(nlls).sum() / (n_samples * seqlen))

    data = load_dataset(data_path,  split="test")
    data = tokenizer("\n\n".join(data["text"]), return_tensors="pt")
    data = data.input_ids.to('cuda:0')

    seqlen = 2048
    model = model.eval()
    n_samples = data.numel() // seqlen

    nlls = []

    with tqdm(range(n_samples), desc="Perplexity -") as progress_bar:
        for i in progress_bar:
            start_index = i * seqlen
            end_index = (i + 1) * seqlen
            batch = data[:, start_index:end_index].to('cuda:0')
            with torch.no_grad():
                logits = model(batch).logits
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = data[:, start_index:end_index][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)

            curr_ppl = _perplexity(nlls, i + 1, seqlen)
            progress_bar.set_description(f"Perplexity {curr_ppl:.3f}")

    ppl = _perplexity(nlls, n_samples, seqlen)

    return ppl.item()

if __name__ == "__main__":
    
    args = parser.parse_args()

    if args.model_path != "":
        print("pretrained model：",args.model_path.split('/')[-1])
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map='cuda', trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        gpu_usage = GPUtil.getGPUs()[0].memoryUsed 
        print(f"gpu usage: {round(gpu_usage/1024,2)}GB")
        evaluate_perplexity(model, tokenizer, args.data_path)
        del model

    if args.awq_path != "":
        from awq import AutoAWQForCausalLM
        print("awq model：",args.awq_path.split('/')[-1])
        model = AutoAWQForCausalLM.from_quantized(args.awq_path, fuse_layers=True,device_map={"":'cuda:0'})
        tokenizer = AutoTokenizer.from_pretrained(args.awq_path)
        gpu_usage = GPUtil.getGPUs()[0].memoryUsed 
        print(f"gpu usage: {round(gpu_usage/1024,2)}GB")
        evaluate_perplexity(model, tokenizer, args.data_path)
        del model

#we will support the autogptq  later
    if args.gptq_path != "":
        from auto_gptq import AutoGPTQForCausalLM
        print("gptq model：",args.gptq_path.split('/')[-1])
        tokenizer = AutoTokenizer.from_pretrained(args.gptq_path, use_fast=True)
        model = AutoGPTQForCausalLM.from_quantized(args.gptq_path, device="cuda:0",trust_remote_code=True)
        gpu_usage = GPUtil.getGPUs()[0].memoryUsed 
        print(f"gpu usage: {round(gpu_usage/1024,2)}GB")
        evaluate_perplexity(model, tokenizer, args.data_path)
        del model
    
    if args.bnb_path != "":
        from accelerate.utils import BnbQuantizationConfig
        bnb_quantization_config = BnbQuantizationConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
        print("bnb model：",args.gptq_path.split('/')[-1])
        # config=AutoConfig.from_pretrained(args.bnb_path,trust_remote_code=True)
        # bnb_config=config.quantization_config
        tokenizer = AutoTokenizer.from_pretrained(args.bnb_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(args.bnb_path, trust_remote_code=True,)#quantization_config=bnb_config,)
        gpu_usage = GPUtil.getGPUs()[0].memoryUsed 
        print(f"gpu usage: {round(gpu_usage/1024,2)}GB")
        evaluate_perplexity(model, tokenizer, args.data_path)
        del model
