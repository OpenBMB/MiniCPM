import argparse
import json
import os
import shutil
import sys
from collections import OrderedDict

import torch
from transformers import AutoTokenizer

from vllm.transformers_utils.configs import CPMMistralConfig


def find_ckpt(directory):
    files = os.listdir(directory)
    ckpt = [file for file in files if file.endswith(".pt")]
    assert len(ckpt) == 1
    return '/'.join((directory, ckpt[0]))

def convert_model(ckpt, layernum):
    print("Total number in orgckpt", len(ckpt))
    model_hf = OrderedDict()

    model_hf['model.embed_tokens.weight'] = ckpt["input_embedding.weight"].contiguous()
    model_hf['model.norm.weight'] = ckpt["encoder.output_layernorm.weight"].contiguous()
    try:
        model_hf['lm_head.weight'] = ckpt['lm_head.weight'].contiguous()
        print("lm_head found")
    except:
        model_hf['lm_head.weight'] = ckpt["input_embedding.weight"].contiguous()
        print("lm_head not found")


    # print(ckpt.keys())

    for lnum in range(layernum):
        hf_pfx = f"model.layers.{lnum}"
        bmt_pfx = f"encoder.layers.{lnum}"
        
        model_hf[f"{hf_pfx}.input_layernorm.weight"] = ckpt[f"{bmt_pfx}.self_att.layernorm_before_attention.weight"].contiguous()

        model_hf[f"{hf_pfx}.self_attn.q_proj.weight"] = ckpt[f"{bmt_pfx}.self_att.self_attention.project_q.weight"].contiguous()
        model_hf[f"{hf_pfx}.self_attn.k_proj.weight"] = ckpt[f"{bmt_pfx}.self_att.self_attention.project_k.weight"].contiguous()
        model_hf[f"{hf_pfx}.self_attn.v_proj.weight"] = ckpt[f"{bmt_pfx}.self_att.self_attention.project_v.weight"].contiguous()
        model_hf[f"{hf_pfx}.self_attn.o_proj.weight"] = ckpt[f"{bmt_pfx}.self_att.self_attention.attention_out.weight"].contiguous()

        model_hf[f"{hf_pfx}.post_attention_layernorm.weight"] = ckpt[f"{bmt_pfx}.ffn.layernorm_before_ffn.weight"].contiguous()

        model_hf[f"{hf_pfx}.mlp.gate_proj.weight"] = ckpt[f"{bmt_pfx}.ffn.ffn.w_in.w_0.weight"].contiguous()
        model_hf[f"{hf_pfx}.mlp.up_proj.weight"] = ckpt[f"{bmt_pfx}.ffn.ffn.w_in.w_1.weight"].contiguous()

        model_hf[f"{hf_pfx}.mlp.down_proj.weight"] = ckpt[f"{bmt_pfx}.ffn.ffn.w_out.weight"].contiguous()

        try:
            model_hf[f"{hf_pfx}.self_attn.q_norm.weight"] = ckpt[f"{bmt_pfx}.self_att.self_attention.q_norm.weight"].contiguous()
            model_hf[f"{hf_pfx}.self_attn.k_norm.weight"] = ckpt[f"{bmt_pfx}.self_att.self_attention.k_norm.weight"].contiguous()
        except:
            print("Error!")
            pass

    
    print("Total number in converted", len(model_hf))

    return model_hf






        


def load_model_ckpt(args):
    ckpt = torch.load(find_ckpt(args.load))
    # with open(os.path.join(os.path.dirname(args.load), "config.json"), 'r') as fin:
    #     config = json.load(fin)
    config = CPMMistralConfig.from_pretrained(args.load)

    print(config)
    # from IPython import embed; embed(header="222")
    
    if args.save is not None:
        config_name = args.save #"cpmlive_llama7b" #os.path.dirname(args.load).split("/")[-1]
    else:
        config_name = args.load

    hf_ckpt = convert_model(ckpt, config.num_layers)

    

    os.makedirs(f"{config_name}", exist_ok=True)

    torch.save(hf_ckpt, f"{config_name}/pytorch_model.bin")
    config.save_pretrained(f"{config_name}")

    # try:
    #     tokenizer = AutoTokenizer.from_pretrained(
    #             args.load,
    #             trust_remote_code=True)
    #     tokenizer.save_pretrained(f"{config_name}")

    #     # for fast tokenizer bug
    #     if 'tokenizer.json' in os.listdir(f"{config_name}"):
    #         os.remove(f"{config_name}/tokenizer.json")

    #     print("Tokenizer loaded")
    # except Exception as e:
    #     print(e)
    #     try:
    #         shutil.copyfile("autoeval/120k_v2.model", f"wip.{config_name}/tokenizer.model")
    #         shutil.copyfile("autoeval/special_tokens_map.json", f"wip.{config_name}/special_tokens_map.json")
    #         shutil.copyfile("autoeval/tokenizer_config.json", f"wip.{config_name}/tokenizer_config.json")
    #         print("Tokenizer copied")
    #     except Exception as e:
    #         print("Tokenizer not exists!")
    #         print(e)
       
    

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--load", type=str, default="/mnt/data/user/tc_agi/user/zhaoweilin/cpmlive-llama-2-7b/pytorch_model.pt")
    parser.add_argument("--load", type=str, default="wip.job_469747_ckpt_4069")

    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    load_model_ckpt(args)
