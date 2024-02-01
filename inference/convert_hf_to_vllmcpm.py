import argparse
import json
import os
import shutil
from tqdm import tqdm
from collections import OrderedDict
import torch

def convert_model(config, ckpt):
    # config
    config_bmt = OrderedDict(
        {
            "_dtype": "bf16",
            "activate_fn": "silu",
            "architectures": [
                "CPMDragonflyForCausalLM"
            ],
            "model_type": "cpm_dragonfly",
            "base": 10000,
            "dim_ff": config['intermediate_size'],
            "dim_head": config['hidden_size'] // config['num_attention_heads'],
            "dim_model": config['hidden_size'],
            "dim_model_base": 256,
            "dropout_p": 0.0,
            "eps": config['rms_norm_eps'],
            "init_std": config['initializer_range'],
            "num_heads": config['num_attention_heads'],
            "num_kv_heads": config['num_key_value_heads'],
            "num_layers": config['num_hidden_layers'],
            "orig_max_length": 4096,
            "pose_prob": 0.0,
            "pose_scaling_factor": 1.0,
            "qk_norm": False,
            "rope_scaling_factor": 1,
            "rope_scaling_type": "",
            "scale": True,
            "scale_depth": config['scale_depth'],
            "scale_emb": config['scale_emb'],
            "tie_lm_head": True,
            "tp": 0,
            "transformers_version": "4.35.0",
            "vocab_size": config['vocab_size']
        }
    )


    model_bmt = OrderedDict()
    model_bmt["input_embedding.weight"] = ckpt['model.embed_tokens.weight'].contiguous()
    model_bmt["encoder.output_layernorm.weight"] = ckpt['model.norm.weight'].contiguous()
    for lnum in tqdm(range(config_bmt['num_layers'])):
        hf_pfx = f"model.layers.{lnum}"
        bmt_pfx = f"encoder.layers.{lnum}"
        model_bmt[f"{bmt_pfx}.self_att.layernorm_before_attention.weight"] = ckpt[f"{hf_pfx}.input_layernorm.weight"].contiguous()
        model_bmt[f"{bmt_pfx}.self_att.self_attention.project_q.weight"] = ckpt[f"{hf_pfx}.self_attn.q_proj.weight"].contiguous()
        model_bmt[f"{bmt_pfx}.self_att.self_attention.project_k.weight"] = ckpt[f"{hf_pfx}.self_attn.k_proj.weight"].contiguous()
        model_bmt[f"{bmt_pfx}.self_att.self_attention.project_v.weight"] = ckpt[f"{hf_pfx}.self_attn.v_proj.weight"].contiguous()
        model_bmt[f"{bmt_pfx}.self_att.self_attention.attention_out.weight"] = ckpt[f"{hf_pfx}.self_attn.o_proj.weight"].contiguous()
        model_bmt[f"{bmt_pfx}.ffn.layernorm_before_ffn.weight"] = ckpt[f"{hf_pfx}.post_attention_layernorm.weight"].contiguous()
        model_bmt[f"{bmt_pfx}.ffn.ffn.w_in.w_0.weight"] = ckpt[f"{hf_pfx}.mlp.gate_proj.weight"].contiguous()
        model_bmt[f"{bmt_pfx}.ffn.ffn.w_in.w_1.weight"] = ckpt[f"{hf_pfx}.mlp.up_proj.weight"].contiguous()
        model_bmt[f"{bmt_pfx}.ffn.ffn.w_out.weight"] = ckpt[f"{hf_pfx}.mlp.down_proj.weight"].contiguous()


    return config_bmt, model_bmt

def load_model_ckpt(args):
    with open(os.path.join(args.load, "config.json"), 'r') as fin:
        config = json.load(fin)
    ckpt = torch.load(os.path.join(args.load, "pytorch_model.bin"))

    os.makedirs(f"{args.save}", exist_ok=True)

    # model and config
    hf_config, hf_ckpt = convert_model(config, ckpt)
    with open(os.path.join(args.save, "config.json"), 'w') as fout:
        json.dump(hf_config, fout, indent=4)
    torch.save(hf_ckpt, f"{args.save}/pytorch_model.pt")

    # tokenizer
    shutil.copyfile(f"{args.load}/tokenizer.json", f"{args.save}/tokenizer.json")
    shutil.copyfile(f"{args.load}/tokenizer.model", f"{args.save}/tokenizer.model")
    shutil.copyfile(f"{args.load}/special_tokens_map.json", f"{args.save}/special_tokens_map.json")
    shutil.copyfile(f"{args.load}/tokenizer_config.json", f"{args.save}/tokenizer_config.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, default="")
    parser.add_argument("--save", type=str, default="")
    args = parser.parse_args()

    load_model_ckpt(args)