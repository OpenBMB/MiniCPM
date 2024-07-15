"""
由于autogptq已经不更新很久了，使用gptq量化前，请先安装我们的autogptq分支,否则代码无法正常运行。

‘’‘bash
git clone https://github.com/LDLINGLINGLING/AutoGPTQ/tree/minicpm_gptq
cd Autogptq
pip install e .
‘’‘

"""





import json
import random
import time
from argparse import ArgumentParser
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

import os
import shutil

def copy_missing_files(src_path, dst_path):
    src_files=os.listdir(src_path)
    dst_files=os.listdir(dst_path)
    for src_file in src_files:
        if src_file not in dst_files and src_file.endswith(('.bin', '.json'))!=True and src_file.startswith('.')!=True:
            src_file_path = os.path.join(src_path, src_file)
            dst_file_path = os.path.join(dst_path, src_file)
            shutil.copy2(src_file_path, dst_file_path)

def load_data(data_path, tokenizer, n_samples):

    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    raw_data = random.sample(raw_data, k=min(n_samples, len(raw_data)))
    def dummy_gen():
        return raw_data

    def tokenize(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]

        prompts = []
        texts = []
        input_ids = []
        attention_mask = []
        for istr, inp, opt in zip(instructions, inputs, outputs):
            if inp:
                prompt = f"<AI>:\n{istr+inp}<AI>:\n"
                text = "<s>" + prompt + opt + "</s>"
            else:
                prompt = f"<USER>\n{istr}\n<AI>:\n"
                text = "<s>" + prompt + opt+ "</s>"
            if len(tokenizer(prompt)["input_ids"]) >= tokenizer.model_max_length:
                continue

            tokenized_data = tokenizer(text)

            input_ids.append(tokenized_data["input_ids"][: tokenizer.model_max_length])
            attention_mask.append(tokenized_data["attention_mask"][: tokenizer.model_max_length])
            prompts.append(prompt)
            texts.append(text)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt": prompts,
        }

    dataset = Dataset.from_generator(dummy_gen)

    dataset = dataset.map(
        tokenize,
        batched=True,
        batch_size=len(dataset),
        num_proc=1,
        keep_in_memory=True,
        load_from_cache_file=False,
        remove_columns=["instruction", "input"],
    )

    dataset = dataset.to_list()

    for sample in dataset:
        sample["input_ids"] = torch.LongTensor(sample["input_ids"])
        sample["attention_mask"] = torch.LongTensor(sample["attention_mask"])

    return dataset


def main():
    parser = ArgumentParser()
    parser.add_argument("--pretrained_model_dir", type=str,default='/root/ld/ld_model_pretrain/MiniCPM-1B-sft-bf16')
    parser.add_argument("--quantized_model_dir", type=str, default='/root/ld/pull_request/MiniCPM/quantize/gptq_cpm_1b_4bit')
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4])#do not use 8 bit
    parser.add_argument(
        "--group_size",
        type=int,
        default=128,
        help="group size, -1 means no grouping or full rank",
    )
    parser.add_argument("--desc_act", action="store_true", default=True,help="whether to quantize with desc_act")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=128,
        help="how many samples will be used to quantize model",
    )
    parser.add_argument(
        "--save_and_reload",
        action="store_true",
        default=True,
        help="whether save quantized model to disk and reload back",
    )
    parser.add_argument("--fast_tokenizer", action="store_true", help="whether use fast tokenizer")
    parser.add_argument(
        "--use_triton",
        action="store_true",
        help="whether use triton to speedup at inference",
    )
    parser.add_argument(
        "--per_gpu_max_memory",
        type=int,
        default=None,
        help="max memory used to load model per gpu",
    )
    parser.add_argument(
        "--cpu_max_memory",
        type=int,
        default=None,
        help="max memory used to offload model to cpu",
    )
    parser.add_argument(
        "--quant_batch_size",
        type=int,
        default=8,
        help="examples batch size for quantization",
    )
    parser.add_argument(
        "--trust_remote_code",
        default=True,
        action="store_true",
        help="whether to trust remote code when loading model",
    )
    parser.add_argument(
        "--quant_data",
        default='quantize_data/alpaca_data_cleaned.json',
        help="the quant data path",
    )

    args = parser.parse_args()

    max_memory = {}
    if args.per_gpu_max_memory is not None and args.per_gpu_max_memory > 0:
        if torch.cuda.is_available():
            max_memory.update({i: f"{args.per_gpu_max_memory}GIB" for i in range(torch.cuda.device_count())})
    if args.cpu_max_memory is not None and args.cpu_max_memory > 0 and max_memory:
        max_memory["cpu"] = f"{args.cpu_max_memory}GIB"
    if not max_memory:
        max_memory = None

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_dir,
        use_fast=args.fast_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoGPTQForCausalLM.from_pretrained(
        args.pretrained_model_dir,
        quantize_config=BaseQuantizeConfig(bits=args.bits, group_size=args.group_size, desc_act=args.desc_act),
        max_memory=max_memory,
        trust_remote_code=args.trust_remote_code,
    )

    examples = load_data(args.quant_data, tokenizer, args.num_samples)
    examples_for_quant = [
        {"input_ids": example["input_ids"], "attention_mask": example["attention_mask"]} for example in examples
    ]

    start = time.time()
    model.quantize(
        examples_for_quant,
        batch_size=args.quant_batch_size,
        use_triton=args.use_triton,
        autotune_warmup_after_quantized=args.use_triton,
    )
    end = time.time()
    print(f"quantization took: {end - start: .4f}s")

    if not args.quantized_model_dir:
        args.quantized_model_dir = args.pretrained_model_dir

    if args.save_and_reload:
        model.save_quantized(args.quantized_model_dir)
        tokenizer.save_pretrained(args.quantized_model_dir)
        copy_missing_files(args.pretrained_model_dir,args.quantized_model_dir)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = AutoGPTQForCausalLM.from_quantized(
            args.quantized_model_dir,
            device="cuda:0",
            use_triton=args.use_triton,
            max_memory=max_memory,
            inject_fused_mlp=True,
            inject_fused_attention=True,
            trust_remote_code=args.trust_remote_code,
        )

    pipeline_init_kwargs = {"model": model, "tokenizer": tokenizer}
    if not max_memory:
        pipeline_init_kwargs["device"] = "cuda:0"
    for example in random.sample(examples, k=min(4, len(examples))):
        print(f"prompt: {example['prompt']}")
        print("-" * 42)
        print(f"golden: {example['output']}")
        print("-" * 42)
        start = time.time()
        print(tokenizer.decode(model.generate(**tokenizer("{}".format(example['prompt']), return_tensors="pt").to(model.device),max_new_tokens=100)[0]))

if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()