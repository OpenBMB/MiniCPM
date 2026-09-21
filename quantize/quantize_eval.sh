#!/bin/bash

awq_path="/root/ld/pull_request/MiniCPM/quantize/awq_cpm_1b_4bit"
gptq_path="/root/ld/pull_request/MiniCPM/quantize/gptq_cpm_1b_4bit"
model_path="/root/ld/ld_model_pretrain/MiniCPM-1B-sft-bf16"
bnb_path="/root/ld/ld_model_pretrain/MiniCPM-1B-sft-bf16_int4"
python quantize_eval.py --awq_path "${awq_path}" \
 --model_path "${model_path}" --gptq_path "${gptq_path}" --bnb_path "${bnb_path}"