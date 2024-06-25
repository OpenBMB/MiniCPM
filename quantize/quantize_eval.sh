#!/bin/bash

awq_path="/root/ld/ld_project/AutoAWQ/examples/awq_cpm_1b_4bit"
gptq_path=""
model_path=""

python quantize_eval.py --awq_path "${awq_path}" \
 --model_path "${model_path}" --gptq_path "${gptq_path}"