#!/bin/bash

CHECKPOINT=$1
SAVE_TO=$2
HF_MODEL_NAME=wip.$SAVE_TO  # huggingface上的模型名

python vllm_convert_checkpoint_to_hf.py --load $CHECKPOINT --save $SAVE_TO
python inference.py --model_path=$HF_MODEL_NAME
