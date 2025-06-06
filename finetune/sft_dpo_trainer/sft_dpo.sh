#!/bin/bash

# DPO训练脚本示例
# 使用带有SFT损失的DPO训练

python finetune_dpo_trainer.py \
    --model_name_or_path "/root/autodl-tmp/MiniCPM3-4B" \
    --train_data_path "/root/autodl-tmp/dpo_train_data.json" \
    --eval_data_path "/root/autodl-tmp/dpo_train_data.json" \
    --output_dir "./output_dpo_sft" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-6 \
    --weight_decay 0.01 \
    --warmup_steps 100 \
    --logging_steps 10 \
    --save_steps 500 \
    --eval_steps 500 \
    --model_max_length 512 \
    --use_lora True \
    --bf16 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --remove_unused_columns False \
    --use_dpo True \
    --dpo_beta 0.1 \
    --sft_loss_weight 0.5