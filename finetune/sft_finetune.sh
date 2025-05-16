#!/bin/bash
# 获取当前时间戳（格式：年月日时分秒）
formatted_time=$(date +"%Y%m%d%H%M%S")
echo $formatted_time

# 启动DeepSpeed分布式训练
deepspeed --include localhost:0,1 finetune.py \
    # 预训练模型路径或名称
    --model_name_or_path MiniCPM-2B-sft-bf16 \
    # 输出目录（包含时间戳）
    --output_dir output/AdvertiseGenSFT/$formatted_time/ \
    # 训练数据路径
    --train_data_path data/AdvertiseGenChatML/train.json \
    # 验证数据路径
    --eval_data_path data/AdvertiseGenChatML/dev.json \
    # 学习率设置
    --learning_rate 5e-5 \
    # 每个设备的训练批次大小
    --per_device_train_batch_size 14 \
    # 每个设备的验证批次大小
    --per_device_eval_batch_size 32 \
    # 启用BF16混合精度
    --bf16 \
    # 梯度累积步数
    --gradient_accumulation_steps 2 \
    # 预热步数
    --warmup_steps 100 \
    # 最大训练步数
    --max_steps 3000 \
    # 权重衰减系数
    --weight_decay 0.01 \
    # 评估策略（按步数）
    --evaluation_strategy steps \
    # 每100步评估一次
    --eval_steps 100 \
    # 保存策略（按步数）
    --save_strategy steps \
    # 每500步保存一次
    --save_steps 500 \
    # 随机种子
    --seed 42 \
    # 日志级别
    --log_level info \
    # 日志记录策略（按步数）
    --logging_strategy steps \
    # 每10步记录一次日志
    --logging_steps 10 \
    # DeepSpeed配置文件路径
    --deepspeed configs/ds_config_zero2.json
