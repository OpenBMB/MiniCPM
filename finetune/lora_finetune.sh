formatted_time=$(date +"%Y%m%d%H%M%S")
echo $formatted_time

CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --model_name_or_path <your_model_name_or_path> \
    --output_dir output/AdvertiseGenLoRA/$formatted_time/ \
    --train_data_path data/AdvertiseGenChatML/train.json \
    --eval_data_path data/AdvertiseGenChatML/dev.json \
    --learning_rate 1e-3 --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 32 --fp16\
    --gradient_accumulation_steps 8 --warmup_steps 100 \
    --max_steps 3000 --weight_decay 0.01 \
    --evaluation_strategy steps --eval_steps 500 \
    --save_strategy steps --save_steps 500 \
    --use_lora true --seed 42 \
    --log_level info --logging_strategy steps --logging_steps 10
