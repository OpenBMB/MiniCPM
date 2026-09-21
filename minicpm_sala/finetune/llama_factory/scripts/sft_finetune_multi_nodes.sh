accelerate launch \
    --config_file examples/accelerate/fsdp_config_multiple_nodes.yaml \
    src/train.py configs/minicpm_sft.yaml