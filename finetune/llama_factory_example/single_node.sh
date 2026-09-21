#!/bin/bash

NPROC_PER_NODE=8
NNODES=1
RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1 
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    src/train.py /root/ld/ld_project/LLaMA-Factory/examples/minicpm/minicpm_sft.yaml
