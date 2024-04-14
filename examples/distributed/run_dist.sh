#!/bin/bash
if [[ $1 == 'xl' ]]; then
    echo 'Run distributed training on MoE-TransformerXL...'
    torchrun --nproc_per_node 2 --nnodes 2 --node_rank 0 --master_addr "172.31.9.143" --master_port 2345 main.py \
        --model_name 'xl' \
        --train_batch_size 1 \
        --eval_batch_size 1 \
        --num_epochs 1 \
        --cuda \
        --debug \
        --log_interval 10 \
        --eval_interval 1000 \
        --work_dir 'logs/' \
        --moe \
        --moe-num-experts 4 \
        --moe-top-k 2 \
        --expert_parallel \
        --moe_world_size 4 \
        --use_wandb \
        ${@:2}
elif [[ $1 == 'bert' ]]; then
    echo 'Run distributed training on MoE-BERT...'
    torchrun --nproc_per_node 2 --nnodes 2 --node_rank 0 --master_addr "172.31.9.143" --master_port 2345 main.py \
        --model_name 'bert' \
        --train_batch_size 1 \
        --eval_batch_size 1 \
        --num_epochs 1 \
        --cuda \
        --debug \
        --log_interval 10 \
        --eval_interval 1000 \
        --work_dir 'logs/' \
        --moe \
        --moe-num-experts 4 \
        --moe-top-k 2 \
        --expert_parallel \
        --moe_world_size 4 \
        --use_wandb \
        ${@:2}
elif [[ $1 == 'gpt' ]]; then
    echo 'Run distributed training on MoE-GPT2...'
    torchrun --nproc_per_node 2 --nnodes 2 --node_rank 0 --master_addr "172.31.9.143" --master_port 2345 main.py \
        --model_name 'gpt' \
        --train_batch_size 1 \
        --eval_batch_size 1 \
        --num_epochs 1 \
        --cuda \
        --debug \
        --log_interval 10 \
        --eval_interval 1000 \
        --work_dir 'logs/' \
        --moe \
        --moe-num-experts 4 \
        --moe-top-k 2 \
        --expert_parallel \
        --moe_world_size 4 \
        --use_wandb \
        ${@:2}   
else
    echo 'unknown argment 1'
fi