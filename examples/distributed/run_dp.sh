#!/bin/bash
if [[ $1 == 'xl' ]]; then
    echo 'Run distributed training on MoE-TransformerXL...'
    torchrun --nproc_per_node 4 --nnodes 1 --node_rank 0 --master_addr "127.0.0.1" --master_port 1234 main.py \
        --model_name 'xl' \
        --train_batch_size 2 \
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
    torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr "127.0.0.1" --master_port 1234 main.py \
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
        --moe-num-experts 1 \
        --moe-top-k 1 \
        --expert_parallel \
        --moe_world_size 1 \
        # --use_wandb \
        ${@:2}
elif [[ $1 == 'gpt' ]]; then
    echo 'Run distributed training on MoE-GPT2...'
    torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr "127.0.0.1" --master_port 1234 main.py \
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
        --moe-num-experts 1 \
        --moe-top-k 1 \
        --expert_parallel \
        --moe_world_size 1 \
        # --use_wandb \
        ${@:2}   
else
    echo 'unknown argment 1'
fi