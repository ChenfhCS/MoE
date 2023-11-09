#!/bin/bash
if [[ $1 == 'xl' ]]; then
    echo 'Run distributed training on MoE-TransformerXL...'
    torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr "127.0.0.1" --master_port 1234 main.py \
        --model_name 'xl' \
        --num_layer 12 \
        --train_batch_size 4 \
        --eval_batch_size 4 \
        --num_epochs 1 \
        --cuda \
        --debug \
        --log_interval 10 \
        --work_dir 'logs/' \
        --moe \
        --moe-num-experts 4 \
        --moe-top-k 2 \
        --expert_parallel \
        # --use_wandb \
        ${@:2}
elif [[ $1 == 'bert' ]]; then
    echo 'Run distributed training on MoE-BERT...'
    torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr "127.0.0.1" --master_port 1234 main.py \
        --model_name 'bert' \
        --num_layer 12 \
        --train_batch_size 4 \
        --eval_batch_size 4 \
        --num_epochs 1 \
        --cuda \
        --debug \
        --log_interval 10 \
        --work_dir 'logs/' \
        --moe \
        --moe-num-experts 4 \
        --moe-top-k 2 \
        --expert_parallel \
        # --use_wandb \
        ${@:2}
elif [[ $1 == 'gpt' ]]; then
    echo 'Run distributed training on MoE-GPT2...'
    torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr "127.0.0.1" --master_port 1234 main.py \
        --model_name 'gpt' \
        --num_layer 12 \
        --train_batch_size 4 \
        --eval_batch_size 4 \
        --num_epochs 1 \
        --cuda \
        --debug \
        --log_interval 10 \
        --work_dir 'logs/' \
        --moe \
        --moe-num-experts 4 \
        --moe-top-k 2 \
        --expert_parallel \
        # --use_wandb \
        ${@:2}   
else
    echo 'unknown argment 1'
fi