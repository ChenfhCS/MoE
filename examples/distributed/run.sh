#!/bin/bash
if [[ $1 == 'xl' ]]; then
    echo 'Run training on MoE-TransformerXL...'
    python main.py \
        --model_name 'xl' \
        --num_layer 12 \
        --train_batch_size 2 \
        --eval_batch_size 2 \
        --num_epochs 1 \
        --cuda \
        --debug \
        --log_interval 10 \
        --work_dir 'logs/' \
        --moe \
        --moe-num-experts 2 \
        --moe-top-k 2 \
        # --use_wandb \
        ${@:2}
elif [[ $1 == 'bert' ]]; then
    echo 'Run training on MoE-BERT...'
    python main.py \
        --model_name 'bert' \
        --num_layer 12 \
        --train_batch_size 2 \
        --eval_batch_size 2 \
        --num_epochs 1 \
        --cuda \
        --debug \
        --log_interval 10 \
        --work_dir 'logs/' \
        --moe \
        --moe-num-experts 2 \
        --moe-top-k 2 \
        # --use_wandb \
        ${@:2}
elif [[ $1 == 'gpt' ]]; then
    echo 'Run training on MoE-GPT2...'
    python main.py \
        --model_name 'gpt' \
        --num_layer 12 \
        --train_batch_size 2 \
        --eval_batch_size 2 \
        --num_epochs 1 \
        --cuda \
        --debug \
        --log_interval 10 \
        --work_dir 'logs/' \
        --moe \
        --moe-num-experts 1 \
        --moe-top-k 1 \
        # --use_wandb \
        ${@:2}   
else
    echo 'unknown argment 1'
fi