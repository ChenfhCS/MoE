#!/bin/bash
if [[ $1 == 'xl' ]]; then
    echo 'Run training on MoE-TransformerXL...'
    python main.py \
        --model_name 'xl' \
        --train_batch_size 1 \
        --eval_batch_size 1 \
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
        --train_batch_size 1 \
        --eval_batch_size 1 \
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
        --train_batch_size 1 \
        --eval_batch_size 1 \
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
else
    echo 'unknown argment 1'
fi