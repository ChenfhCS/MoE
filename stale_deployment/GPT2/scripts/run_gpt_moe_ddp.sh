if [[ $1 == 'train' ]]; then
        echo 'Run training...'
        torchrun --nproc_per_node 4 --nnodes 2 --node_rank 0 --master_addr "172.31.9.143" --master_port 2344 train.py \
                        --train_corpus           ../data/gpt2_wiki/build/corpus.train.txt \
                        --eval_corpus            ../data/gpt2_wiki/build/corpus.test.txt \
                        --vocab_path             ../data/gpt2_wiki/build/vocab.txt \
                        --save_checkpoint_path   works/ckpt-gpt2.pth \
                        --save_model_path        works/gpt2-pretrained.pth \
                        --batch_train            128 \
                        --batch_eval             8 \
                        --seq_len                64 \
                        --total_steps            110 \
                        --eval_steps             500 \
                        --save_steps             5000 \
                        --cuda \
                        --multi_gpu \
			            --moe --moe-num-expert 48 --moe-top-k 4 \
                        --expert_parallel \
                        ${@:2}
else
    echo 'unknown argment 1'
fi