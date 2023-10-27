if [[ $1 == 'train' ]]; then
        echo 'Run training...'
        python train.py --train_corpus           ../data/gpt2_wiki/build/corpus.train.txt \
                        --eval_corpus            ../data/gpt2_wiki/build/corpus.test.txt \
                        --vocab_path             ../data/gpt2_wiki/build/vocab.txt \
                        --save_checkpoint_path   works/ckpt-gpt2.pth \
                        --save_model_path        works/gpt2-pretrained.pth \
                        --batch_train            8 \
                        --batch_eval             8 \
                        --seq_len                64 \
                        --total_steps            1000000 \
                        --eval_steps             500 \
                        --save_steps             5000 \
                        --cuda \
                        ${@:2}
else
    echo 'unknown argment 1'
fi