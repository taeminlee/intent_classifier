TOKENIZERS_PARALLELISM=false\
    python -W ignore train.py \
        --tags=kobert \
        --train_file=data/fci_train_val.txt \
        --dev_file=data/fci_test.txt \
        --gpus=1 \
        --precision=16 \
        --bs=20 \
        --max_epochs=1000