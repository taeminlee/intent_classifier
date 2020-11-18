TOKENIZERS_PARALLELISM=false\
    python -W ignore train.py \
        --tags=distilkobert \
        --model=monologg/distilkobert \
        --train_file=data/fci_train_val.txt \
        --dev_file=data/fci_test.txt \
        --gpus=1 \
        --precision=16 \
        --bs=160 \
        --max_epochs=1000