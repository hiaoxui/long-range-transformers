#!/usr/bin/bash
ARGS="-m $ENCODER -s cache/$EXP -d /home/gqin2/data/ontonotes --do_train --do_eval --num_train_epochs 129 --save_steps 3000 --eval_steps 1000 --max_seq_length $MAXSEQ --normalise_loss --max_total_seq_len $MAXSEQ --warmup_steps 5600 --head_learning_rate 3e-4 --top_lambda 0.4 --experiment_name \"$EXP\" --save_if_best --checkpoint --segment $SEGMENT --performer --pm $ARCHIVE/pytorch_model.bin"

if [ $AMP = "true" ]; then
    ARGS="$ARGS --amp "
fi

echo $ARGS

python3 run_coref.py $ARGS

