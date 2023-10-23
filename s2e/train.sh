#!/usr/bin/bash
ARGS="-m $ENCODER -s cache/$EXP -d /home/gqin2/data/ontonotes --do_train --do_eval --num_train_epochs 129 --save_steps 3000 --eval_steps 1000 --max_seq_length $MAXSEQ --normalise_loss --max_total_seq_len $MAXSEQ --warmup_steps 5600 --head_learning_rate 3e-4 --top_lambda 0.4 --experiment_name \"$EXP\" --save_if_best --checkpoint "

if [ ! -z ${SEGMENT} ]; then
    ARGS="$ARGS --segment $SEGMENT "
fi

if [ ! -z ${MEM} ]; then
    ARGS="$ARGS --mem $MEM "
fi

if [ $GLOBAL = "true" ]; then
    ARGS="$ARGS --use_global "
fi

if [ $AMP = "true" ]; then
    ARGS="$ARGS --amp "
fi

if [ $NO_LOAD = "true" ]; then
    ARGS="$ARGS --no_load "
fi

if [ $LSTM = "true" ]; then
    ARGS="$ARGS --lstm "
fi

if [ $OVERLAP = "true" ]; then
    ARGS="$ARGS --overlap "
fi

echo $ARGS

python3 run_coref.py $ARGS

