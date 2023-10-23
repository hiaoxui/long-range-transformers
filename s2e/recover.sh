#!/usr/bin/bash

export ENCODER=allenai/longformer-base-4096
export SEGMENT=256
export MEM=0
export GLOBAL=true
export AMP=true
export MAXSEQ=2500
export EXP=lf-256-global

ARGS="-m $ENCODER -s cache/unfinished/s2e-lf-256-global_79390 -d /home/gqin2/data/ontonotes -r --do_train --do_eval --num_train_epochs 129 --save_steps 3000 --eval_steps 1000 --max_seq_length $MAXSEQ --normalise_loss --max_total_seq_len $MAXSEQ --warmup_steps 5600 --head_learning_rate 3e-4 --top_lambda 0.4 --experiment_name \"$EXP\" --save_if_best --checkpoint "

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

echo $ARGS

python3 run_coref.py $ARGS

