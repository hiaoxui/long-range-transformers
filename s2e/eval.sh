export CACHE_DIR=$1
export DATA_DIR=$HOME/data/ontonotes

export MODEL_NAME=$(python -c "
args=open('$1/args.txt').read();
if 'longformer-base' in args:
    print('allenai/longformer-base-4096');
elif 'bigbird' in args:
    print('google/bigbird-roberta-base')
elif 'xlnet-base' in args:
    print('xlnet-base-cased');
elif 'roberta' in args:
    print('roberta-base')
elif 'spanbert' in args:
    print('SpanBERT/spanbert-base-cased')
")
export SEGMENT=$(python -c "
import re
args=open('$1/args.txt').read();
rst = re.findall(r'segment=(\d+)', args)
if len(rst) == 1:
    print(rst[0])
else:
    print(4096)
")
export MEM=$(python -c "
import re
args=open('$1/args.txt').read();
rst = re.findall(r'mem=(\d+)',args)
if len(rst) == 1:
    print(rst[0])
else:
    print(0)
")
export GLOBAL=$(python -c "
args=open('$1/args.txt').read();
if 'use_global=True' in args:
    print('true')
else:
    print('false')
")
export PERFORMER=$(python -c "
if 'performer' in '$1':
    print(' --performer ')
else:
    print('')
")
export LSTM=$(python -c "
import re
args=open('$1/args.txt').read();
if len(re.findall(r'lstm=True', args)) > 0:
    print(' --lstm ')
else:
    print(' ')
")
export OVERLAP=$(python -c "
import re
args=open('$1/args.txt').read();
if len(re.findall('overlap=True', args)) > 0:
    print(' --overlap ')
else:
    print(' ')
")

export ARGS=" -d $DATA_DIR -m $MODEL_NAME --do_eval --max_total_seq_len=30000 --experiment_name=eval_model --mem $MEM --segment $SEGMENT $PERFORMER $LSTM $OVERLAP "
if [ $GLOBAL == "true" ]; then
    export ARGS="$ARGS --use_global"
fi

if [ ! -d $CACHE_DIR/final_eval ] && [ -f $CACHE_DIR/dev_out/pytorch_model.bin ]; then
    echo eval on final model
    python run_coref.py -s $CACHE_DIR/final_eval --pm $CACHE_DIR/dev_out/pytorch_model.bin $ARGS
    python run_coref.py -s $CACHE_DIR/final_eval --pm $CACHE_DIR/dev_out/pytorch_model.bin --do_test $ARGS
fi

if [ -d $CACHE_DIR/dev_out ]; then
    echo eval on checkpoint model
    export CKPT=$(python -c "
import re, os
max_step = -1
for fn in os.listdir(\"$CACHE_DIR/dev_out\"):
    rst = re.findall(r'checkpoint-(\d+)', fn)
    if len(rst) > 0:
        max_step = max(max_step, int(rst[0]))
print(\"$CACHE_DIR/dev_out/checkpoint-\" + str(max_step))
    ")
    python run_coref.py -s $CACHE_DIR/ckpt_eval --pm $CKPT/pytorch_model.bin $ARGS
    python run_coref.py -s $CACHE_DIR/ckpt_eval --pm $CKPT/pytorch_model.bin --do_test $ARGS
fi


