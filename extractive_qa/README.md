## Prepare Data
```shell
wget https://nlp.cs.washington.edu/triviaqa/data/triviaqa-rc.tar.gz
tar -xvzf triviaqa-rc.tar.gz -C triviaqa-rc/
rm triviaqa-rc.tar.gz

mkdir data

python -m triviaqa_utils.convert_to_squad_format  \
  --triviaqa_file triviaqa-rc/qa/wikipedia-train.json  \
  --wikipedia_dir triviaqa-rc/evidence/wikipedia/   \
  --web_dir triviaqa-rc/evidence/web/  \
  --max_num_tokens 4096 \ 
  --squad_file data/squad-wikipedia-train.json

python -m triviaqa_utils.convert_to_squad_format  \
  --triviaqa_file triviaqa-rc/qa/wikipedia-dev.json  \
  --wikipedia_dir triviaqa-rc/evidence/wikipedia/   \
  --web_dir triviaqa-rc/evidence/web/  \
  --max_num_tokens 4096 \
  --squad_file data/squad-wikipedia-dev.json
```
## How To Run
```shell
SAVE_PREFIX=""
$MODEL_NAME=allenai/longformer-base-4096 #xlnet-base-cased / google/bigbird-roberta-base
$USE_CONTEXT=1 #0
$SEG_LENGTH=4096 #512
$CTX_LENGTH=512 #64
$USE_GLOBAL=1 #0

python3 run.py \
    --train_dataset data/squad-wikipedia-train.json \
    --dev_dataset data/squad-wikipedia-dev.json \
    --gpus 4 --max_seq_len 4096 --doc_stride -1 \
    --save_prefix $SAVE_PREFIX \
    --model-name=$MODEL_NAME \
    --use-context=$USE_CONTEXT \
    --segment-length=$SEG_LENGTH \
    --context-length=$CTX_LENGTH \
    --use-global=$USE_GLOBAL
```