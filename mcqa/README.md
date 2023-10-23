## Data Processing
Put the data from https://www.scrolls-benchmark.com/ to the folder as below.
```
├── quality
│   ├──train.jsonl
│   └──validation.jsonl
├── README.md
├── run.py
```

## How To Run
```shell
LENGTH=1536 #15360
python3 run.py --segment-length=$LENGTH --ckpt-index=0 # --use-global
```