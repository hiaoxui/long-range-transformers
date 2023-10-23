## Data Processing
Put the data from https://github.com/mingdachen/SummScreen for summscreee and https://www.scrolls-benchmark.com/ for gov_report in the folder as below.
```
├── data
│   ├── sumscreen 
│   │   ├──fd_train.json
│   │   ├──fd_dev.json
│   │   └──fd_test.json
│   ├── gov_report
│   │   ├──train.jsonl
│   │   └──valid.jsonl
├── README.md
├── run.py
```

## How To Run
```shell
TASKNAME=gov #fd
LENGTH=1536 #15360

python3 run.py --task-name=$TASKNAME --segment-length=$LENGTH --ckpt-index=0
```