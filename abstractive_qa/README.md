# Longformer Encoder Decoder Baselines for Qasper

This is an implementation of the baselines reported in the paper **A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers** by Dasigi et al., published at NAACL 2021.

## Prerequisites

 - Download data from [here](https://allenai.org/data/qasper).
 
```
pip install -r requirements.txt
```

## Experiments

### Without evidence selection scaffold

```shell
export SEGMENT_LENGTH=4096 #512
export USE_GLOBAL=1 #0

allennlp train training_config/config.jsonnet -s cache-512 --include-package qasper_baselines
```
