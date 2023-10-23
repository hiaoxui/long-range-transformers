# How to run

Example bash script for NLI:

```shell
#!/usr/bin/env bash

`cd allennlp_modules

export ENCODER="xlnet-base-cased"
export DATA="data/docnli"
export DEVICE="0"
export MEM="256"
export SEGMENT=256
export USE_GLOBAL="false"
export MAX_LENGTH=1024
export BATCH_SIZE=4
`
cd /home/gqin2/dev/lrt/coarse2fine_coref
allennlp train --include-package lre -s cache/debug -f config/nli.jsonnet
```

For coref, you should change the config to `config/coref.jsonnet`.

Explanations of the environment variables:

- `ENCODER`: Transformer models. 
We currently support XLNet, Longformer, and most of the simple transformers.
- `DATA`: Data path.
- `DEVICE`: CUDA device number. If input is a list, like "[0,1,2,3]", DDP will be used.
- `MEM`: Mem length for XLNet. No effect for other models.
- `SEGMENT`: For XLNet, the length of each segment for recurrence.
For other transformers, the chunk size for the splitting.
- `USE_GLOBAL`: Longformer will treat all tokens as global tokens. 
No effect for other transformers.
- `MAX_LENGTH`: For NLI experiments, will guarantee `len(premise) + len(hypothesis) < MAX_LENGTH`.
If not satisfied, will truncate premise as the first priority.
- `BATCH_SIZE`: The name speaks for itself.
