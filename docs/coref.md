# Coreference with long context

## Coref Models

- coarse2fine ([paper](https://arxiv.org/abs/1804.05392)). This was done in 2018, combined with SpanBERT, which was
  proposed in 2019, achieved SOTA. It has AllenNLP implementations.
- e2e ([paper](https://arxiv.org/abs/2107.01700)). This is simple and neat, and we've got the codebase
  ([link](https://drive.google.com/file/d/1SLKX6JR4ynox9CP_hf9FOXMyAzPE2GeK/view)).

TODO: Check e2e codebase.

The above script would start training, after which the test will also be conducted on the test set.

## Memory usage experiments

I (Guanghui) observed that the model may take different amount of GPU memories with different versions of packages.
To save memory, I set up a few experiments to test the memory usage of the model with different combinations
of AllenNLP, HuggingFace Transformers, and PyTorch.
The script is saved to the scripts folder (`scripts/mem_test.sh` and `scripts/gen_script.py`).

However, as I observed, the memory usage is pretty much random, and I didn't find a "good" package version
that can significantly save memories.

The scripts there can be used in the future with other purposes like space complexity comparison
between different models.

