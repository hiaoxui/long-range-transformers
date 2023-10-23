# Document-Level NLI

We're specifically interested in the dataset [DocNLI](https://arxiv.org/abs/2106.09449).
It proposed a simple baseline, where we use transformers to encode the concatenation of premise and hypothesis,
and do a classification on the CLS token.
Based on their experiments,
RoBERTa is shown to be more effective than Longformer while RoBERTa is unable to process long contexts.

The labels in DocNLI are collapsed: We only have entailment and non-entailment labels.

The metrics of this dataset is accuracy and F1.

Since this is a sequence classification, we cannot simply chunk the sequences and concatenate them.
We can only truncate the sequence.


