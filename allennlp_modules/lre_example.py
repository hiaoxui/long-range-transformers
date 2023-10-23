import torch
from transformers import AutoTokenizer

from lrt.long_range_transformers.long_range_embedder import LongRangeEmbedder


model_name = 'roberta-base'
tok = AutoTokenizer.from_pretrained(model_name)
sentences = [
    'The quick fox jumps over a lazy dog.'
    'The quick'
]
inputs = tok(sentences, padding=True, return_tensors='pt')

# This is equivalent to the huggingface implementation
# `max_length` is the segment length
dense_model = LongRangeEmbedder(model_name, max_length=512)
# Add attn_artgs to change the behavior
performer = LongRangeEmbedder(model_name, max_length=512, attn_args={'type': 'performer'})

output1 = dense_model.forward(token_ids=inputs['input_ids'], mask=inputs['attention_mask'])
output2 = performer.forward(token_ids=inputs['input_ids'], mask=inputs['attention_mask'])
