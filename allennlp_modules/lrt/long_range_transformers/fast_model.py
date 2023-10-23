import os

from torch import nn
from transformers.models.bert.modeling_bert import (
    BertModel, BertLayer, BertAttention, BertEmbeddings, BertEncoder, BertPooler, BertIntermediate,
    BertOutput, BertSelfOutput, BertSelfAttention
)
from .fast_adaptor import FastSelfAttn


class FastBertAttention(BertAttention):
    def __init__(self, config, attn_args):
        super(BertAttention, self).__init__()
        if attn_args['type'] == 'huggingface':
            self.self = BertSelfAttention(config)
        else:
            self.self = FastSelfAttn(config, attn_args)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()


class FastBertLayer(BertLayer):
    def __init__(self, config, attn_args):
        super(BertLayer, self).__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = FastBertAttention(config, attn_args)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)


class FastBertEncoder(BertEncoder):
    def __init__(self, config, attn_config):
        super(BertEncoder, self).__init__()
        self.config = config
        if 'FAST_LAYER' not in os.environ:
            self.layer = nn.ModuleList([FastBertLayer(config, attn_config) for _ in range(config.num_hidden_layers)])
        else:
            layer_indices = list(map(int, os.environ['FAST_LAYER'].split(',')))
            layers = list()
            for idx in range(config.num_hidden_layers):
                if idx in layer_indices:
                    layers.append(FastBertLayer(config, attn_config))
                else:
                    layers.append(FastBertLayer(config, {'type': 'huggingface'}))
            self.layer = nn.ModuleList(layers)
        self.gradient_checkpointing = False


class FastBert(BertModel):
    def __init__(self, config, add_pooling_layer=True, attn_args=None):
        super(BertModel, self).__init__(config)
        if attn_args is None:
            attn_args = {'type': 'huggingface'}
        # Some shortcut
        if attn_args['type'] == 'performer':
            attn_args = {'type': 'linear', 'feature_map': 'favor'}
        elif attn_args['type'] == 'random_fourier_features':
            attn_args = {'type': 'linear', 'feature_map': 'random_fourier_features'}
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = FastBertEncoder(config, attn_args)
        self.pooler = BertPooler(config) if add_pooling_layer else None
        self.init_weights()
