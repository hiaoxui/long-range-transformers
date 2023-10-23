import os

import torch
from torch import nn
from transformers.models.bert.modeling_bert import BertSelfAttention
try:
    from fast_transformers.masking import LengthMask, FullMask
    from fast_transformers.builders import AttentionBuilder
    from fast_transformers.feature_maps.fourier_features import (
        RandomFourierFeatures, SmoothedRandomFourierFeatures, Favor, GeneralizedRandomFeatures
    )
    feature_maps = {
        'random_fourier_features': RandomFourierFeatures,
        'smoothed_random_fourier_features': SmoothedRandomFourierFeatures,
        'favor': Favor,
        'generalized_random_features': GeneralizedRandomFeatures,
    }
except ImportError:
    pass


class FastSelfAttn(BertSelfAttention):
    def __init__(self, config, attn_config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.attn_config = attn_config
        builder_args = {
            'attention_dropout': config.hidden_dropout_prob,
            'query_dimensions': config.hidden_size // config.num_attention_heads,
        }
        if 'feature_map' in attn_config:
            constructor = feature_maps[attn_config['feature_map']]
            feature_args = attn_config.get('feature_args', dict())
            if 'FAVOR_DIM' in os.environ:
                feature_args['n_dims'] = int(os.environ['FAVOR_DIM'])
            builder_args['feature_map'] = constructor.factory(**feature_args)

        attn_builder = AttentionBuilder.from_kwargs(
            **builder_args,
            **{k: v for k, v in attn_config.items() if k not in ['type', 'feature_args', 'feature_map']},
        )
        self.attn_fn = attn_builder.get(self.attn_config['type'])

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        assert self.position_embedding_type == 'absolute'

        self.is_decoder = config.is_decoder

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        context_layer = self.big_multiplication(query_layer, key_layer, value_layer, attention_mask)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        assert not output_attentions
        # outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        outputs = (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

    def big_multiplication(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: torch.Tensor,
    ) -> torch.Tensor:
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        lengths = (mask > -1.0).squeeze(2).squeeze(1).sum(1)
        length_mask = LengthMask(lengths, device=query.device)
        attn_mask = FullMask(query.shape[1], device=query.device)
        out = self.attn_fn(query, key, value, attn_mask, length_mask, length_mask)
        out = out.transpose(1, 2)
        return out
