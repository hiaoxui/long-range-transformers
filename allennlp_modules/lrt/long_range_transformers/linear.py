from typing import *

import torch
from transformers.models.bert.modeling_bert import BertModel
from transformers import AutoModel, AutoTokenizer


class LinearTransformer(torch.nn.Module):
    def __init__(self, bert: BertModel):
        super().__init__()
        from fast_transformers.builders import TransformerEncoderBuilder
        builder = TransformerEncoderBuilder.from_kwargs(
            n_layers=bert.config.num_hidden_layers,
            n_heads=bert.config.num_attention_heads,
            query_dimensions=64,
            value_dimensions=64,
            feed_forward_dimensions=3072,
            attention_type='linear',
            activation=bert.config.hidden_act,
            final_normalization=False,
        )
        self.fast_model = builder.get()
        self.parameter_map(bert, self.fast_model)
        self.embeddings = bert.embeddings

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        batch_size, seq_length = input_ids.shape
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros((batch_size, seq_length), dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            past_key_values_length=0,
        )
        from fast_transformers.masking import LengthMask
        lengths = LengthMask(attention_mask.sum(dim=1))
        attention_output = self.fast_model(embedding_output, length_mask=lengths)
        return attention_output

    @staticmethod
    def linear_map(from_module: torch.nn.Linear, to_module: torch.nn.Linear):
        to_module.weight.data[:] = from_module.weight.data.clone().detach()
        to_module.bias.data[:] = from_module.bias.data.clone().detach()

    @staticmethod
    def parameter_map(bert: BertModel, fast):
        for i_layer in range(bert.config.num_hidden_layers):
            fl = fast.layers[i_layer]
            bl = bert.encoder.layer[i_layer]
            LinearTransformer.linear_map(bl.attention.self.query, fl.attention.query_projection)
            LinearTransformer.linear_map(bl.attention.self.key, fl.attention.key_projection)
            LinearTransformer.linear_map(bl.attention.self.value, fl.attention.value_projection)
            LinearTransformer.linear_map(bl.attention.output.dense, fl.attention.out_projection)
            LinearTransformer.linear_map(bl.attention.output.dense, fl.attention.out_projection)
            LinearTransformer.linear_map(bl.attention.output.dense, fl.attention.out_projection)
            LinearTransformer.linear_map(bl.intermediate.dense, fl.linear1)
            LinearTransformer.linear_map(bl.output.dense, fl.linear2)
            LinearTransformer.linear_map(bl.attention.output.LayerNorm, fl.norm1)
            LinearTransformer.linear_map(bl.output.LayerNorm, fl.norm2)


def test():
    bert = AutoModel.from_pretrained('bert-base-cased')
    tok = AutoTokenizer.from_pretrained('bert-base-cased')
    li = LinearTransformer(bert)
    inputs = tok('How do you do?', return_tensors='pt')
    li.eval()
    bert.eval()
    choice = input()
    if choice in ['linear', '']:
        linear_out = li(**inputs)
    if choice in ['bert', '']:
        bert_out = bert(**inputs)
    x = 1


if __name__ == '__main__':
    test()
