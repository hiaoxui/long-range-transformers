"""
Node by Guanghui:
This file is modified from the AllenNLP modules (v2.8.0)
This module can only be used for xlnet, and only works when batch_size=1 (no mask involved)

3 new parameters for init:
I did very few modifications. With two init parameters added, the module would segment
the sequences into `segment_length`.
Also, the model would store a `mem_length` of token memories for future use.
"""

import logging
from typing import *

import torch
from torch.nn import LSTM
from allennlp.modules.token_embedders import (
    PretrainedTransformerEmbedder, PretrainedTransformerMismatchedEmbedder, TokenEmbedder
)
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.modules.scalar_mix import ScalarMix
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.nn.util import batched_index_select
from transformers import AutoConfig

from .fast_model import FastBert

logger = logging.getLogger('lre')


@TokenEmbedder.register("lre")
class LongRangeEmbedder(PretrainedTransformerEmbedder):
    def __init__(
            self,
            model_name: str,
            max_length: int = None,
            sub_module: str = None,
            train_parameters: bool = True,
            eval_mode: bool = False,
            last_layer_only: bool = True,
            override_weights_file: Optional[str] = None,
            override_weights_strip_prefix: Optional[str] = None,
            load_weights: bool = True,
            gradient_checkpointing: Optional[bool] = None,
            tokenizer_kwargs: Optional[Dict[str, Any]] = None,
            transformer_kwargs: Optional[Dict[str, Any]] = None,

            attn_args: Optional[dict] = None,
            mem_length: int = 4096,
            use_global: bool = False,
            first_global: bool = False,
            # Following configs are for special pooling
            overlapping: bool = False,
            lstm: bool = False,
    ) -> None:
        """

        :param first_global:
         If set, then the first token will be treated as global for longformer.
        """
        super(PretrainedTransformerEmbedder, self).__init__()
        self.mem_length, self.use_global = mem_length, use_global
        if self.mem_length == 0:
            logger.warning('Mem length is set as zero. Will ignore mems.')
        self.first_global = first_global

        if attn_args is None or attn_args['type'] == 'huggingface':
            from allennlp.common import cached_transformers

            self.transformer_model = cached_transformers.get(
                model_name,
                True,
                override_weights_file=override_weights_file,
                override_weights_strip_prefix=override_weights_strip_prefix,
                load_weights=load_weights,
                **(transformer_kwargs or {}),
            )
        else:
            if load_weights:
                self.transformer_model = FastBert.from_pretrained(model_name, attn_args=attn_args)
            else:
                config = AutoConfig.from_pretrained(model_name)
                self.transformer_model = FastBert(config, attn_args=attn_args)

        if gradient_checkpointing is not None:
            self.transformer_model.config.update({"gradient_checkpointing": gradient_checkpointing})

        self.config = self.transformer_model.config

        if self.model_type == 'longformer':
            aw = max(self.config.attention_window)
            if max_length is not None and aw > max_length*2:
                self.config.attention_window = [max_length*2] * len(self.config.attention_window)
                for layer in self.transformer_model.encoder.layer:
                    layer.attention.self.one_sided_attn_window_size = max_length

        if sub_module:
            assert hasattr(self.transformer_model, sub_module)
            self.transformer_model = getattr(self.transformer_model, sub_module)
        self._max_length = max_length

        # I'm not sure if this works for all models; open an issue on github if you find a case
        # where it doesn't work.
        self.output_dim = self.config.hidden_size

        self._scalar_mix: Optional[ScalarMix] = None
        if not last_layer_only:
            self._scalar_mix = ScalarMix(self.config.num_hidden_layers)
            self.config.output_hidden_states = True

        tokenizer = PretrainedTransformerTokenizer(
            model_name,
            tokenizer_kwargs=tokenizer_kwargs,
        )

        try:
            if self.transformer_model.get_input_embeddings().num_embeddings != len(
                    tokenizer.tokenizer
            ):
                self.transformer_model.resize_token_embeddings(len(tokenizer.tokenizer))
        except NotImplementedError:
            # Can't resize for transformers models that don't implement base_model.get_input_embeddings()
            logger.warning(
                "Could not resize the token embedding matrix of the transformer model. "
                "This model does not support resizing."
            )

        self._num_added_start_tokens = len(tokenizer.single_sequence_start_tokens)
        self._num_added_end_tokens = len(tokenizer.single_sequence_end_tokens)
        self._num_added_tokens = self._num_added_start_tokens + self._num_added_end_tokens

        self.train_parameters = train_parameters
        if not train_parameters:
            for param in self.transformer_model.parameters():
                param.requires_grad = False

        self.eval_mode = eval_mode
        if eval_mode:
            self.transformer_model.eval()
        self.reserve_emb = False

        self.overlapping = overlapping
        self.lstm = None
        if lstm:
            self.lstm = LSTM(self.output_dim, self.output_dim//2, 1, batch_first=True, bidirectional=True)

    @property
    def model_type(self):
        return self.transformer_model.config.model_type

    def forward(
            self,
            token_ids: torch.Tensor,
            mask: torch.Tensor,
            type_ids: Optional[torch.Tensor] = None,
            segment_concat_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if type_ids is not None:
            max_type_id = type_ids.max()
            if max_type_id == 0:
                type_ids = None
            else:
                if max_type_id >= self._number_of_token_type_embeddings():
                    raise ValueError("Found type ids too large for the chosen transformer model.")
                assert token_ids.shape == type_ids.shape

        batch_size, n_token = token_ids.shape[:2]
        fold_long_sequences = self._max_length is not None and token_ids.size(1) > self._max_length
        if fold_long_sequences:
            if segment_concat_mask is None:
                token_ids, segment_concat_mask = self.guess_segment(token_ids, mask)
                if type_ids is not None:
                    type_ids, _ = self.guess_segment(type_ids, mask)
            batch_size, num_segment_concat_wordpieces = token_ids.size()
            token_ids, segment_concat_mask, type_ids = self._fold_long_sequences(
                token_ids, segment_concat_mask, type_ids
            )
        if self.overlapping:
            n_segment, token_ids = self.overlap(token_ids)
            _, mask = self.overlap(mask)
            if segment_concat_mask is not None:
                _, segment_concat_mask = self.overlap(segment_concat_mask)

        attention_mask = segment_concat_mask if segment_concat_mask is not None else mask
        parameters = {"input_ids": token_ids, "attention_mask": attention_mask.float()}
        if type_ids is not None:
            parameters["token_type_ids"] = type_ids

        self.embs = []
        if self.model_type == 'xlnet' and self.mem_length > 0:
            mems = None
            last_hidden_states = list()
            parameters = {
                k: v.reshape(batch_size, -1, token_ids.shape[1]) for k, v in parameters.items()
            }
            n_segment = parameters['input_ids'].shape[1]
            for i_segment in range(n_segment):
                local_param = {k: v[:, i_segment] for k, v in parameters.items()}
                if self.reserve_emb:
                    emb = self.transformer_model.word_embedding(local_param.pop('input_ids'))
                    emb.retain_grad()
                    self.embs.append(emb)
                    local_param['inputs_embeds'] = emb
                transformer_output = self.transformer_model(**local_param, mems=mems, use_mems=True)
                last_hidden_states.append(transformer_output['last_hidden_state'])
                # mems shape: #layer x [segment_length, #batch, dim]
                mems = transformer_output['mems']
                mems = None if self.mem_length == 0 else [m[-self.mem_length:, :, :] for m in mems]
            embeddings = torch.stack(last_hidden_states).reshape(*token_ids.shape, -1)
        else:
            if self.model_type == 'longformer':
                if self.use_global:
                    parameters['global_attention_mask'] = token_ids.new_ones(size=token_ids.shape)
                elif self.first_global:
                    global_attention_mask = token_ids.new_zeros(size=token_ids.shape)
                    global_attention_mask[:, 0] = 1
                    parameters['global_attention_mask'] = global_attention_mask
            transformer_output = self.transformer_model(**parameters)
            embeddings = transformer_output['last_hidden_state']

        if fold_long_sequences:
            embeddings = self._unfold_long_sequences(
                embeddings, segment_concat_mask, batch_size, num_segment_concat_wordpieces
            )
        if self.overlapping and n_segment > 1:
            embeddings = self.de_overlap(embeddings.reshape(batch_size, n_segment, *embeddings.shape[1:]))
            embeddings = embeddings[:, :n_token]
        if self.lstm is not None:
            embeddings = self.lstm(embeddings)[0]
        return embeddings

    def guess_segment(self, token_ids: torch.Tensor, mask: torch.Tensor):
        if self.model_type == 'xlnet':
            return self.guess_xlnet(token_ids, mask)
        first_batch_length = mask.sum(1)[0]
        real_token_num = (mask.sum(1) - 2).tolist()
        first_token, last_token = token_ids[0][0].item(), token_ids[0][first_batch_length-1].item()
        batch_size, num_tokens = token_ids.shape
        no_special_token_ids = token_ids[:, 1:-1]
        to_pad = (self._max_length - 2) - (num_tokens-2) % (self._max_length-2)
        if to_pad == self._max_length - 2:
            to_pad = 0
        padded_token_ids = torch.nn.functional.pad(no_special_token_ids, [0, to_pad], value=0)
        padded_token_ids = padded_token_ids.reshape(batch_size, -1, self._max_length-2)
        num_segments = padded_token_ids.shape[1]
        ret_token_ids = torch.cat([
            token_ids.new_full([batch_size, num_segments, 1], fill_value=first_token),
            padded_token_ids,
            token_ids.new_full([batch_size, num_segments, 1], fill_value=last_token),
        ], dim=2).reshape(batch_size, -1)
        lengths = list()
        for i, rtn in enumerate(real_token_num):
            batch_segment = int(rtn//(self._max_length-2) + (rtn % (self._max_length-2) != 0))
            batch_length = batch_segment * 2 + rtn
            if batch_length != ret_token_ids.shape[1]:
                ret_token_ids[i, batch_length:] = 1
                ret_token_ids[i, batch_length-1] = last_token
            lengths.append(batch_length)
        max_length = max(lengths)
        ret_token_ids = ret_token_ids[:, :max_length]
        lengths = torch.tensor(lengths, dtype=torch.int64, device=token_ids.device)
        segment_mask = (torch.arange(0, max_length, device=token_ids.device).unsqueeze(0).expand_as(ret_token_ids).T < lengths).T
        return ret_token_ids, segment_mask

    def guess_xlnet(self, token_ids: torch.Tensor, mask: torch.Tensor):
        first_batch_length = mask.sum(1)[0]
        real_token_num = (mask.sum(1) - 2).tolist()
        first_token, last_token = token_ids[0][first_batch_length-2].item(), token_ids[0][first_batch_length-1].item()
        batch_size, num_tokens = token_ids.shape
        no_special_token_ids = token_ids[:, :-2]
        to_pad = (self._max_length - 2) - (num_tokens-2) % (self._max_length-2)
        if to_pad == self._max_length - 2:
            to_pad = 0
        padded_token_ids = torch.nn.functional.pad(no_special_token_ids, [0, to_pad], value=0)
        padded_token_ids = padded_token_ids.reshape(batch_size, -1, self._max_length-2)
        num_segments = padded_token_ids.shape[1]
        ret_token_ids = torch.cat([
            padded_token_ids,
            token_ids.new_full([batch_size, num_segments, 1], fill_value=first_token),
            token_ids.new_full([batch_size, num_segments, 1], fill_value=last_token),
        ], dim=2).reshape(batch_size, -1)
        lengths = list()
        for i, rtn in enumerate(real_token_num):
            batch_segment = int(rtn//(self._max_length-2) + (rtn % (self._max_length-2) != 0))
            batch_length = batch_segment * 2 + rtn
            if batch_length != ret_token_ids.shape[1]:
                ret_token_ids[i, batch_length:] = 0
                ret_token_ids[i, batch_length-2] = first_token
                ret_token_ids[i, batch_length-1] = last_token
            lengths.append(batch_length)
        max_length = max(lengths)
        ret_token_ids = ret_token_ids[:, :max_length]
        lengths = torch.tensor(lengths, dtype=torch.int64, device=token_ids.device)
        segment_mask = (torch.arange(0, max_length, device=token_ids.device).unsqueeze(0).expand_as(ret_token_ids).T < lengths).T
        return ret_token_ids, segment_mask

    @staticmethod
    def overlap(tensor: torch.Tensor):
        batch_size, n_token = tensor.shape
        if n_token <= 512:
            return 1, tensor
        if n_token % 256 != 0:
            pad_length = (n_token // 256 + 1) * 256 - n_token
            padding = tensor.new_zeros([batch_size, pad_length], dtype=torch.int64)
            tensor = torch.cat([tensor, padding], dim=1)
            n_token = tensor.shape[1]
        n_segment = max(n_token // 256 - 1, 1)
        ret = list()
        for i in range(n_segment):
            ret.append(tensor[:, i*256: i*256+512])
        # Shape [batch, segment, 512]
        ret = torch.stack(ret, dim=1)
        ret = ret.reshape(-1, 512)
        return n_segment, ret

    @staticmethod
    def de_overlap(embeds: torch.Tensor) -> torch.Tensor:
        ret = list()
        n_batch, n_seg = embeds.shape[:2]
        ret.append(embeds[:, 0, :256])
        for i in range(n_seg-1):
            ret.append(embeds[:, i, 256:] + embeds[:, i+1, :256])
        ret.append(embeds[:, -1, 256:])
        ret = torch.cat(ret, dim=1)
        return ret


@TokenEmbedder.register("lre_mismatched")
class LongRangeMismatchedEmbedder(PretrainedTransformerMismatchedEmbedder):
    def __init__(
            self,
            model_name: str,
            mem_length: int = 0,
            use_global: bool = False,
            max_length: int = None,
            train_parameters: bool = True,
            last_layer_only: bool = True,
            override_weights_file: Optional[str] = None,
            override_weights_strip_prefix: Optional[str] = None,
            load_weights: bool = True,
            gradient_checkpointing: Optional[bool] = None,
            tokenizer_kwargs: Optional[Dict[str, Any]] = None,
            transformer_kwargs: Optional[Dict[str, Any]] = None,
            sub_token_mode: Optional[str] = "avg",
            attn_args: Optional[dict] = None,
            overlapping: bool = False,
            lstm: bool = False,
    ) -> None:
        super(TokenEmbedder, self).__init__()
        # The matched version v.s. mismatched
        self._matched_embedder = LongRangeEmbedder(
            model_name,
            max_length=max_length,
            train_parameters=train_parameters,
            last_layer_only=last_layer_only,
            override_weights_file=override_weights_file,
            override_weights_strip_prefix=override_weights_strip_prefix,
            load_weights=load_weights,
            gradient_checkpointing=gradient_checkpointing,
            tokenizer_kwargs=tokenizer_kwargs,
            transformer_kwargs=transformer_kwargs,
            mem_length=mem_length,
            use_global=use_global,
            attn_args=attn_args,
            lstm=lstm,
            overlapping=overlapping,
        )
        self.sub_token_mode = sub_token_mode
