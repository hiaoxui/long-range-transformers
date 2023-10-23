from transformers import LEDPreTrainedModel, LEDConfig, LEDModel, BigBirdModel, AutoConfig, XLNetModel, LongformerModel
import torch.nn.functional as F

import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.led.modeling_led import LEDEncoderBaseModelOutput, LEDSeq2SeqModelOutput, LEDSeq2SeqLMOutput

import torch
import math


def fold_long_sequences(token_ids, mask, type_ids, max_length):
    num_segment_concat_wordpieces = token_ids.size(1)
    num_segments = math.ceil(num_segment_concat_wordpieces / max_length)  # type: ignore
    padded_length = num_segments * max_length  # type: ignore
    length_to_pad = padded_length - num_segment_concat_wordpieces

    def fold(tensor):
        tensor = F.pad(tensor, [0, length_to_pad], value=0)
        return tensor.reshape(-1, max_length)

    return fold(token_ids), fold(mask), fold(type_ids) if type_ids is not None else None


class BigBird(nn.Module):
    def __init__(self, model_name, use_context, segment_length, context_length=0):
        super().__init__()

        self.transformer = BigBirdModel.from_pretrained(model_name,
                                                        attention_type="block_sparse" if use_context else "original_full",
                                                        block_size=context_length,
                                                        gradient_checkpointing=True, add_pooling_layer=False)

        self.use_context = use_context
        self.segment_length = segment_length

        self.config = AutoConfig.from_pretrained(model_name)

    def get_output_embeddings(self):
        return self.transformer.get_output_embeddings()

    def forward(self, token_ids, mask, **kwargs):
        wordpiece_mask = mask.long()
        batch_size, length = token_ids.size()

        input_ids, attention_mask, _ = \
            fold_long_sequences(token_ids, wordpiece_mask, None, self.segment_length)

        embeddings = self.transformer(input_ids=input_ids,
                                      attention_mask=attention_mask).last_hidden_state

        dim = embeddings.size()[-1]
        embeddings = embeddings.reshape(batch_size, -1, dim)
        embeddings = embeddings[:, :length, :]

        if "return_dict" in kwargs and kwargs["return_dict"]:
            return BaseModelOutput(last_hidden_state=embeddings)

        return (embeddings,)


class XLNet(nn.Module):
    def __init__(self, model_name, use_context, segment_length=0, context_length=0):
        super().__init__()

        self.transformer = XLNetModel.from_pretrained(model_name, use_mems_train=use_context,
                                                      use_mems_eval=use_context, mem_len=context_length)

        self.use_context = use_context
        self.segment_length = segment_length

        self.config = AutoConfig.from_pretrained(model_name)

    def get_output_embeddings(self):
        return self.transformer.get_output_embeddings()

    def forward(self, token_ids, mask, type_ids=None, **kwargs):

        wordpiece_mask = mask.long()

        batch_size, length = token_ids.size()

        input_ids, attention_mask, token_type_ids = \
            fold_long_sequences(token_ids, wordpiece_mask, type_ids, 4096)  # self.segment_length * 2)
        input_ids = input_ids.reshape(-1, self.segment_length)
        attention_mask = attention_mask.reshape(-1, self.segment_length)

        if not self.use_context:
            model_out = self.transformer(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         token_type_ids=token_type_ids)

            embeddings = model_out.last_hidden_state

        else:
            embeddings = []
            mems = None
            input_ids = input_ids.reshape(batch_size, -1, self.segment_length)
            attention_mask = attention_mask.reshape(batch_size, -1, self.segment_length)

            if token_type_ids is not None:
                token_type_ids = token_type_ids.reshape(batch_size, -1, self.segment_length)

            num_segs = input_ids.size()[1]

            for segment_index in range(num_segs):
                parameters = {"input_ids": input_ids[:, segment_index, :],
                              "attention_mask": attention_mask[:, segment_index, :],
                              "mems": mems}
                if token_type_ids is not None:
                    parameters["token_type_ids"] = token_type_ids[:, segment_index, :]

                model_out = self.transformer(**parameters)
                last_hidden_state = model_out.last_hidden_state
                embeddings.append(last_hidden_state.unsqueeze(1))
                mems = model_out.mems

            embeddings = torch.cat(embeddings, 1)
            dim = embeddings.size()[-1]

            embeddings = embeddings.reshape(-1, self.segment_length, dim)

        dim = embeddings.size()[-1]
        embeddings = embeddings.reshape(batch_size, -1, dim)
        embeddings = embeddings[:, :length, :]

        if "return_dict" in kwargs and kwargs["return_dict"]:
            return BaseModelOutput(last_hidden_state=embeddings)

        return (embeddings,)


class Longformer(nn.Module):
    def __init__(self, model_name, use_context, segment_length, context_length=0):
        super().__init__()

        self.transformer = LongformerModel.from_pretrained(model_name,
                                                           attention_window=context_length,
                                                           gradient_checkpointing=True,
                                                           add_pooling_layer=False)

        self.use_context = use_context
        self.segment_length = segment_length

        self.config = AutoConfig.from_pretrained(model_name)

    def get_output_embeddings(self):
        return self.transformer.get_output_embeddings()

    def forward(self, token_ids, mask, **kwargs):
        wordpiece_mask = mask.long()
        batch_size, length = token_ids.size()

        input_ids, attention_mask, _ = \
            fold_long_sequences(token_ids, wordpiece_mask, None, self.segment_length)

        if self.use_context:
            global_attention_mask = None if "global_attention_mask" not in kwargs else kwargs["global_attention_mask"]
            if global_attention_mask is not None:
                global_attention_mask = torch.cat((global_attention_mask, torch.zeros(
                    (global_attention_mask.size()[0], input_ids.size()[1] - global_attention_mask.size()[1]),
                    dtype=torch.long, device=input_ids.device)), 1)
        else:
            global_attention_mask = None

        embeddings = self.transformer(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      global_attention_mask=global_attention_mask).last_hidden_state

        dim = embeddings.size()[-1]
        embeddings = embeddings.reshape(batch_size, -1, dim)
        embeddings = embeddings[:, :length, :]

        if "return_dict" in kwargs and kwargs["return_dict"]:
            return BaseModelOutput(last_hidden_state=embeddings)

        return (embeddings,)


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class _LEDModel(LEDModel):
    def __init__(self, config: LEDConfig):
        super().__init__(config)

        self.max_length = config.max_length
        print("-----------------{}------------------".format(self.max_length))

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            global_attention_mask=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Using this like Bart, as LED is derived from it. So far
        # No checkpoint on the hub exists that uses that in practice.
        # https://github.com/huggingface/transformers/blob/ac3cb660cad283163f7c73cad511124e845ca388/src/transformers/models/bart/modeling_bart.py#L1153
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        encoder_mask = attention_mask
        if encoder_outputs is None:
            batch_size, length = input_ids.size()

            if self.max_length == 1536:
                input_ids, attention_mask, _ = \
                    fold_long_sequences(input_ids, attention_mask, None, self.max_length)
                bsz, _ = input_ids.size()

                global_attention_mask = None

            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            if return_dict:
                embeddings = encoder_outputs.last_hidden_state
            else:
                embeddings = encoder_outputs[0]

            if self.max_length == 1536:
                dim = embeddings.size()[-1]
                embeddings = embeddings.reshape(batch_size, -1, dim)
                embeddings = embeddings[:, :length, :]

            if return_dict:
                encoder_outputs.last_hidden_state = embeddings
            else:
                encoder_outputs[0] = embeddings

        # If the user passed a tuple for encoder_outputs, we wrap it in a LEDEncoderBaseModelOutput when return_dict=False
        elif return_dict and not isinstance(encoder_outputs, LEDEncoderBaseModelOutput):
            encoder_outputs = LEDEncoderBaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                global_attentions=encoder_outputs[3] if len(encoder_outputs) > 3 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0] if isinstance(encoder_outputs,
                                                                   tuple) else encoder_outputs.last_hidden_state,
            encoder_attention_mask=encoder_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return LEDSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            encoder_global_attentions=encoder_outputs.global_attentions,
        )


class LEDForConditionalGeneration(LEDPreTrainedModel):
    base_model_prefix = "led"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        r"lm_head\.weight",
    ]

    def __init__(self, config: LEDConfig):
        super().__init__(config)
        self.led = _LEDModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.led.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.led.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.led.get_encoder()

    def get_decoder(self):
        return self.led.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            global_attention_mask=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.led(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return LEDSeq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            encoder_global_attentions=outputs.encoder_global_attentions,
        )

    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            past=None,
            attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


def get_enecoder_model(model_name, **kwargs):
    if "xlnet" in model_name:
        encoder = XLNet(model_name, **kwargs)
    elif "longformer" in model_name:
        encoder = Longformer(model_name, **kwargs)
    elif "bigbird" in model_name:
        encoder = BigBird(model_name, **kwargs)

    return encoder


def get_enecoder_decoder_model(max_length):
    return LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384",
                                                       max_length=max_length,
                                                       attention_window=[min(1536, max_length)] * 6,
                                                       attention_dropout=0.1,
                                                       gradient_checkpointing=True)