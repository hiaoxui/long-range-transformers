import logging
from typing import *
import json
import os
import random

import torch
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer, Token
from allennlp.data.instance import Instance, Field
from allennlp.data.fields import *

logger = logging.getLogger('docnli')


@DatasetReader.register('docnli')
class DocNLIReader(DatasetReader):
    def __init__(
            self,
            pretrained_model: str,
            max_length: Optional[int] = None,
            debug: bool = False,
            combine_input_fields: Optional[bool] = True,
            max_total_length: Optional[int] = None,
            split_size: Optional[int] = None,
            **extras
    ) -> None:
        super().__init__(**extras)
        self._token_indexers = {
            'tokens': PretrainedTransformerIndexer(pretrained_model, namespace='tokens')
        }
        self.tokenizer = PretrainedTransformerTokenizer(
            pretrained_model, max_length=max_length, add_special_tokens=False
        )
        self.pretrained_model_name = pretrained_model
        self.combine_input_fields = combine_input_fields
        self.debug = debug
        self.max_total_length = max_total_length
        self.split_size = split_size
        self.cur_idx = 0

    def _read(self, file_path) -> Iterable[Instance]:
        data = json.load(open(file_path))
        random.shuffle(data)
        if self.split_size is None or 'test.json' in file_path or 'EVAL' in os.environ:
            for idx, line in enumerate(data):
                yield self.text_to_instance(**line, idx=idx)
        else:
            yielded = 0
            while yielded < self.split_size:
                yield self.text_to_instance(**data[self.cur_idx % len(data)], idx=self.cur_idx)
                self.cur_idx += 1
                yielded += 1

    def text_to_instance(self, premise, hypothesis, label, idx) -> Instance:
        fields: Dict[str, Field] = dict()
        p_tokens, h_tokens = self.tokenizer.tokenize(premise), self.tokenizer.tokenize(hypothesis)
        meta = {
            'idx': idx,
            'label': label,
            'len_premise': len(premise.split(' ')),
            'len_hypothesis': len(hypothesis.split(' ')),
        }
        if self.combine_input_fields:
            if self.max_total_length is not None:
                p_len = max(0, self.max_total_length - len(h_tokens) - 2)
                h_tokens = h_tokens[:self.max_total_length-2]
                p_tokens = p_tokens[:p_len]
            tokens = self.tokenizer.add_special_tokens(p_tokens, h_tokens)
            fields["tokens"] = TextField(tokens)
        else:
            premise_tokens = self._tokenizer.add_special_tokens(premise)
            hypothesis_tokens = self._tokenizer.add_special_tokens(hypothesis)
            fields["premise"] = TextField(premise_tokens)
            fields["hypothesis"] = TextField(hypothesis_tokens)
            metadata = {
                "premise_tokens": [x.text for x in premise_tokens],
                "hypothesis_tokens": [x.text for x in hypothesis_tokens],
            }
            fields["metadata"] = MetadataField(metadata)

        if label is not None:
            label_idx = 1 if label == 'entailment' else 0
            fields["label"] = ArrayField(torch.tensor(label_idx), dtype=torch.int64)

        fields['meta'] = MetadataField(meta)

        return Instance(fields)

    def apply_token_indexers(self, instance: Instance) -> Instance:
        if "tokens" in instance.fields:
            instance.fields["tokens"]._token_indexers = self._token_indexers
        else:
            instance.fields["premise"]._token_indexers = self._token_indexers
            instance.fields["hypothesis"]._token_indexers = self._token_indexers
