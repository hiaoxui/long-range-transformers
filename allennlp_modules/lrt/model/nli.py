import logging
from typing import *

import torch
from torch import nn
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.common.from_params import Params, T, pop_and_construct_arg
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import TextFieldEmbedder
from allennlp.nn.activations import LinearActivation
from allennlp.training.metrics import BooleanAccuracy, F1Measure

from ..utils import mean_pooler, first_pooler, last_pooler

logger = logging.getLogger('nli')


@Model.register("nli")
class DocNLI(Model):
    @classmethod
    def from_params(
            cls: Type[T],
            params: Params,
            constructor_to_call: Callable[..., T] = None,
            constructor_to_inspect: Union[Callable[..., T], Callable[[T], None]] = None,
            **extras,
    ) -> T:
        text_field_embedder: TextFieldEmbedder = pop_and_construct_arg(
            'DocNLI', 'text_field_embedder', TextFieldEmbedder, None, params, **extras
        )
        dense: FeedForward = pop_and_construct_arg(
            'DocNLI', 'dense', FeedForward, None, params, **extras, input_dim=text_field_embedder.get_output_dim()
        )
        classifier = FeedForward(dense.get_output_dim(), 1, 1, LinearActivation())
        return super().from_params(
            params,
            text_field_embedder=text_field_embedder,
            dense=dense,
            classifier=classifier,
            **extras
        )

    def __init__(
            self,
            vocab: Vocabulary,
            text_field_embedder: TextFieldEmbedder,
            dense: FeedForward,
            classifier: FeedForward,
            **kwargs
    ) -> None:
        super().__init__(vocab, **kwargs)
        self.text_field_embedder = text_field_embedder
        self.dense = dense
        self.classifier = classifier
        if self.text_field_embedder.token_embedder_tokens.model_type == 'big_bird':
            self.pooler = first_pooler
        else:
            self.pooler = getattr(text_field_embedder.token_embedder_tokens.transformer_model, 'pooler', None)
        if self.pooler is None:
            if self.text_field_embedder.token_embedder_tokens.model_type == 'xlnet':
                self.pooler = last_pooler
            else:
                self.pooling = first_pooler
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.acc_metric = BooleanAccuracy()
        self.f1_metric = F1Measure(1)

    def forward(
            self,
            tokens: TextFieldTensors,
            label: torch.Tensor,
            meta: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, document_length, embedding_size)
        token_embeds = self.text_field_embedder(tokens)

        # shape [#batch, dim]
        pooling_features = self.pooler(token_embeds)
        dense_features = self.dense(pooling_features)
        logits = self.classifier(dense_features).squeeze(1)
        loss = self.loss_fn(logits, label.float())
        predictions = (logits > 0.).to(torch.int64)
        n_batch = predictions.shape[0]
        cat_predictions = predictions.new_zeros([n_batch, 2], dtype=torch.int64)
        cat_predictions.scatter_(1, predictions.unsqueeze(1).expand_as(cat_predictions), 1)
        self.acc_metric(predictions, label)
        self.f1_metric(cat_predictions, label)
        return {
            "loss": loss,
            "predictions": predictions,
            "meta": meta,
        }

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {'acc': self.acc_metric.get_metric(reset) * 100}
        if reset:
            metrics.update({k: v*100 for k, v in self.f1_metric.get_metric(reset).items()})
        return metrics
