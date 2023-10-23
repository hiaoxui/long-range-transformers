import logging
import collections
from typing import *

from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance

from allennlp_models.common.ontonotes import Ontonotes
from allennlp_models.coref.util import make_coref_instance
from allennlp_models.coref.dataset_readers.conll import ConllCorefReader

logger = logging.getLogger('conll')


@DatasetReader.register("conll")
class CustomConllCorefReader(ConllCorefReader):
    def __init__(
            self,
            max_tokens: int = 32768,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.max_tokens = max_tokens

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache

        ontonotes_reader = Ontonotes()
        for sentences in ontonotes_reader.dataset_document_iterator(file_path):
            clusters: DefaultDict[int, List[Tuple[int, int]]] = collections.defaultdict(list)

            total_tokens = 0
            max_sentences = None
            for i_sent, sentence in enumerate(sentences):
                for typed_span in sentence.coref_spans:
                    # Coref annotations are on a _per sentence_
                    # basis, so we need to adjust them to be relative
                    # to the length of the document.
                    span_id, (start, end) = typed_span
                    clusters[span_id].append((start + total_tokens, end + total_tokens))
                total_tokens += len(sentence.words)
                if total_tokens > self.max_tokens and max_sentences is None:
                    max_sentences = i_sent
            if self._max_sentences is not None and max_sentences is not None:
                max_sentences = min(self._max_sentences, max_sentences)

            yield self.text_to_instance([s.words for s in sentences], list(clusters.values()), max_sentences)

    @overrides
    def text_to_instance(
            self,  # type: ignore
            sentences: List[List[str]],
            gold_clusters: Optional[List[List[Tuple[int, int]]]] = None,
            max_sentences: Optional[int] = None
    ) -> Instance:
        ins = make_coref_instance(
            sentences,
            self._token_indexers,
            self._max_span_width,
            gold_clusters,
            self._wordpiece_modeling_tokenizer,
            max_sentences,
            self._remove_singleton_clusters,
        )
        return ins
