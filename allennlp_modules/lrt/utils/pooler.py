from typing import *

import torch


def first_pooler(embeddings: torch.Tensor):
    return embeddings[:, 0, :]


def last_pooler(embeddings: torch.Tensor):
    return embeddings[:, -1, :]


def mean_pooler(embeddings: torch.Tensor):
    return embeddings.mean(1)
