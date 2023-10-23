from typing import *
from argparse import ArgumentParser
import json
from collections import defaultdict
import os

import torch
from allennlp.predictors import Predictor
from allennlp.data.vocabulary import Vocabulary
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from lrt.tools.gaussian_kernel_regression import GaussianKernelRegression

plt.style.use('classic')


def normalize(mat):
    dim_sum = mat.sum(axis=-1).reshape(*mat.shape[:-1], 1)
    dim_sum[dim_sum == 0.] = 1.0
    dist = mat / dim_sum
    return dist


def draw(pct, name, ax, method):
    if method == 'gkr':
        distances, weights = list(zip(*pct))
        gkr = GaussianKernelRegression(distances, weights)
        length = 1024
        ys = gkr.predict(np.arange(1, length+1), 10) / 1e-3
        ax.plot(np.arange(1, length+1), ys, label=name)
    else:
        chunks = 9
        upper = [2**i for i in range(3, chunks+2)] + [float('inf')]
        lower = [1] + upper[:-1]
        weights = [[] for _ in range(chunks)]
        for d, w in pct:
            assert d >= 1
            i_chunk = (d >= np.array(lower)).sum() - 1
            weights[i_chunk].append(w)
        weights = np.array([np.mean(we) if len(we) > 0 else 0. for we in weights])
        weights /= 1e-3
        ax.plot(range(len(weights)), weights, label=name)
        ax.set_xticks(range(len(lower)), lower)


def calculate_pct(saver, attn, cluster):
    for i in range(len(cluster)):
        for j in range(len(cluster)):
            if i == j:
                continue
            p1 = cluster[i]
            p2 = cluster[j]
            weights = attn[p1[0]:p1[1]+1, p2[0]:p2[1]+1]
            w = float(weights.sum(axis=1).mean())
            saver.append([abs(p1[0]-p2[0]), w])


def inspect(data):
    attn_pct_prod, attn_pct_mean = list(), list()
    for attn, clusters in tqdm(data, desc='running'):
        # attn shape [#layer, #head, #token (from), #token (to)]
        attn = attn.mean(dim=1) # Avg over all heads
        prod_attn = torch.eye(attn.shape[1])
        for a_ in attn.unbind(0):
            prod_attn @= a_
        mean_attn = attn.mean(dim=0)
        prod_attn, mean_attn = normalize(prod_attn.numpy()), normalize(mean_attn.numpy())
        for cluster in clusters:
            calculate_pct(attn_pct_prod, prod_attn, cluster)
            calculate_pct(attn_pct_mean, mean_attn, cluster)
    return {'prod': attn_pct_prod, 'mean': attn_pct_mean}


def longformer_attn_mat(rel_attn, seq_len):
    # rel_attn [#layer, #head, #token, 2w+1]
    # attn [#layer, #head, #token, #token]
    window_size = (rel_attn.shape[3]-1)//2
    attn = rel_attn.new_zeros(size=(rel_attn.shape[0], rel_attn.shape[1], seq_len, seq_len))
    for i in range(seq_len):
        attn_start, attn_end = max(0, i-window_size), min(seq_len, i+window_size+1)
        rel_start, rel_end = max(0, window_size-i), min(2*window_size+1, seq_len-i+window_size)
        attn[:, :, i, attn_start:attn_end] = rel_attn[:, :, i, rel_start:rel_end]
    return attn.contiguous()


def calculate_attn(
        archive_path, data_path, max_instances, cuda, segment=999999
) -> Iterable[Tuple[torch.Tensor, List[List[Tuple[int, int]]]]]:
    """
    This is a function for data and model loading.
    :param archive_path: The path to the archive.
    :param data_path:
    :param max_instances:
    :param cuda:
    :return:
    """
    predictor = Predictor.from_path(archive_path, cuda_device=cuda)
    if cuda < 0: cuda = None
    predictor._model.eval()
    embedder = predictor._model._text_field_embedder._token_embedders['tokens']._matched_embedder.transformer_model
    for idx, ins in enumerate(predictor._dataset_reader.read(data_path)):
        if idx == max_instances:
            break
        ins['text'].index(Vocabulary())
        indexed = ins['text']._indexed_tokens['tokens']
        inputs = {
            'input_ids': torch.tensor(indexed['token_ids'], device=cuda).unsqueeze(0),
            'token_type_ids': torch.tensor(indexed['type_ids'], device=cuda).unsqueeze(0),
        }
        length = inputs['input_ids'].shape[1]
        cur = 0
        all_attn = list()
        while cur < length:
            cur_inputs = {k: v[:, cur: cur+segment] for k, v in inputs.items()}
            with torch.no_grad():
                cur_attn = embedder(**cur_inputs, output_attentions=True)['attentions']
            # [#layer, #head, #token, 2w+1]
            cur_attn = torch.cat([item.cpu() for item in cur_attn], dim=0)
            all_attn.append(cur_attn)
            cur += segment
        attn = longformer_attn_mat(torch.cat(all_attn, dim=2), length)
        offsets = indexed['offsets']
        spans = [(span_.span_start, span_.span_end) for span_ in ins['spans'].field_list]
        wp_spans = [(offsets[s][0], offsets[e][1]) for (s, e) in spans]
        labels = ins['span_labels'].labels
        clusters = defaultdict(list)
        for span, lbl in zip(wp_spans, labels):
            if lbl != -1:
                clusters[lbl].append(span)
        yield attn, list(clusters.values())


def run():
    parser = ArgumentParser()
    parser.add_argument('-a', type=str, nargs='*', help='archive to load')
    parser.add_argument('-n', type=str, nargs='*', help='names')
    parser.add_argument('-s', type=int, nargs='*', help='segments')
    parser.add_argument('-o', type=str, help='output folder')
    parser.add_argument('-d', type=str, help='data path')
    parser.add_argument('-m', type=int, help='max_instances')
    parser.add_argument('-c', type=int, default=-1, help='cuda device')
    args = parser.parse_args()
    pct = dict()
    for archive_folder, name, n_segment in zip(args.a, args.n, args.s):
        fp = os.path.join(archive_folder, 'pct.json')
        if os.path.exists(fp):
            pct[name] = json.load(open(fp))
        else:
            data = calculate_attn(archive_folder, args.d, args.m, args.c, n_segment)
            cur_pct = inspect(data)
            json.dump(cur_pct, open(fp, 'w'))
            pct[name] = cur_pct

    os.makedirs(args.o, exist_ok=True)
    for sm_mtd in ['chunk', 'gkr']:
        for attn_mtd in ['mean', 'prod']:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            save_path = os.path.join(args.o, f'{attn_mtd}.{sm_mtd}.pdf')
            for name, cur_pct in pct.items():
                draw(cur_pct[attn_mtd], name, ax, sm_mtd)
            ax.set_ylabel('Aggregated Attn Weights / 1e-3')
            ax.legend()
            ax.set_xlabel('Mention Distances')
            fig.tight_layout()
            fig.savefig(save_path)


if __name__ == '__main__':
    run()
