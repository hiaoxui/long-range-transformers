import json
import os
import pickle

import torch
from transformers import AutoModel
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.data import Vocabulary
from allennlp.predictors import Predictor
from allennlp.nn import util as nn_util
import tqdm
from matplotlib import pyplot as plt
import numpy as np

import lrt

from lrt.tools.plot.util import Plotter


def load_data(model_name, max_ins, segment):
    min_len, max_len = 1024 - 100, 1024
    reader = lrt.DocNLIReader(model_name, None)
    test = json.load(open('/home/hiaoxui/data/docnli/test.json'))
    for line in test:
        if len(line['premise'].split(' ')) + len(line['hypothesis'].split(' ')) not in range(800, 1024):
            continue
        ins = reader.text_to_instance(line['premise'], line['hypothesis'], line['label'], segment)
        reader.apply_token_indexers(ins)
        batch_ins = list(
            SimpleDataLoader([ins], 1, vocab=Vocabulary())
        )[0]
        len_ins = batch_ins['tokens']['tokens']['token_ids'].shape[1]
        if min_len < len_ins < max_len:
            batch_ins = nn_util.move_to_device(batch_ins, device=0)
            yield batch_ins
            max_ins -= 1
        if max_ins == 0:
            return


def get_dist(grads):
    dists = []
    for grad in grads:
        grad = grad.abs().sum(1)
        dists.append(grad / grad.sum())
    dist = sum(dists)
    dist = dist / dist.sum()
    return dist


def smooth(xs, ys, bucket):
    le = len(xs)
    effective_length = (len(xs) // bucket) * bucket
    xs, ys = xs[:effective_length], ys[:effective_length]
    xs = xs.reshape(-1, bucket)[:, 0]
    ys = ys.reshape(-1, bucket).mean(1)
    return [xs, ys]


def draw_helper(all_pts, out):
    plt.style.use('classic')
    fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))
    plotter = Plotter(fig, ax)
    ylim = -1
    for xs, ys, label in all_pts:
        plotter(xs, ys*100, label, no_marker=True)
        ylim = max(ys.max()*100, ylim)
    ax.set_yticks(torch.arange(0.2, ylim, 0.2))
    ax.legend(prop={'size': 10}, frameon=False, loc=0)
    ax.set_ylabel('Gradient Percentage')
    ax.set_xlabel('Doc Length')
    plotter.save_fig(out)


def draw(all_grads, output_folder):
    le = 1000
    bucket = 4
    all_pts, all_pts_smooth = [], []
    for seg, grads in all_grads.items():
        dist = get_dist(grads)
        dist = dist[:le]
        all_pts.append([range(le), dist, f'XLNet ($L$={seg})'])
        smoothed = smooth(torch.arange(le), dist, bucket)
        all_pts_smooth.append(smoothed + [f'XLNet ($L$={seg})'])
    # draw_helper(all_pts, '/home/hiaoxui/.cache/python/nli/grad.pdf')
    draw_helper(all_pts_smooth, os.path.join(output_folder, f'grad.{bucket}.pdf'))


def run():
    data_root = 'figs/data/xlnet'
    output_folder = 'figs/outputs'
    xlnet_att(data_root, output_folder)


def xlnet_att(data_root, output_folder):
    n_ex = 128
    model_name = 'xlnet-base-cased'
    all_grads = dict()
    for seg in [128, 256, 512]:
        cache_path = os.path.join(data_root, f'grad.{seg}.json')
        if os.path.exists(cache_path):
            grads = pickle.load(open(cache_path, 'rb'))
        else:
            data = list(load_data(model_name, n_ex, seg))
            predictor = Predictor.from_path(f'/home/hiaoxui/.cache/python/nli/xl-{seg}-{seg}', cuda_device=-1)
            model = predictor._model
            lre = model.text_field_embedder.token_embedder_tokens
            model.text_field_embedder.token_embedder_tokens.reserve_emb = True
            lre.transformer_model.cache_mem = get_cache_mem(lre.transformer_model)
            n_seg = 1024 // seg
            grads = []
            for ins in tqdm.tqdm(data):
                model.train()
                ret = model(**ins)
                ret['loss'].backward()
                grad = torch.cat([emb.grad.detach().cpu().squeeze(0) for emb in lre.embs[:n_seg]], 0)
                model.zero_grad()
                if grad.shape[0] != 1024:
                    continue
                grads.append(grad)
            pickle.dump(grads, open(cache_path, 'wb'))
        all_grads[seg] = grads
    draw(all_grads, output_folder)


def get_cache_mem(xl):
    def cache_mem(curr_out, prev_mem):
        # cache hidden states into memory.
        if xl.reuse_len is not None and xl.reuse_len > 0:
            curr_out = curr_out[: xl.reuse_len]

        if xl.mem_len is None or xl.mem_len == 0:
            cutoff = 0
        else:
            cutoff = -xl.mem_len
        if prev_mem is None:
            new_mem = curr_out[cutoff:]
        else:
            new_mem = torch.cat([prev_mem, curr_out], dim=0)[cutoff:]
        return new_mem
    return cache_mem


if __name__ == '__main__':
    run()
