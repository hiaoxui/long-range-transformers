from typing import *
import random
import json
import os
from argparse import ArgumentParser

from allennlp.predictors import Predictor
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

import lrt
from lrt.tools.plot.util import full_name, Plotter


def b_cubed_helper(ac, bc, lower, upper):
    mention2b = dict()
    for clu in bc:
        for span in clu:
            mention2b[span] = clu
    ret = []
    for clu in ac:
        for span in clu:
            if span not in mention2b:
                ret.append(0)
            else:
                valid_numerator = len([
                    men for men in mention2b[span]
                    if lower <= abs(men[0]-span[0]) < upper and men in clu
                ])
                valid_denominator = len([
                    men for men in clu
                    if lower <= abs(men[0]-span[0]) < upper
                ])
                if valid_denominator == 0:
                    continue
                else:
                    ret.append(valid_numerator / valid_denominator)
    return ret


def b_cubed(outputs, lower=float('-inf'), upper=float('inf'), return_mean=True):
    p, r = list(), list()
    for out in outputs:
        gold = [tuple(tuple(span) for span in clu) for clu in out['gold']]
        pred = [tuple(tuple(span) for span in clu) for clu in out['pred']]
        p.extend(b_cubed_helper(pred, gold, lower, upper))
        r.extend(b_cubed_helper(gold, pred, lower, upper))
    if return_mean:
        precision = np.mean(p)
        recall = np.mean(r)
        if precision == 0 or recall == 0:
            return 0
        return 2/(1/precision + 1/recall)
    else:
        return p, r


def get_chunks(chunks=10):
    upper = [2**i for i in range(3, chunks+2)] + [float('inf')]
    lower = [1] + upper[:-1]
    return lower, upper


def bd_c2f(data_root, output_folder, exps):
    n_chunk = 8
    archives = [os.path.join(data_root, exp) for exp in exps]
    labels = [full_name(exp, with_length=True) for exp in exps]
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, 'b3_abs.pdf')
    cache_files = {label: os.path.join(cache_path, 'pred.json') for cache_path, label in zip(archives, labels)}
    data = dict()
    for idx, (label, cache_file_path) in enumerate(cache_files.items()):
        assert os.path.exists(cache_file_path), f'{cache_file_path} not exist.'
        data[label] = json.load(open(cache_file_path))
    lower, upper = get_chunks(n_chunk)
    bounds = list(zip(lower, upper))

    rst = dict()
    for (label, outputs) in data.items():
        f1 = list()
        for l, u in bounds:
            f1.append(b_cubed(outputs, l, u))
        f1 = 100 * np.array(f1)
        rst[label] = f1

    plt.style.use('classic')

    def plot_abs():
        fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))
        plotter = Plotter(fig, ax)
        for label, f1s in rst.items():
            plotter(range(len(f1s)), f1s, label)
        ax.legend(prop={'size': 10}, frameon=False, loc=0)
        ax.set_ylabel('B$^3$ F$_1$')
        ax.set_xlabel('Mention Distances')
        ax.set_xticks(range(len(lower)), lower)
        ax.set_yticks(range(0, 71, 10))
        # ax.set_yticks(range(-10, 31, 10), range(30, 71, 10))
        ax.set_ylim(35, 70)
        edge = 0.5
        ax.set_xlim(-edge, len(lower)-1+edge)
        plotter.save_fig(output_path)

    plot_abs()


def run():
    output_folder = 'figs/outputs'
    data_root = 'figs/data/c2f'
    exps = ['lf-128', 'lf-512', 'lf-4096', 'xl-512-512']
    bd_c2f(data_root, output_folder, exps)


if __name__ == '__main__':
    run()


def predict(archive_path, data_path, max_ins, cuda):
    predictor = Predictor.from_path(archive_path, cuda_device=cuda if cuda >=0 else None)
    instances = list(tqdm(predictor._dataset_reader.read(data_path), desc='reading data'))
    outputs = []
    for ins in tqdm(instances[:max_ins], desc='predicting...'):
        pred = predictor.predict_instance(ins)['clusters']
        gold = ins.fields['metadata'].metadata['clusters']
        pred = [[tuple(span) for span in clu] for clu in pred]
        outputs.append({
            'gold': gold,
            'pred': pred,
            'doc_len': len(ins.fields['metadata'].metadata['original_text'])
        })
    return outputs
