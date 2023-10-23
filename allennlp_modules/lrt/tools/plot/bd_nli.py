from argparse import ArgumentParser
import os
import json
import numpy as np
from matplotlib import pyplot as plt
from lrt.tools.plot.util import Plotter, full_name


class Metric:
    def __init__(self):
        self.tp = self.fp = self.tn = self.fn = 0
    def __call__(self, pred, label):
        if isinstance(label, str):
            label = 0 if label == 'not_entailment' else 1
        if pred == label == 0:
            self.tn += 1
        elif pred == label == 1:
            self.tp += 1
        elif pred == 0 and label == 1:
            self.fn += 1
        else:
            self.fp += 1
    @property
    def total(self):
        return self.tn + self.tp + self.fn + self.fp
    def metric(self):
        p = self.tp / (self.tp + self.fp) if (self.tp + self.fp > 0) else 0.
        r = self.tp / (self.tp + self.fn) if (self.tp + self.fn > 0) else 0.
        f = 2 / (1/p + 1/r) if p * r > 0. else 0.
        acc = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn) if (self.tp + self.tn + self.fp + self.fn) > 0 else 0.
        return p*100,r*100,f*100,acc*100


def get_chunks(chunks=10, start_bucket=3):
    upper = [2**i for i in range(start_bucket, chunks+start_bucket-1)] + [float('inf')]
    lower = [1] + upper[:-1]
    return lower, upper


def one_model(lower, preds):
    metrics = [Metric() for _ in lower]
    cnt = [0] * len(lower)
    lower = np.array(lower)
    max_le = 0
    for line in preds:
        pre = line['predictions']
        tru = int(line['meta']['label'] == 'entailment')
        le = line['meta']['len_premise'] + line['meta']['len_hypothesis']
        max_le = max(max_le, le)
        bucket = max(0, (le >= lower).sum()-1)
        metrics[bucket](pre, tru)
        cnt[bucket] += 1
    f1s = [me.metric()[2] for me in metrics]
    return f1s, cnt


def draw(rst, lower, output):
    plt.style.use('classic')
    fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))
    plotter = Plotter(fig, ax)
    # colors = ['teal', 'coral']
    for label, f1s in rst:
        plotter(range(len(f1s)), f1s, label)
    ax.legend(prop={'size': 10}, frameon=False, loc=0)
    ax.set_ylabel('F$_1$ Score')
    ax.set_xlabel('Document Length')
    ax.set_xticks(range(len(lower)), lower)
    # ax.set_yticks(range(0, 71, 10))
    ax.set_yticks(range(30, 71, 20))
    ax.set_ylim(28, 85)
    edge = 0.5
    ax.set_xlim(-edge, len(lower) - 1 + edge)
    plotter.save_fig(os.path.join(output, 'nli_bd.pdf'))


def bd_nli(data_root, output_folder, exps):
    archives = [os.path.join(data_root, ar) for ar in exps]
    # args = parser.parse_args()
    n_bucket = 6
    start_bucket = 6
    lower, upper = get_chunks(n_bucket, start_bucket)
    rst = list()
    for cache in archives:
        rst_cache_path = os.path.join(cache, f'f1s.{n_bucket}.{start_bucket}.json')
        if os.path.exists(rst_cache_path):
            f1s = json.load(open(rst_cache_path))
        else:
            pred = list(map(json.loads, open(os.path.join(cache, 'test_pred.jsonl')).read().splitlines()))
            f1s, cnt = one_model(lower, pred)
            json.dump(f1s, open(rst_cache_path, 'w'))
        rst.append([full_name(os.path.basename(cache), True), f1s])
    draw(rst, lower, output_folder)


def run():
    output_dir = 'figs/outputs'
    data_root = 'figs/data/nli'
    exps = ['ro-128', 'lf-1024', 'xl-256-256']
    bd_nli(data_root, output_dir, exps)


if __name__ == '__main__':
    run()
