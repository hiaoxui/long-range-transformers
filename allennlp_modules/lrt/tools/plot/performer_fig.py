import json
import os
import re
from collections import defaultdict

from matplotlib import pyplot as plt
import numpy as np
import torch

from lrt.tools.plot.util import Plotter


def run():
    data_path = 'figs/data/performer'
    output_folder = 'figs/outputs'
    performer(data_path, output_folder)


def performer(data_path, output_folder):
    f1s = defaultdict(list)
    to_show = [4, 16, 64]
    for fn in os.listdir(data_path):
        if not re.match(r'fast.\d+.11.\d+.json', fn):
            continue
        i_layer = int(fn.split('.')[1])
        num_features = int(fn.split('.')[3])
        if num_features not in to_show:
            continue
        metrics = json.load(open(os.path.join(data_path, fn)))
        f1 = metrics['coref_f1']
        f1s[num_features].append([i_layer, f1])
    for v in f1s.values():
        v.sort(key=lambda x: x[0])

    plt.style.use('classic')
    fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))
    plotter = Plotter(fig, ax)
    f1s = list(f1s.items())
    f1s.sort(key=lambda x: x[0])
    ax.plot([0.5, 12], [76.59, 76.59], '--', label='Baseline', color='red')
    for nf, f1s_ in f1s:
        xs, ys = list(zip(*f1s_))
        xs = 12 - np.array(xs)
        plotter(xs, ys, f'#Fea={nf}')

    edge = 0.5
    ax.set_xlim(1-edge, 12 + edge)
    ax.set_yticks(torch.arange(20, 81, 20))
    ax.set_xticks(torch.arange(1, 12, 2))
    ax.legend(prop={'size': 10}, frameon=False, loc=0)
    ax.set_xlabel('#Replaced Layers')
    ax.set_ylabel('Avg. F$_1$')
    plotter.save_fig(os.path.join(output_folder, 'performer.pdf'))


if __name__ == '__main__':
    run()
