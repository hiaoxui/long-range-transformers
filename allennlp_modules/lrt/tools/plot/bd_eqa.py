import os
import json

from matplotlib import pyplot as plt
import numpy as np

from lrt.tools.plot.util import Plotter


def run():
    data_file = 'figs/data/extqa_breakdown.json'
    dst = 'figs/outputs'
    bd_eqa(data_file, dst)


def bd_eqa(data_file, output_folder):
    data = json.load(open(data_file))
    dst = os.path.join(output_folder, 'extqa_bd.pdf')
    name2label = [
        ['Lonformer-G', 'Longformer$^G$'],
        ['BigBird', 'BigBird ($L$=$\infty$)'],
        ['Lonformer', 'Longformer ($L$=$\infty$)'],
        # ['Lonformer-S', 'Longformer ($L$=512)'],
        # ['XLNet-m', 'XLNet$^m$'],
        # ['XLNet', 'XLNet'],
        ['BigBird-S', 'BigBird ($L$=$512$)'],
    ]
    plt.style.use('classic')
    fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))
    plotter = Plotter(fig, ax)
    xs = np.array(data[name2label[0][0]]['x'])
    for name, label in name2label:
        plotter(xs, np.array(data[name]['y'])*100, label)
    ax.legend(prop={'size': 8}, frameon=False, loc=0)
    ax.set_xlabel('Document Length (thousands)')
    ax.set_ylabel('F$_1$ Score')
    ax.set_yticks(np.arange(40, 81, 20))
    ax.set_xticks(np.arange(1000, 4001, 1000), np.arange(1, 5))
    plotter.save_fig(dst)


if __name__ == '__main__':
    run()
