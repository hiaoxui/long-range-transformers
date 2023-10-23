import os
import json

from matplotlib import pyplot as plt
import numpy as np

from lrt.tools.plot.util import Plotter


def run():
    data_file = 'figs/data/extqa-query-gradient.json'
    output_folder = 'figs/outputs'
    query_gradient(data_file, output_folder)


def query_gradient(data_file, output_folder):
    dst = os.path.join(output_folder, 'trivia-gradient.pdf')
    name2label = [
        ['Lonformer', 'Longformer'],
        ['Lonformer-G', 'Longformer$^G$'],
        # ['XLNet-m', 'XLNet$^m$'],
        ['BigBird', 'BigBird'],
    ]
    plt.style.use('classic')
    data = json.load(open(data_file))
    fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))
    plotter = Plotter(fig, ax)
    for name, label in name2label:
        xs, ys = [np.array(data[name][z]) for z in 'xy']
        plotter(xs, ys, label)

    ax.legend(prop={'size': 10}, frameon=False, loc=0)
    ax.set_xlabel('Query Proportion')
    ax.set_ylabel('Query Gradient')
    ax.set_xlim(-0.05, 0.65)
    plotter.save_fig(dst)


if __name__ == '__main__':
    run()
