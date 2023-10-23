import os
import json

from matplotlib import pyplot as plt
import numpy as np

from lrt.tools.plot.util import Plotter


def smooth(xs, ys, bucket):
    effective_length = (len(xs) // bucket) * bucket
    xs, ys = xs[:effective_length], ys[:effective_length]
    xs = xs.reshape(-1, bucket)[:, 0]
    ys = ys.reshape(-1, bucket).mean(1)
    return [xs, ys]


def draw(data, dst, xmin, xmax, ylim, yticks, ylabel, xlabel):
    name2label = [
        ['LED', 'LED ($L$=$\infty$)'],
        ['LED-G', 'LED$^G$ ($L$=$\infty$)'],
        # ['LED-S', 'LED ($L$=512)']
    ]
    plt.style.use('classic')
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if xlabel:
        fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(5, 2.0))
    plotter = Plotter(fig, ax)
    for name, label in name2label:
        xs, ys = [np.array(data[name][z]) for z in 'xy']
        xs = xs/1000
        xs, ys = xs[xs <= xmax], ys[xs <= xmax]
        xs, ys = xs[xmin <= xs], ys[xmin <= xs]
        plotter(xs, ys, label)
    ax.legend(prop={'size': 9}, frameon=False, loc=0)
    if xlabel:
        ax.set_xlabel('Document Length (thousands)')
    ax.set_ylabel(ylabel)
    ax.set_xlim(xmin-0.5, xmax+0.5)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    plotter.save_fig(dst)


def query_entropy_qasper(data_path, output_folder):
    entropy = json.load(open(data_path))
    qasper_output = os.path.join(output_folder, 'qasper-entropy.pdf')
    quality_output = os.path.join(output_folder, 'quality-entropy.pdf')
    draw(entropy['abs_qa'], qasper_output, 0.5, 11, (2, 5.2), np.arange(2.5, 4.9, 0.5), 'Entropy (Qasper)', False)
    draw(entropy['mrc_qa'], quality_output, 2.5, 10, (1.3, 2.0), np.arange(1.4, 1.9, 0.2), 'Entropy (QuALITY)', True)


def run():
    data_path = 'figs/data/qa-entropy.json'
    output_folder = 'figs/outputs'
    query_entropy_qasper(data_path, output_folder)


if __name__ == '__main__':
    run()
