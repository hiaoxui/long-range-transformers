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


def run():
    data_file = 'figs/data/summ-entropy.json'
    output_folder = 'figs/outputs'
    s2s_entropy(data_file, output_folder)


def s2s_entropy(data_file, output_folder):
    entropy = json.load(open(data_file))
    name2label = [
        ['LED', 'LED ($L$=$\infty$)'],
        ['LED-S', 'LED ($L$=1536)']
    ]
    en_xs = np.array(entropy[name2label[0][0]]['x'])
    en_xs = en_xs / 1000
    plt.style.use('classic')
    fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))
    plotter = Plotter(fig, ax)

    for name, label in name2label:
        ys = np.array(entropy[name]['y'])
        xs = en_xs
        ys = ys[xs >= 2.0]
        xs = xs[xs >= 2.0]
        ys = ys[xs <= 14.0]
        xs = xs[xs <= 14.0]
        plotter(xs, ys, label=label)
    ax.legend(prop={'size': 12}, frameon=False, loc=0)
    ax.set_xlabel('Document Length (thousands)')
    ax.set_ylabel('Entropy')
    ax.set_xlim(1.5, 14.5)
    ax.set_ylim(3.3, 5.5)
    ax.set_yticks(np.arange(3.5, 5.6, 0.5))

    plotter.save_fig(os.path.join(output_folder, 's2s-entropy.pdf'))


if __name__ == '__main__':
    run()
