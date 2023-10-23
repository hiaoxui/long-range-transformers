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
    data_file = 'figs/data/summ-gradient.json'
    output_folder = 'figs/outputs'
    s2s_gradient(data_file, output_folder)


def s2s_gradient(data_file, output_folder):
    output_path = os.path.join(output_folder, 's2s-gradient.pdf')
    gradient = json.load(open(data_file))
    name2label = [
        ['LED', 'LED ($L$=$\infty$)'],
        ['LED-S', 'LED ($L$=1536)']
    ]
    plt.style.use('classic')
    fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))
    plotter = Plotter(fig, ax)

    for name, label in name2label:
        ys = np.array(gradient[name]['y'])
        xs = np.array(gradient[name]['x'])/1000
        xs, ys = smooth(xs, ys, 40)
        ys = ys[xs >= 2.0]
        xs = xs[xs >= 2.0]
        plotter(xs, ys*10000, label, no_marker=True)
    ax.set_ylim(0.4, 1.8)
    ax.set_xlim(1.5, 15.5)
    ax.legend(prop={'size': 12}, frameon=False, loc=0)
    ax.set_ylabel('Gradient Dist. / $10^{-4}$')
    ax.set_xlabel('Token Index /  $10^{3}$')
    ax.set_yticks(np.arange(0.5, 2.0, 0.5))
    plotter.save_fig(output_path)


if __name__ == '__main__':
    run()
