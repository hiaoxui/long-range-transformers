import os
import sys


short_name = [
    ['bb', 'BigBird'],
    ['xl', 'XLNet'],
    ['lf', 'Longformer'],
    ['ro', 'RoBERTa']
]


def full_name(short, with_length=False):
    full = 'Unknown'
    for sh, lo in short_name:
        if sh in short:
            full = lo
    items = short.split('-')
    le = mem = 0
    if len(items) > 1:
        le = int(items[1])
    if len(items) > 2:
        mem = int(items[2])
    if full == 'XLNet' and mem > 0:
        full = full + '$^m$'
    if with_length:
        full = full + f' ($L$={le})'
    return full


cb_colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
cb_colors = [(int(co[1:3], base=16)/256, int(co[3:5], base=16)/256, int(co[5:7], base=16)/256) for co in cb_colors]
markers = ['.', '^', 'v', ',', 'p', '*']


class Plotter:
    def __init__(self, fig, ax):
        self.fig = fig
        self.ax = ax
        self.idx = 0

    def __call__(self, xs, ys, label, no_marker=False):
        if no_marker:
            self.ax.plot(xs, ys, '-', label=label, color=cb_colors[self.idx])
        else:
            lin, = self.ax.plot(xs, ys, '-', label=label, marker=markers[self.idx], color=cb_colors[self.idx], markersize=4)
            lin.set_markerfacecolor(list(cb_colors[self.idx]) + [0.5])
        self.idx += 1

    def save_fig(self, save_path):
        self.fig.tight_layout()
        self.fig.show()
        tmp_dst = os.path.join(os.path.dirname(save_path), 'tmp.pdf')
        self.fig.savefig(tmp_dst)
        from pdfCropMargins import crop
        crop([tmp_dst, '-o', save_path])
        os.remove(tmp_dst)
