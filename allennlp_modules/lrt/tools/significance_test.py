import os
import json
from collections import defaultdict

import numpy as np
from scipy.stats import norm


def sample(xs):
    return xs[np.random.randint(0, len(xs), [len(xs)])]


def mean_equality(small, big):
    def statistics(xs, ys):
        return (xs.mean() - ys.mean()) / np.sqrt(np.std(xs) ** 2 / len(xs) + np.std(ys) ** 2 / len(ys))
    t = statistics(big, small)
    nb = 1024
    mean = np.concatenate([small, big], axis=0).mean()
    small_, big_ = small-small.mean()+mean, big-big.mean()+mean
    ts = list()
    for _ in range(nb):
        s, b = sample(small_), sample(big_)
        ts.append(statistics(b, s))
    return (np.array(ts) >= t).sum()/nb


def single_side(small, big):
    nb = 1024
    trues = list()
    for _ in range(nb):
        idx = np.random.randint(0, len(small), [len(small)])
        s, b = small[idx], big[idx]
        trues.append(b.mean() < s.mean())
    n_true = sum(trues)
    mu, sigma = nb / 2, np.sqrt(nb / 4)
    normal = (n_true - mu) / sigma
    pv = norm.cdf(normal)
    return pv


def b_cubed(predictions, lower, upper):
    from lrt.tools.plot.bd_c2f import b_cubed_helper
    f1s = list()
    for pred in predictions:
        gold = [tuple(tuple(span) for span in clu) for clu in pred['gold']]
        pred = [tuple(tuple(span) for span in clu) for clu in pred['pred']]
        p = b_cubed_helper(pred, gold, lower, upper)
        r = b_cubed_helper(gold, pred, lower, upper)
        if len(p) * len(r) == 0:
            f1s.append(0)
            continue
        p = np.mean(p)
        r = np.mean(r)
        f1s.append(0. if p * r == 0 else 2 / (1/p + 1/r))
    return np.array(f1s)


def test_c2f():
    from lrt.tools.plot.bd_c2f import get_chunks
    exps = ['lf-128', 'lf-512', 'lf-4096', 'xl-512-512']
    preds = {exp: json.load(open(os.path.join(f'figs/data/c2f/{exp}/pred.json'))) for exp in exps}
    n_chunk = 8
    lower, upper = get_chunks(n_chunk)
    bounds = list(zip(lower, upper))
    rst = dict()
    for label, outputs in preds.items():
        f1s = list()
        for bd in bounds:
            f1s.append(b_cubed(outputs, *bd))
        rst[label] = np.array(f1s)
    pvs = list()
    for chunk_idx in range(n_chunk):
        print('Test for chunk index', chunk_idx)
        big = rst['lf-512'][chunk_idx]
        pv = list()
        for exp in exps:
            if exp == 'lf-512': continue
            small = rst[exp][chunk_idx]
            pv.append(single_side(small, big))
        pvs.append(pv)
    pvs = np.array(pvs)
    x = 1


def test():
    test_c2f()


if __name__ == '__main__':
    test()
