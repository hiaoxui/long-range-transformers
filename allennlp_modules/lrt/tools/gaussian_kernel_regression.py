from typing import *

import numpy as np


class GaussianKernelRegression:
    def __init__(self, xs: np.ndarray, ys: np.ndarray):
        self.xs, self.ys = np.array(xs), np.array(ys)

    def predict(self, new: np.ndarray, bandwidth=10.):
        n, m = len(self.xs), len(new)
        new = new.reshape(-1, 1).repeat(n, axis=1)
        old = self.xs.reshape(-1, 1).repeat(m, axis=1).T
        square = (new - old)**2
        kernel = np.exp(-0.5 * square / bandwidth**2)
        sum1 = kernel.sum(axis=1)
        sum1[sum1 == 0.] = 1.0
        weights = (kernel.T / sum1).T
        return weights @ self.ys
