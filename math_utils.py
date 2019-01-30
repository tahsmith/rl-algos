import numpy as np


def constant(x):
    def f(*_, **__):
        return x

    return f


def clipped_exp_decay(x0, xmin, t):
    def f(i):
        return max(
            x0 * np.exp(-i * t),
            xmin
        )

    return f
