"""We analyze two scenarios"""

import numpy as np

from utils import set_seeds


def dataset1(opts):
    """Global data for scenario 1"""
    set_seeds(opts.seed)
    x1 = np.random.uniform(-10, 10, opts.samples)
    x2 = np.random.uniform(-1, 1, opts.samples)
    data = np.hstack((x1, x2))
    noise = 0.8 * np.random.randn(opts.samples)
    targets = data.dot(opts.weights) + noise

    return data, targets


def dataset2(opts):
    """Global data for scenario 2"""
    set_seeds(opts.seed)
    x1 = np.random.uniform(-1, 1, opts.samples)
    x2 = np.random.uniform(-10, 10, opts.samples)
    noise = 0.8 * np.random.randn(opts.samples)


def main(opts):
    # Dataset for scenario 1
    data, targets = dataset1(opts)


if __name__ == "__main__":
    from types import SimpleNamespace
    from ipdb import launch_ipdb_on_exception

    config = dict(seed=42, samples=1000, agents=5,
                  weights=[0.8, -1.5])
    opts = SimpleNamespace(**config)

    with launch_ipdb_on_exception():
        main(opts)
