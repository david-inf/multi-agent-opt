"""We analyze two scenarios"""

import numpy as np

from utils import set_seeds


def dataset1(seed, n_samples, n_agents):
    """Global data for scenario 1"""
    set_seeds(seed)
    # generate data
    x1 = np.random.uniform(-10, 10, n_samples)
    x2 = np.random.uniform(-1, 1, n_samples)
    data = np.column_stack((x1, x2))
    noise = 0.8 * np.random.randn(n_samples)
    # generate weights
    weights = np.random.multivariate_normal([0.2, 0.1], [[1, 0], [0, 1]])
    # compute targets
    targets = data.dot(weights) + noise
    # generate agent-specific biases
    bias = np.random.uniform(-1, 1, n_agents)

    return (data, targets), bias


# def dataset2():
#     """Global data for scenario 2"""


def main(opts):
    # Dataset for scenario 1
    data, targets = dataset1(opts.seed, opts.n_samples, opts.n_agents)


if __name__ == "__main__":
    from types import SimpleNamespace
    from ipdb import launch_ipdb_on_exception

    config = dict(seed=42, n_samples=1000, n_agents=5)
    opts = SimpleNamespace(**config)

    with launch_ipdb_on_exception():
        main(opts)
