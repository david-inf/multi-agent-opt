"""We analyze two scenarios"""

import numpy as np

from utils import set_seeds


def split_data(n_agents, data, bias, targets):
    # split data and targets
    data_splits = np.split(data, n_agents)
    targets_splits = np.split(targets, n_agents)
    sizes = [split.size for split in targets_splits]

    agent_splits = []  # list of dict
    for i in range(n_agents):
        agent_data = {}
        # x1 and x2 splits
        agent_data["global_features"] = data_splits[i]
        # local features may be different for each agent (not in this case)
        agent_data["local_features"] = np.ones((sizes[i]))
        # update targets with local bias
        agent_data["targets"] = targets_splits[i] + bias[i]
        # add to splits
        agent_splits.append(agent_data)

    return agent_splits


def dataset1(seed, n_samples, n_agents):
    """Global data for scenario 1"""
    set_seeds(seed)
    # generate the full dataset
    x1 = np.random.uniform(-10, 10, n_samples)
    x2 = np.random.uniform(-1, 1, n_samples)
    data = np.column_stack((x1, x2))  # global features
    # random noise for the target model
    noise = 0.8 * np.random.randn(n_samples)

    # generate global weights
    weights = np.random.multivariate_normal([0.2, 0.1], [[1, 0], [0, 1]])
    print(f"True weights: {weights}")

    # compute targets
    targets = data.dot(weights) + noise
    # generate agent-specific biases
    bias = np.random.uniform(-1, 1, n_agents)

    # get splits
    agent_splits = split_data(n_agents, data, bias, targets)
    return agent_splits


def dataset2():
    """Global data for scenario 2"""
    return None


def main(opts):
    # Dataset for scenario 1
    agent_splits = dataset1(opts.seed, opts.n_samples, opts.n_agents)


if __name__ == "__main__":
    from types import SimpleNamespace
    from ipdb import launch_ipdb_on_exception

    config = dict(seed=42, n_samples=1000, n_agents=5)
    opts = SimpleNamespace(**config)

    with launch_ipdb_on_exception():
        main(opts)
