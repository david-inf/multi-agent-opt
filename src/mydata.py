"""
We analyze two scenarios, in each we have:
- 2 coefficients for global features
- An agent-specific bias that modifies local targets
"""

import numpy as np
from utils import set_seeds, LOG, AverageMeter


def split_data(shared_features, n_samples, n_agents):
    # split data
    features = np.column_stack((shared_features, np.ones(n_samples)))
    data_splits = np.array_split(features, n_agents)  # list of arrays 2D

    w_local = AverageMeter()
    agent_splits = []  # list of dict
    for i in range(n_agents):
        # local split
        features_i = data_splits[i]  # [N,(p+p_i)]

        # generate weights from gaussian distribution
        mu_i = [0.5, -1.5, 0.2]  # [(p+p_i)]
        sigma_i = 0.1 * np.eye(3)  # [(p+p_i),(p+p_i)]
        # sigma_i = np.zeros((3,3))  # just as test

        weights_i = np.random.multivariate_normal(mu_i, sigma_i)  # [(p+p_i)]
        w_local.update(weights_i)

        with np.printoptions(precision=4):
            LOG.info(f"Agent {i}, w_i={weights_i}")

        # additive noise then local targets
        noise = 0.8*np.random.randn(features_i.shape[0])  # [N]
        targets_i = features_i.dot(weights_i) + noise  # [N]

        # add to agent splits
        agent_data = dict(features=features_i, targets=targets_i)
        agent_splits.append(agent_data)

    with np.printoptions(precision=4):
        LOG.info(f"Synthetic w_i_avg={w_local.avg}\n")

    return agent_splits  # list of dict


def get_dataset(opts):
    set_seeds(opts.seed)
    # p=2 and p_i=1
    if opts.dataset == "dataset1":  # scenario 1
        x1 = np.random.uniform(-10, 10, opts.n_samples)
        x2 = np.random.uniform(-1, 1, opts.n_samples)
    elif opts.dataset == "dataset2":  # scenario 2
        x1 = np.random.uniform(-1, 1, opts.n_samples)
        x2 = np.random.uniform(-10, 10, opts.n_samples)
    else:
        raise ValueError(f"Unknown dataset {opts.dataset}")

    shared_features = np.column_stack((x1, x2))
    agent_splits = split_data(shared_features, opts.n_samples, opts.n_agents)

    return agent_splits


def main(opts):
    import matplotlib.pyplot as plt

    n_ag = opts.n_agents
    # TODO: better handling of dynamic number of plots
    fig, axs = plt.subplots(n_ag//2, 4, layout="constrained")
    fig.suptitle("Local targets distribution")

    agent_splits = get_dataset(opts)

    for i, (agent_data, ax) in enumerate(zip(agent_splits, axs.flatten())):
        LOG.info(f"Agent: {i}")

        local_features = agent_data["features"]
        local_targets = agent_data["targets"]
        print("Features:", local_features.shape,
              "Targets:", local_targets.shape)
        print(local_features[:3], local_targets[:3])
        print("")

        ax.hist(local_targets, bins=10, density=True)
        ax.set_title(f"Agent {i}")

    plt.show()


if __name__ == "__main__":
    from cmd_args import parse_args
    from ipdb import launch_ipdb_on_exception
    opts = parse_args()
    with launch_ipdb_on_exception():
        main(opts)
