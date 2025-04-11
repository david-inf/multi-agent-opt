"""
We analyze two scenarios, in each we have:
- 2 coefficients for global features
- An agent-specific bias that modifies local targets
"""

import numpy as np

from utils import set_seeds, LOG


def split_data(data, n_agents):
    # split data
    data_splits = np.array_split(data, n_agents)  # list of arrays 2D

    targets_splits = []  # list of arrays 1D
    agent_splits = []  # list of dict
    for i in range(n_agents):
        # local split
        features_i = data_splits[i]  # [N,(p+p_i)]

        # generate weights from gaussian distribution
        # TODO: consider fixing the weights or reducing variability
        # for getting the actual weights the clt applies to n_agents
        mu_i = [0.5, -0.8, np.random.rand()-0.5]  # [(p+p_i)]
        with np.printoptions(precision=4):
            LOG.info(f"Agent {i}, mu={np.array(mu_i)}")
        sigma_i = [[1., 0.5, 0.], [0.5, 1., 0.], [0., 0., 0.3]]  # [(p+p_i),(p+p_i)]
        weights_i = np.random.multivariate_normal(mu_i, sigma_i)  # [(p+p_i)]

        # TODO: weights with fixed common part
        # weights_i = [0.5, -0.8, 2*np.random.rand()-1]
        # with np.printoptions(precision=4):
        #     LOG.info(f"Agent {i}, weights={np.array(weights_i)}")

        # additive noise
        noise = 0.8*np.random.randn(features_i.shape[0])  # [N]
        # agent-specific targets
        local_targets = features_i.dot(weights_i) + noise  # [N]
        targets_splits.append(local_targets)

        # add to agent splits
        agent_data = {}
        agent_data["features"] = features_i  # [N,(x1,x2,1)]
        agent_data["targets"] = local_targets  # [N]
        agent_splits.append(agent_data)

    return agent_splits  # list of dict


def dataset1(seed, n_samples, n_agents):
    """Global data for scenario 1"""
    set_seeds(seed)

    # generate the full dataset p=2, p_i=1
    x1 = np.random.uniform(-10, 10, n_samples)
    x2 = np.random.uniform(-1, 1, n_samples)
    global_features = np.column_stack((x1, x2))  # coefficients
    local_features = np.ones(n_samples)  # bias
    data = np.column_stack((global_features, local_features))

    agent_splits = split_data(data, n_agents)
    return agent_splits


def dataset2(seed, n_samples, n_agents):
    """Global data for scenario 2"""
    set_seeds(seed)

    # generate the full dataset p=2, p_i=1
    x1 = np.random.uniform(-1, 1, n_samples)
    x2 = np.random.uniform(-10, 10, n_samples)
    global_features = np.column_stack((x1, x2))  # coefficients
    local_features = np.ones(n_samples)  # bias
    data = np.column_stack((global_features, local_features))

    agent_splits = split_data(data, n_agents)
    return agent_splits


def get_dataset(dataset_name):
    if dataset_name == "dataset1":
        dataset_fun = dataset1
    elif dataset_name == "dataset2":
        dataset_fun = dataset2
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
    return dataset_fun


def main(opts):
    import matplotlib.pyplot as plt

    n_ag = opts.n_agents
    # TODO: better handling of dynamic number of plots
    fig, axs = plt.subplots(n_ag//2, 4, layout="constrained")
    fig.suptitle("Local targets distribution")

    dataset_fun = get_dataset(opts.dataset)
    agent_splits = dataset_fun(opts.seed, opts.n_samples, opts.n_agents)

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
