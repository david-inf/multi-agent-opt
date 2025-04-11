"""
We analyze two scenarios, in each we have:
- 2 coefficients for global features
- An agent-specific bias that modifies local targets
"""

from types import SimpleNamespace
import numpy as np

from utils import set_seeds


def split_data(data, n_agents):
    # split data
    data_splits = np.split(data, n_agents)  # list of arrays 2D

    targets_splits = []  # list of arrays 1D
    agent_splits = []  # list of dict
    for i in range(n_agents):
        # local split
        features_i = data_splits[i]  # [N,(p+p_i)]

        # generate weights from gaussian distribution
        mu_i = [0.2, -1.5, np.random.rand()-0.5]  # [(p+p_i)]
        sigma_i = [[1, 0, 0.5], [0, 1, 0], [0.5, 0, 1]]  # [(p+p_i),(p+p_i)]
        weights_i = np.random.multivariate_normal(mu_i, sigma_i)  # [(p+p_i)]
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

    agent_splits = split_data(data, n_samples, n_agents)
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

    agent_splits = split_data(data, n_samples, n_agents)
    return agent_splits


def main(opts):
    import matplotlib.pyplot as plt

    n_ag = opts.n_agents
    fig, axs = plt.subplots(n_ag//2, n_ag-n_ag//2, layout="constrained")
    fig.suptitle("Local targets distribution")

    # Dataset for scenario 1
    agent_splits = dataset1(opts.seed, opts.n_samples, opts.n_agents)

    for i, (agent_data, ax) in enumerate(zip(agent_splits, axs.flatten())):
        print("Agent", i)

        local_features = agent_data["features"]
        local_targets = agent_data["targets"]
        print("Features:", local_features.shape,
              "Targets:", local_targets.shape)
        print(local_features[:3], local_targets[:3])
        print()

        ax.hist(local_targets, bins=10, density=True)
        ax.set_title(f"Agent {i}")
    plt.show()


if __name__ == "__main__":
    from ipdb import launch_ipdb_on_exception

    config = dict(seed=42, n_samples=2000, n_agents=5,
                  dataset="dataset1")
    opts = SimpleNamespace(**config)

    with launch_ipdb_on_exception():
        main(opts)
