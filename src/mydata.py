"""
Each agent has the following parameters:
- 2 coefficients for global features
- An agent-specific bias that modifies local targets
We analyze two scenarios:
- All agents get the same amount of data
- All agents get a different amount of data
We'd like to se the impact on consensus convergence
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import set_seeds, LOG, AverageMeter


def get_features(opts):
    """Generate random features of size 2 and constant columns for bias"""
    N = opts.n_samples

    x1_half1 = np.random.uniform(-10, 10, N//2)
    x1_half2 = np.random.uniform(-1, 1, N-N//2)
    x1 = np.hstack((x1_half1, x1_half2))

    x2_half1 = np.random.uniform(-10, 10, int(0.2*N))
    x2_half2 = np.random.uniform(-1, 1, N-int(0.2*N))
    x2 = np.hstack((x2_half1, x2_half2))

    # randomize rows so each agents gets both types of features
    set_seeds(opts.seed)
    np.random.shuffle(x1)
    np.random.shuffle(x2)

    # full dataset with constant column for local bias
    features = np.column_stack((x1, x2, np.ones(N)))

    return features


def targets_and_splits(opts, features: np.ndarray):
    """Given the full dataset, split for each agent"""
    if opts.dataset == "balanced":
        # all agents get the same amount of samples
        data_splits = np.array_split(features, opts.n_agents)  # list of arrays 2D
    elif opts.dataset == "unbalanced":
        # all agents get a different amount of samples
        # 1) generate a random number of samples for each agent
        splits = np.random.randint(1, opts.n_samples, opts.n_agents-1)
        splits = np.sort(splits)  # sort to avoid empty splits
        # 2) split the dataset
        data_splits = np.split(features, splits)  # list of arrays 2D
    else:
        raise ValueError(f"Unknown dataset {opts.dataset}")

    # fixed common parameters and variable local biases
    w = [0.5, -0.8]  # [p]
    beta_i = np.random.uniform(-2, 2, opts.n_agents)  # [p_i]
    # beta_i = np.array([0.8]*n_agents)

    LOG.info("Actual parameters:")
    w_local = AverageMeter()
    agent_splits = []  # list of dict
    for i in range(opts.n_agents):
        # local split
        features_i = data_splits[i]  # [N,(p+p_i)]

        # local weights (common + local)
        w_i = np.hstack((w, beta_i[i]))  # [p+p_i]
        w_local.update(w_i)

        with np.printoptions(precision=4):
            LOG.info("Agent %d, w_i=%s, samples=%d",
                     i, w_i, features_i.shape[0])

        # additive noise
        noise = 0.8*np.random.randn(features_i.shape[0])  # [N]
        # local targets
        targets_i = features_i.dot(w_i) + noise  # [N]

        # add to agent splits
        agent_data = dict(features=features_i, targets=targets_i)
        agent_splits.append(agent_data)
    print()

    return agent_splits  # list of dict


def get_dataset(opts):
    """
    Generate data (just the covariates) for each agent with features of two types
        p=2 (coefficients) and p_i=1 (bias)
    """
    set_seeds(opts.seed)
    # 1) Generate features
    features = get_features(opts)

    # 2) Split dataset and generate targets for each agent
    agent_splits = targets_and_splits(opts, features)

    return agent_splits


def main(opts):
    n_ag = opts.n_agents
    # TODO: better handling of dynamic number of plots
    fig, axs = plt.subplots(n_ag//2, 4, layout="constrained")
    fig.suptitle("Local targets distribution")

    agent_splits = get_dataset(opts)

    for i, (agent_data, ax) in enumerate(zip(agent_splits, axs.flatten())):
        LOG.info("Agent: %d", i)

        local_features = agent_data["features"]
        local_targets = agent_data["targets"]
        print("Features:", local_features.shape,
              "Targets:", local_targets.shape)
        print(local_features[:3], local_targets[:3])
        print("")

        ax.hist(local_targets, bins=10, density=True)
        ax.set_title("Agent: %d", i)

    plt.show()


if __name__ == "__main__":
    from cmd_args import parse_args
    from ipdb import launch_ipdb_on_exception
    args = parse_args()
    with launch_ipdb_on_exception():
        main(args)
