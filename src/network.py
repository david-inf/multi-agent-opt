"""Design the multi-agent system"""

import numpy as np
import numpy.linalg as la

from utils import set_seeds


def distance_matrix(coords):
    n_agents = coords.shape[0]
    dist_mat = np.zeros((n_agents, n_agents))
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            dist = la.norm(coords[i, :] - coords[j, :])
            # symmetric matrix
            dist_mat[i, j], dist_mat[j, i] = dist, dist

    return dist_mat


def generate_network(opts, n_agents=10, l=5):
    set_seeds(opts.seed)
    x_coords = np.random.uniform(-l//2, l//2, n_agents)
    y_coords = np.random.uniform(-l//2, l//2, n_agents)
    coords = np.column_stack((x_coords, y_coords))
    dist = distance_matrix(coords)
    return dist, coords


def plot_network(coords):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter(coords[:, 0], coords[:, 1], marker="o", s=30)
    ax.grid(True, which="both", axis="both")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.savefig("./plots/network.pdf")


class Agent:
    def __init__(self, agent_id, trainset):
        self.agent_id = agent_id
        self.trainset = trainset


def main(opts):
    dist, coords = generate_network(opts, opts.agents)


if __name__ == "__main__":
    from types import SimpleNamespace
    from ipdb import launch_ipdb_on_exception

    config = dict(seed=42, samples=1000, agents=10, dist_thresh=2.,
                  weights=[0.8, -1.5])
    opts = SimpleNamespace(**config)

    with launch_ipdb_on_exception():
        main(opts)
