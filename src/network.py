"""Design the multi-agent system"""

from types import SimpleNamespace
import numpy as np
import numpy.linalg as la

from train import Agent
from utils import set_seeds


def distance_matrix(coords) -> np.ndarray:
    n_agents = coords.shape[0]
    dist_mat = np.zeros((n_agents, n_agents))
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            # euclidean distance
            dist = la.norm(coords[i, :] - coords[j, :])
            # symmetric matrix
            dist_mat[i, j], dist_mat[j, i] = dist, dist

    return dist_mat


def generate_network(seed, n_agents, l=5):
    set_seeds(seed)
    # random coordinates
    x_coords = np.random.uniform(-l//2, l//2, n_agents)
    y_coords = np.random.uniform(-l//2, l//2, n_agents)
    # coords matrix
    coords = np.column_stack((x_coords, y_coords))
    # distance matrix between nodes
    dist = distance_matrix(coords)

    return dist, coords


def connect_agents(thresh, dist_mat, agents):
    """
    Check distance then add connection
    One must ensure that the graph is connected
    """
    for i, agent in enumerate(agents):
        dist = dist_mat[i]
        # adjacency for the given agent as boolean array
        adj = np.logical_and(dist > 0., dist < thresh)
        # add agent_id of neighbors
        agent.update_neighbor(np.where(adj)[0].tolist())


def adjacency_matrix(agents) -> np.ndarray:
    # get the adjacency matrix given the agent objects
    adj_mat = np.zeros((len(agents), len(agents)))
    for i, agent in enumerate(agents):
        for j in agent.neighbors:
            adj_mat[i, j] = 1.

    return adj_mat


def laplacian_consensus(adjacency, eps=0.01) -> np.ndarray:
    # get the laplacian consensus matrix
    d_max = adjacency.sum(1).max()
    assert eps > 0. and eps < 1/d_max, "Broken Laplacian weights condition"
    degree_mat = np.diag(adjacency.sum(1))
    laplacian_mat = degree_mat - adjacency
    return np.eye(adjacency.shape[0]) - eps * laplacian_mat


def metropolis_consensus(adjacency, agents) -> np.ndarray:
    # get the consensus matrix with metropolis weights
    # exploits only local information, however we build the full matrix
    metropolis_mat = np.zeros_like(adjacency)
    for i, agent in enumerate(agents):
        # build off-diagonal first
        neighbor_sum = 0
        for j in agent.neighbors:
            weight_ij = 1 / (1 + max(agent.degree, agents[j].degree))
            metropolis_mat[i, j] = weight_ij
            neighbor_sum += weight_ij
        # then build diagonal
        metropolis_mat[i, i] = 1 - neighbor_sum

    return metropolis_mat


def plot_network(coords, agents):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    # plot agent nodes
    ax.scatter(coords[:, 0], coords[:, 1], marker="o", s=50)
    # plot agent ids and connections
    for i, agent in enumerate(agents):
        ax.text(coords[i, 0]+0.15, coords[i, 1]+0.15, str(i),
                fontsize=12, ha="center", va="center")
        for j in agent.neighbors:
            ax.plot([coords[i, 0], coords[j, 0]], [
                    coords[i, 1], coords[j, 1]], "c-")
    # decorate
    ax.grid(True, which="both", axis="both")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.savefig("./plots/network.pdf")


def main(opts):
    # generate nodes
    dist_mat, coords = generate_network(opts.seed, opts.n_agents)
    # create agents
    agents = [Agent(i) for i in range(opts.n_agents)]
    # connect agents that are near
    connect_agents(opts.dist_thresh, dist_mat, agents)
    [print(agent) for agent in agents]
    # print adjacency matrix
    adj_mat = adjacency_matrix(agents)
    print("\nAdjacency matrix:")
    print(adj_mat)
    # plot network
    plot_network(coords, agents)

    # laplacian weights
    lap_weights = laplacian_consensus(adj_mat)
    print("\nLaplacian weights:")
    print(lap_weights)
    print(lap_weights.sum(1))

    # metropolis weights
    metr_weights = metropolis_consensus(adj_mat, agents)
    print("\nMetropolis weights:")
    print(metr_weights)
    print(metr_weights.sum(1))


if __name__ == "__main__":
    from ipdb import launch_ipdb_on_exception

    config = dict(seed=42, n_samples=1000, n_agents=8, dist_thresh=3.)
    opts = SimpleNamespace(**config)

    with launch_ipdb_on_exception():
        main(opts)
