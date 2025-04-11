"""Design the multi-agent system"""

import os
from types import SimpleNamespace
import numpy as np
import numpy.linalg as la
from typing import List

from train import Agent
from utils import set_seeds


def distance_matrix(coords: np.ndarray) -> np.ndarray:
    n_agents = coords.shape[0]
    dist_mat = np.zeros((n_agents, n_agents))
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            # euclidean distance
            dist = la.norm(coords[i, :] - coords[j, :])
            # symmetric matrix
            dist_mat[i, j], dist_mat[j, i] = dist, dist

    return dist_mat


def random_nodes(seed, n_agents, l=5):
    """Random coordinates for each node"""
    set_seeds(seed)
    # TODO: without replacement and minimum distance between nodes
    x_coords = np.random.uniform(-l//2, l//2, n_agents)
    y_coords = np.random.uniform(-l//2, l//2, n_agents)
    coords = np.column_stack((x_coords, y_coords))
    dist = distance_matrix(coords)
    return dist, coords


def ring_nodes(n_agents, r=5):
    """Place nodes in a circle"""
    # delta between each node in polar coordinates
    theta = 2 * np.pi / n_agents
    # nodes polar coordinates (theta, rho)
    coords_pol = np.zeros((n_agents, 2))
    coords_pol[0, :] = np.array([0, r])  # first node
    # assign position to node
    for i in range(n_agents - 1):
        coords_pol[i + 1, :] = np.array([coords_pol[i, 0] + theta, r])
    # nodes cartesian coordinates (x, y)
    coords = np.zeros((n_agents, 2))
    for i in range(n_agents):
        coords[i, :] = r * np.array([np.cos(coords_pol[i, 0]), np.sin(coords_pol[i, 0])])
    dist_mat = distance_matrix(coords)
    return dist_mat, coords


# def grid_nodes(n_agents)


def connect_agents(thresh, dist_mat: np.ndarray, agents: List[Agent]):
    """
    Check distance then add connection using update_neighbor method
    One must ensure that the graph is connected
    """
    for i, agent in enumerate(agents):
        dist = dist_mat[i]
        # adjacency for the given agent as boolean array
        adj = np.logical_and(dist > 0., dist < thresh)
        # add agent_id of neighbors
        neighbors_ids = np.where(adj)[0].tolist()  # list of bool
        # add agent objects to neighbors list
        neighbors = [agents[j] for j in neighbors_ids]
        # update neighbors list
        agent.update_neighbors(neighbors)


def adjacency_matrix(agents: List[Agent]) -> np.ndarray:
    """Get the adjacency matrix given the agent objects"""
    adj_mat = np.zeros((len(agents), len(agents)))

    for i, agent in enumerate(agents):
        for j, _ in enumerate(agent.neighbors):
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


def plot_network(coords, agents: List[Agent], fname="network.pdf"):
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
    # save network plot
    output_dir = os.path.join("plots", fname)
    os.makedirs("plots", exist_ok=True)
    plt.savefig(output_dir)


def main(opts):
    # generate nodes
    if opts.topology == "random":
        dist_mat, coords = random_nodes(opts.seed, opts.n_agents)
    # elif opts.topology == "circle":
    #     dist_mat, coords = circle_nodes(opts.n_agent)
    else:
        raise ValueError(f"Unknown topology {opts.topology}")
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

    config = dict(seed=42, n_samples=1000, n_agents=8, dist_thresh=3.,
                  topology="random")
    opts = SimpleNamespace(**config)

    with launch_ipdb_on_exception():
        main(opts)
