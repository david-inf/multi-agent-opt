
import numpy as np
from mydata import dataset1, dataset2
from network import (random_nodes, ring_nodes, plot_network,
                     connect_agents, metropolis_consensus)
from train import Agent
from utils import LOG, set_seeds


def get_dataset(dataset_name):
    if dataset_name == "dataset1":
        dataset_fun = dataset1
    elif dataset_name == "dataset2":
        dataset_fun = dataset2
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
    return dataset_fun


def agents_setup(opts, dataset_fun: callable):
    # Get global dataset and agent-specific biases
    agent_splits = dataset_fun(opts.seed, opts.n_samples, opts.n_agents)
    # Agents
    agents = []
    for i in range(opts.n_agents):
        # get agent-specific dataset (global fraction and local data)
        agent_data = agent_splits[i]
        # create agent object with empty neighbors list
        agent_i = Agent(i, agent_data["global_features"],
                        agent_data["local_features"], agent_data["targets"])
        agents.append(agent_i)

    return agents


def get_network(opts):
    if opts.topology == "random":
        dist_mat, coords = random_nodes(opts.seed, opts.n_agents)
    elif opts.topology == "ring":
        dist_mat, coords = ring_nodes(opts.n_agents)
    else:
        raise ValueError(f"Unknown topology {opts.topology}")
    return dist_mat, coords


def main(opts):
    set_seeds(opts.seed)

    # 1) Get dataset function
    dataset_fun = get_dataset(opts.dataset)

    # 2) Generate data, init agents and create network
    agents = agents_setup(opts, dataset_fun)
    dist_mat, coords = get_network(opts)
    connect_agents(opts.dist_thresh, dist_mat, agents)
    plot_network(coords, agents, opts.experiment_name)

    # 3) Compute (train) local solution for each agent
    for agent in agents:
        agent.train()  # solve local least-squares
    # 4) Consensus algorithm


if __name__ == "__main__":
    from cmd_args import parse_args
    from ipdb import launch_ipdb_on_exception
    opts = parse_args()

    with launch_ipdb_on_exception():
        main(opts)
