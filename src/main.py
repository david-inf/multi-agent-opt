
import numpy as np
from mydata import dataset1, dataset2
from network import generate_nodes, plot_network, connect_agents
from train import Agent
from utils import LOG, set_seeds


def agents_setup(opts, dataset_fun: callable):
    # Get global dataset and agent-specific biases
    agent_splits = dataset_fun(opts.seed, opts.n_samples, opts.n_agents)
    # Agents
    agents = []
    for i in range(opts.n_agents):
        # get agent-specific dataset
        agent_data = agent_splits[i]
        # create agent object
        agent_i = Agent(i, agent_data["global_features"],
                        agent_data["local_features"], agent_data["targets"])
        agents.append(agent_i)

    return agents


def main(opts):
    set_seeds(opts.seed)
    # 1) Generate data
    if opts.dataset == "dataset1":
        dataset_fun = dataset1
    elif opts.dataset == "dataset2":
        dataset_fun = dataset2
    else:
        raise ValueError(f"Unknown dataset {opts.dataset}")
    # 2) Init agents and create network
    agents = agents_setup(opts, dataset_fun)
    dist_mat, coords = generate_nodes(opts.seed, opts.n_agents)
    connect_agents(opts.dist_thresh, dist_mat, agents)
    plot_network(coords, agents, opts.experiment_name)
    # 3) Compute (train) local solution for each agent
    for agent in agents:
        agent.train()  # solve local least-squares
    pass
    # 4) Consensus algorithm


if __name__ == "__main__":
    from cmd_args import parse_args
    from ipdb import launch_ipdb_on_exception
    opts = parse_args()

    with launch_ipdb_on_exception():
        main(opts)
