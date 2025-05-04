
from mydata import get_dataset
from network import random_nodes, ring_nodes, connect_agents
from train import Agent, consensus_algorithm
from utils import set_seeds


def agents_setup(opts):
    """Init agents with data"""
    # Get global dataset and agent-specific biases
    agent_splits = get_dataset(opts)
    # Agents
    agents = []
    for i in range(opts.n_agents):
        # get agent-specific dataset (global fraction and local data)
        agent_data = agent_splits[i]
        # create agent object with empty neighbors list
        agent_i = Agent(i, agent_data["features"], agent_data["targets"])
        agents.append(agent_i)

    return agents


def get_network(opts):
    """Generate network topology"""
    if opts.topology == "random":
        dist_mat, coords = random_nodes(opts.seed, opts.n_agents, opts.grid_size)
    elif opts.topology == "ring":
        dist_mat, coords = ring_nodes(opts.n_agents)
    else:
        raise ValueError(f"Unknown topology {opts.topology}")
    return dist_mat, coords


def main(opts):
    set_seeds(opts.seed)
    # 1) Generate data, init agents and create network
    agents = agents_setup(opts)
    dist_mat, _ = get_network(opts)
    connect_agents(opts.dist_thresh, dist_mat, agents)
    # plot_network(coords, agents, opts.experiment_name)
    # 2) Consensus algorithm
    consensus_algorithm(opts, agents)


if __name__ == "__main__":
    from cmd_args import parse_args
    from ipdb import launch_ipdb_on_exception

    args = parse_args()

    with launch_ipdb_on_exception():
        main(args)
