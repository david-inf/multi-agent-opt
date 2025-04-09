
from mydata import dataset1, dataset2
from train import Agent
from utils import LOG


def network_setup(opts):
    # Get global dataset and agent-specific biases
    if opts.dataset == "dataset1":
        dataset, bias = dataset1(opts.seed, opts.n_agents)
    elif opts.dataset == "dataset2":
        dataset, bias = dataset2(opts.seed, opts.n_agents)
    else:
        raise ValueError(f"Unknown dataset {opts.dataset}")
    # Agents
    agents = []
    for i in range(opts.n_agents):
        # specific data for each agent
        bias_i = bias[i]  # scalar
        # create agent object
        agent_i = Agent(i, dataset, bias_i)
        agents.append(agent_i)


def main(opts):
    return None


if __name__ == "__main__":
    from cmd_args import parse_args
    from ipdb import launch_ipdb_on_exception
    opts = parse_args()

    with launch_ipdb_on_exception():
        main(opts)
