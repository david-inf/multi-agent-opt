
import numpy as np
import numpy.linalg as la

from utils import LOG


class Agent:
    """
    Class for a single agent in the network
    - each agent doesn't know how many other agents are in the network,
        the agent just knows the other agents in its neighborhood
    - 
    """

    def __init__(
            self,
            agent_id: int,
            global_features,  # array [N, p]
            local_features,  # array [N, p_i]
            local_targets,  # array [N]
    ):
        self.agent_id = agent_id

        self.global_features = global_features
        self.local_features = local_features
        self.targets = local_targets

        # local solution: 2 global and 1 local weights
        # w_i = (w, theta_i)
        self.w_i = np.random.rand(3)  # random init

        self.neighbors = []  # list of ints (agent_id)
        self.degree = len(self.neighbors)  # node degree

    def train(self):
        """Closed-form solution for local least-squares"""
        local_model = np.column_stack(
            (self.global_features, self.local_features))  # [N, p+p_i]
        q_i = local_model.T.dot(self.targets)
        omega_i = local_model.T.dot(local_model)
        w_ast_i = la.inv(omega_i).dot(q_i)
        # update local solution
        self.w_i = w_ast_i
        # return self.w_i

    def update_neighbor(self, neighbors):
        self.neighbors.extend(neighbors)
        self.degree = len(self.neighbors)

    def __str__(self):
        with np.printoptions(precision=4):
            msg = f"Agent {self.agent_id}, neighbors: {self.neighbors}," \
                  f" degree: {self.degree}" \
                  f"\nLocal weights: {self.w_i}"
            return LOG.info(msg)


class Server:
    """
    Server class that fuse informations from each agent in the network
    """

    def __init__(
            self,
            agents,
    ):
        self.agents = agents  # list of Agent objects
