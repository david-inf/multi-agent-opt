
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
            features,  # array [N, (p+p_i)]
            local_targets,  # array [N]
    ):
        self.agent_id = agent_id

        self.features = features
        self.targets = local_targets

        # local solution: 2 global and 1 local weights
        # w_i = (w, theta_i)
        self.w_i = np.random.rand(3)  # random init

        self.neighbors = []  # list of ints (agent_id)
        self.degree = len(self.neighbors)  # node degree

    def train(self):
        """Closed-form solution for local least-squares"""
        q_i = self.targets.dot(self.features)  # [p+p_i]
        omega_i = self.features.T.dot(self.features)
        self.w_i = la.inv(omega_i).dot(q_i)  # w_i^\ast

    def update_neighbor(self, neighbors):
        self.neighbors.extend(neighbors)  # add neighbors
        self.degree = len(self.neighbors)  # update node degree

    def consensus_step(self):
        # consensus algorithm single step
        # we can only fuse the common parameters
        q_1i = 

    def __str__(self):
        with np.printoptions(precision=4):
            msg = f"Agent {self.agent_id}, neighbors: {self.neighbors}," \
                  f" degree: {self.degree}" \
                  f"\nLocal weights: {self.w_i}"
            return LOG.info(msg)


# TODO: centralized Server class or update agents in a separated way?
# cioè faccio tutto centralizzato oppure lascio che gli agenti si aggiornino da soli?
class Server:
    """
    Server class that fuse informations from each agent in the network
    """

    def __init__(
            self,
            agents,  # list of Agent objects
            max_iter,  # consensus iterations
    ):
        self.agents = agents
