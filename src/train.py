"""
Few notes:
- each agent doesn't know how many other agents are in the network, the agent just knows the other agents in its neighborhood
"""

import numpy as np


class Agent:
    def __init__(self, agent_id, global_data=None, local_data=None):
        self.agent_id = agent_id  # int

        self.global_data = global_data  # tuple
        self.local_data = local_data  # scalar

        # local solution 2 global and 1 local weights
        self.w_i = np.random.rand(3)

        self.neighbors = []  # list of ints (agent_id)
        self.degree = len(self.neighbors)  # node degree

    # def train(self):
        """Closed-form solution for local least-squares"""

    def update_neighbor(self, neighbors):
        self.neighbors.extend(neighbors)
        self.degree = len(self.neighbors)

    def __str__(self):
        msg = f"Agent {self.agent_id}, neighbors {self.neighbors}," \
            f" degree {self.degree}"
        return msg
