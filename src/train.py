
from typing import List
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
            features: np.ndarray,  # array [N, (p+p_i)]
            local_targets: np.ndarray,  # array [N]
    ):
        self.agent_id = agent_id

        self.features = features
        self.targets = local_targets

        # local solution: 2 global and 1 local weights
        self.w_i = np.random.rand(3)  # w_i = (w, theta_i)

        # consensus steps for common weights [q(l),Omega(l),w(l)]
        self.q_1i, self.omega_1i, self._w1i = None, None, None
        # buffer waiting to sync
        self.q_1i_next, self.omega_1i_next, self.w_1i_next = None, None, None

        self.neighbors: List["Agent"] = []  # list of Agent objects
        self.degree = len(self.neighbors)  # node degree
        self.weights: List[float] = []  # metropolis weights list of ints

    def train(self) -> None:
        """Closed-form solution for local least-squares"""
        # compute local solution (solve least-squares)
        q_i = self.targets.dot(self.features)  # [p+p_i]
        omega_i = self.features.T.dot(self.features)
        self.w_i = la.inv(omega_i).dot(q_i)  # w_i^\ast

        # local weights distribution
        mu_i = la.inv(omega_i).dot(q_i)
        sigma_i = la.inv(omega_i)

        # common weights distribution
        sigma_11i = sigma_i[:2, :2]  # [p,p]
        mu_1i = mu_i[:2]  # [p]
        # q(0) and Omega(0) for consensus algorithm
        self.q_1i = la.inv(sigma_11i).dot(mu_1i)
        self.omega_1i = la.inv(sigma_11i)

    def update_neighbors(self, neighbors: List["Agent"]) -> None:
        """Update neighbors list with new Agent objects"""
        self.neighbors.extend(neighbors)  # add neighbors
        self.degree = len(self.neighbors)  # update node degree

        new_weights = []  # update metropolis weights
        for neighbor in self.neighbors:
            weight_ij = 1 / (1 + max(self.degree, neighbor.degree))
            new_weights.append(weight_ij)  # first neighbors weights
        new_weights.append(1 - sum(new_weights))  # then the node weight
        # reverse weights order: first this node then neighbors weights
        self.weights = new_weights.reverse()  # update attribute

    def consensus_step(self) -> None:
        """Update parameters according to neighbors"""
        # metropolis weights
        pi_ii = self.weights[0]  # float
        pi_ij = self.weights[1:]  # list of floats

        self.q_1i_next = pi_ii * self.q_1i
        self.omega_1i_next = pi_ii * self.omega_1i

        for j, agent in enumerate(self.neighbors):
            self.q_1i_next += pi_ij[j] * agent.q_1i
            self.omega_1i_next += pi_ij[j] * agent.omega_1i

        self.w_i_next = la.inv(self.omega_1i_next).dot(self.q_1i_next)

    def sync(self):
        """Effective consensus step"""
        self.q_1i = self.q_1i_next.copy()
        self.omega_1i = self.omega_1i_next.copy()
        self.w_i = self.w_i_next.copy()
        # reset weights buffer
        self.q_1i_next, self.omega_1i_next, self.w_i_next = None, None, None

    def __str__(self) -> str:
        """String representation of the agent"""
        with np.printoptions(precision=4):
            msg = f"Agent {self.agent_id}, neighbors: {self.neighbors}," \
                f" degree: {self.degree}" \
                f"\nLocal weights: {self.w_i}"
            return LOG.info(msg)


def consensus_algorithm(agents: List[Agent], maxiter):
    """Run consensus algorithm"""
    # start with each agent solving local least-squares
    for agent in agents:
        agent.train()

    # sync agents local solutions
    for l in range(maxiter):  # l=1,...,L
        # consensus step
        for agent in agents:
            agent.consensus_step()
        # effective step
        for agent in agents:
            agent.sync()
        # compute something for performance monitoring