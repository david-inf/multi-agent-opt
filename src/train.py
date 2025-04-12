
from typing import List
import numpy as np
import numpy.linalg as la

from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import r2_score as r2

from tqdm import tqdm

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
            p=2,  # number of common features
            p_i=1,  # number of agent-specific features
    ):
        self.agent_id = agent_id

        self.features = features  # local fraction
        self.targets = local_targets
        self.p = p  # for slicing operations

        # local solution: 2 (p) common and 1 (p_i) specific w_i = (w, theta_i)
        self.w_i = np.zeros(p+p_i)  # full solution
        # consensus variables for common part
        self._q_1i, self._omega_1i = [np.zeros(p), np.zeros((p, p))]
        # buffer when waiting to sync
        self._q_1i_next, self._omega_1i_next = self._q_1i.copy(), self._omega_1i.copy()

        # local parameters distribution, won't be updated
        self.mu_i, self.sigma_i = [np.zeros(p+p_i), np.zeros((p+p_i, p+p_i))]
        # same distribution that will be aligned with consensus
        self.mu_i_new, self.sigma_i_new = self.mu_i.copy(), self.sigma_i.copy()

        self.neighbors: List["Agent"] = []  # list of Agent objects
        self.degree = len(self.neighbors)  # node degree
        self.metropolis: List[float] = []  # metropolis weights list of ints

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
        self.metropolis = new_weights[::-1]  # update attribute

    def fit(self) -> None:
        """Closed-form solution for local least-squares"""
        # compute local solution (solve least-squares)
        q_i = self.targets.dot(self.features)  # [p+p_i]
        omega_i = self.features.T.dot(self.features)
        self.w_i = la.inv(omega_i).dot(q_i)  # w_i^\ast

        # training metrics
        rmse_score = rmse(self.targets, self.features.dot(self.w_i))
        r2_score = r2(self.targets, self.features.dot(self.w_i))
        with np.printoptions(precision=4):
            LOG.info(f"Agent {self.agent_id} local solution w_i={self.w_i},"
                     f" RMSE={rmse_score:.2f}, R2={r2_score:.2f}")

        # local weights distribution from the available data
        self.mu_i = la.inv(omega_i).dot(q_i)  # \mu_i i.e. w_i, stay as is
        self.mu_i_new = self.mu_i.copy()  # to be updated
        self.sigma_i = la.inv(omega_i)  # P_i, stay as is
        self.sigma_i_new = self.sigma_i.copy()  # to be updated

        # consensus init
        sigma_11i = self.sigma_i[:self.p, :self.p]
        mu_1i = self.mu_i[:self.p]
        self._q_1i = la.inv(sigma_11i).dot(mu_1i)  # [p]
        self._omega_1i = la.inv(sigma_11i)  # [p,p]

    def consensus_step(self) -> None:
        """Single consensus step on common parameters"""
        # consensus init
        pi_ii = self.metropolis[0]
        pi_ij = self.metropolis[1:]

        # this node
        self._q_1i_next = pi_ii * self._q_1i.copy()  # [p]
        self._omega_1i_next = pi_ii * self._omega_1i.copy()  # [p,p]
        # neighbors
        for j, neighbor in enumerate(self.neighbors):
            self._q_1i_next += pi_ij[j] * neighbor._q_1i
            self._omega_1i_next += pi_ij[j] * neighbor._omega_1i

    def sync(self):
        """Effective consensus step"""
        self._q_1i = self._q_1i_next.copy()
        self._omega_1i = self._omega_1i_next.copy()
        # update common distribution
        self.mu_i_new[:self.p] = la.inv(self._omega_1i_next).dot(
            self._q_1i_next)  # \mu_{1i}(l)
        self.sigma_i_new[:self.p, :self.p] = la.inv(
            self._omega_1i_next)  # P_{1i}(l)
        # update common part
        self.w_i[:self.p] = self.mu_i_new[:self.p].copy()

    def local_consensus(self):
        """Update agent-specific parameters, only one step needed"""
        sigma_11i = self.sigma_i[:self.p, :self.p]
        sigma_21i = self.sigma_i[self.p:, :self.p]
        sigma_21i_11i = sigma_21i.dot(la.inv(sigma_11i))  # [p_i,p]

        mu_1i_next_1i = self.mu_i_new[:self.p] - self.mu_i[:self.p]  # [p]

        mu_2i_next = self.mu_i[self.p:] + \
            sigma_21i_11i.dot(mu_1i_next_1i)  # [p_i]
        # update distribution
        self.mu_i_new[self.p:] = mu_2i_next.copy()
        # self.sigma_i_new = ?

        # update agent-specific part
        self.w_i[self.p:] = self.mu_i_new[self.p:].copy()

    def __str__(self) -> str:
        """String representation of the agent"""
        with np.printoptions(precision=4):
            msg = f"Agent {self.agent_id}, " \
                f"neighbors: {[neighbor.agent_id for neighbor in self.neighbors]}," \
                f" degree: {self.degree}" \
                f"\nLocal solution: {self.w_i}" \
                f"\nMetropolis weights: {np.array(self.metropolis)}"
            return msg


def consensus_algorithm(opts, agents: List[Agent]):
    """Run consensus algorithm"""
    # start with each agent solving local least-squares
    for agent in agents:
        agent.fit()

    # sync agents local solutions
    for l in range(opts.maxiter):  # l=1,...,L
        # for l in tqdm(range(maxiter), desc="Iterations", unit="it"):
        # consensus step
        for agent in agents:
            # first let all agents align
            agent.consensus_step()

        # effective step
        mean_weights = np.zeros(2)  # [p] only the common part
        for agent in agents:
            # now update variables
            agent.sync()
            mean_weights += agent.w_i[:agent.p].copy()
        mean_weights /= len(agents)

        # compute something for performance monitoring
        if l % opts.log_every == 0 or l == opts.maxiter-1:
            with np.printoptions(precision=4):
                LOG.info(f"\nIteration [{l+1}/{opts.maxiter}]")
                LOG.info(f"Mean common w={mean_weights}")
                for agent in agents:
                    LOG.info(f"Agent {agent.agent_id}, w={agent.w_i[:2]}")

    # update agent-specific parameters
    LOG.info("\nAgent-specific solutions")
    for agent in agents:
        agent.local_consensus()
        rmse_score = rmse(agent.targets, agent.features.dot(agent.w_i))
        r2_score = r2(agent.targets, agent.features.dot(agent.w_i))
        with np.printoptions(precision=4):
            LOG.info(f"Agent {agent.agent_id}, w_i={agent.w_i},"
                     f" RMSE={rmse_score:.2f}, R2={r2_score:.2f}")
