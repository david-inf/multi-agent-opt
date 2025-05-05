
import os
from copy import deepcopy
from typing import List
import numpy as np
import numpy.linalg as la

from tqdm import tqdm
from utils import rmse, r2, plot_metric, LOG, plot_alpha, plot_beta


class Agent:
    """
    Class for a single agent in the network
    - Each agent doesn't know how many other agents are in the network,
        the agent just knows the other agents in its neighborhood
    - Metropolis weights are added with neighbors
    - With consensus the class first stores updated variables in
        _next attributes, then one can sync
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
        self.q_1i, self.omega_1i = [np.zeros(p), np.zeros((p, p))]
        # buffer when waiting to sync
        self._q_1i_next, self._omega_1i_next = self.q_1i.copy(), self.omega_1i.copy()

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

        # local weights distribution from the available data
        self.mu_i = self.w_i.copy()  # \mu_i i.e. w_i, stay as is
        self.mu_i_new = self.w_i.copy()  # to be updated
        self.sigma_i = la.inv(omega_i)  # P_i, stay as is
        self.sigma_i_new = self.sigma_i.copy()  # to be updated

        # consensus init
        sigma_11i = self.sigma_i[:self.p, :self.p].copy()
        mu_1i = self.mu_i[:self.p].copy()
        self.q_1i = la.inv(sigma_11i).dot(mu_1i)  # [p]
        self.omega_1i = la.inv(sigma_11i)  # [p,p]

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict targets for given features"""
        return features.dot(self.w_i)

    def consensus_step(self) -> None:
        """Single consensus step on common parameters"""
        # consensus init
        pi_ii = self.metropolis[0]
        pi_ij = self.metropolis[1:]

        # this node
        self._q_1i_next = pi_ii * self.q_1i.copy()  # [p]
        self._omega_1i_next = pi_ii * self.omega_1i.copy()  # [p,p]
        # neighbors
        for j, neighbor in enumerate(self.neighbors):
            self._q_1i_next += pi_ij[j] * neighbor.q_1i.copy()
            self._omega_1i_next += pi_ij[j] * neighbor.omega_1i.copy()

    def sync(self) -> None:
        """Effective consensus step"""
        self.q_1i = self._q_1i_next.copy()
        self.omega_1i = self._omega_1i_next.copy()
        # update common distribution
        self.mu_i_new[:self.p] = la.inv(self.omega_1i).dot(
            self.q_1i)  # \mu_{1i}(l)
        self.sigma_i_new[:self.p, :self.p] = la.inv(
            self.omega_1i)  # P_{1i}(l)
        # update common parameters with updated mean
        self.w_i[:self.p] = self.mu_i_new[:self.p].copy()

    def local_consensus(self) -> None:
        """Update agent-specific parameters, only one step needed"""
        sigma_11i = la.inv(self.sigma_i_new[:self.p, :self.p])  # [p,p]
        sigma_21i = self.sigma_i[self.p:, :self.p].copy()  # [p_i,p]
        sigma_21i_11i_mult = sigma_21i.dot(sigma_11i)  # [p_i,p]
        mu_new_old_sub = self.mu_i_new[:self.p] - self.mu_i[:self.p]  # [p]

        mu_2i_next = self.mu_i[self.p:].copy() + \
            sigma_21i_11i_mult.dot(mu_new_old_sub)  # [p_i]
        # update distribution
        self.mu_i_new[self.p:] = mu_2i_next.copy()
        # self.sigma_i_new = ?

        # update agent-specific part
        self.w_i[self.p:] = self.mu_i_new[self.p:].copy()

    def __str__(self) -> str:
        """String representation of the agent"""
        with np.printoptions(precision=4):
            msg = f"Agent {self.agent_id}, " \
                f"neighbors={[neighbor.agent_id for neighbor in self.neighbors]}," \
                f" degree={self.degree}" \
                f"\nLocal solution={self.w_i}" \
                f"\nMetropolis weights={np.array(self.metropolis)}"
            return msg


def consensus_algorithm(opts, agents: List[Agent], gt):
    """Run consensus algorithm"""
    LOG.info("Local least-squares:")
    params0 = []
    for agent in agents:
        # solve local least-squares problem for each agent
        agent.fit()
        params0.append(agent.w_i.copy())
        # check fit metrics
        _training_metrics(agent)
    err0 = _consensus_error(agents)

    errs_iters = [err0]  # list of floats
    params_agents_iters = [params0]  # [L, N, 3]
    with tqdm(range(opts.maxiter), desc="Consensus", unit="it") as titers:
        for _ in titers:
            # single consensus step on common and local parameters
            cons_err, params_agents = _make_consensus(agents)

            # metrics
            errs_iters.append(cons_err)
            params_agents_iters.append(deepcopy(params_agents))
            titers.set_postfix(cons_err=cons_err)
            titers.update()

    LOG.info("After consensus:")
    for agent in agents:
        # check fit after consensus
        _training_metrics(agent)

    output_dir = os.path.join("src/plots", opts.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    # 1) Plot consensus error
    output_path = os.path.join(output_dir, opts.experiment_name+"_iters.svg")
    plot_metric(errs_iters, output_path)

    # 2) Plot parameters convergence
    vals_tensor = np.array(params_agents_iters)  # [iters, n_agents, params]
    labels = [r"$\alpha_1$", r"$\alpha_2$", r"$|\beta_i-\beta^\ast|$"]
    fnames = [f"{opts.experiment_name}_{coeff}.svg" for coeff in [
        "alpha1", "alpha2", "beta"]]
    paths = [os.path.join(output_dir, fname) for fname in fnames]

    plot_alpha(vals_tensor[:, :, 0], labels[0],
               "Coefficient convergence", gt[0], paths[0])
    plot_alpha(vals_tensor[:, :, 1], labels[1],
               "Coefficient convergence", gt[1], paths[1])
    plot_beta(vals_tensor[:, :, 2], labels[2], "Coefficient convergence",
              gt[2], paths[2])


def _make_consensus(agents: List[Agent]):
    """Consensus step for each agent"""
    for agent in agents:
        # let all agents make a step
        # computed updates are store in _next variables
        agent.consensus_step()

    # check training metrics
    params_agents = []  # list of np.ndarray
    for agent in agents:
        # update actual variables with values saved in _next
        agent.sync()
        params_agents.append(agent.w_i)
        # update local parameters
        agent.local_consensus()

    cons_err = _consensus_error(agents)

    return cons_err, params_agents


def _training_metrics(agent: Agent):
    """Print training metrics for a single agent"""
    pred = agent.predict(agent.features)  # [N]
    rmse_score = rmse(agent.targets, pred)
    r2_score = r2(agent.targets, pred)

    with np.printoptions(precision=4):
        LOG.info("Agent %d local solution w_i=%s, RMSE=%.2f, R2=%.2f",
                 agent.agent_id, agent.w_i, rmse_score, r2_score)


def _consensus_error(agents: List[Agent]):
    """Compute consensus error for given list of agents"""
    w_common = []
    for agent in agents:
        # add common parameters
        w_common.append(agent.w_i[:agent.p])

    # Consensus error: agreement on shared parameters
    w_avg = np.mean(w_common, axis=0)  # [p]
    _err_per_i = np.array(w_common) - w_avg  # [N,p]
    cons_err = np.mean(la.norm(_err_per_i, axis=1) ** 2)

    return cons_err
