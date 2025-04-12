# Multi-Agent Optimization for Distributed Learning

Multi-agent optimization for distributed least-squares regression with some real-world complications.

- `configs/` folder with yaml configuration files
- `plots/` folder with plotted results
- `cmd_args.py` arguments for main programs
- `main.py` main program with arguments, see `python main.py --help` (for now just the configuration file)
- `mydata.py` utilities for dataset creation
- `network.py` utilities for multi-agent network creation, contains `random_nodes()` `connect_agents()` `plot_network()` functions
- `train.py` utilities for agents training and consensus algorithm, contains `Agent` class and `consensus_algorithm()` function
- `utils.py` other utilities

You can run the main program as follows (also works for `network.py` and `mydata.py` for inspecting agents network and data respectively)

```bash
python main.py --config configs/exp1.yaml
```

## Distributed learning

The learning problem is the least-squares regression, it can be solved in closed form.

The complication here is that being a distributed problem, there are global features that are shared between all the agents and local features that only each agent has access to, which are specific of the local data. In this example we have 2 shared features (coefficients) and a single local feature (bias).

The idea is to solve the local least-squares problems and then align all the solution with the consensus algorithm like in a federated learning setting. See [math.pdf](math.pdf) for the mathematical part (markdown has problems rendering math formulas).

<!-- ### :dart: Centralized problem

We start by describing the centralized problem (which is not directly implemented).

<details>
<summary>Problem description</summary>

Given a dataset $\mathcal{D}=\bigl\{k\mid(\psi_k,y_k),k=1,\dots,K\bigr\}$ where $\psi_k$ is the vector of features, in this example $\psi_k=(x_{1k},x_{2k},1)\in\mathbb{R}^p$, and $y_k$ is a scalar value representing the groundtruth.

The model for the groundtruth is
$$
y_k=w^T\psi_k+\varepsilon_k,\quad\text{where}\quad
\psi_k=(x_{1k},\dots,x_{pk})\,\,\,\text{and}\,\,\,
\varepsilon_k\sim\mathcal{N}(0,\sigma^2)
$$

A single agent solves the problem with these steps:
1. $q=\sum_{k=1}^K\psi_ky_k\in\mathbb{R}^p$
2. $\Omega=\sum_{k=1}^K\psi_k\psi_k^T\in\mathbb{R}^{p\times p}$
3. $w^\ast=\Omega^{-1}q\in\mathbb{R}^p$ unique optimal solution

This has the assumption that $w\sim p(w)=\mathcal{N}(\mu,P)$ with $\mu=\Omega^{-1}q$ and $P=\Omega^{-1}$.

</details> -->

<!-- ### :busts_in_silhouette: Distributed problem

Now we move to the distributed problem where we consider the case in which each agent has a fraction of the global features and its own specific local features. You may find the utilities in the `train.py` file inside the `Agent` class.

<details open>
<summary>Local problem description</summary>

So, the full dataset is splitted between each agent, each agent has a fraction of indices $\mathcal{K}_i\subset\{1,\dots,K\}$. The dataset for each agent will be $\mathcal{K}_i=\bigl\{k\mid([\psi_k;\varphi_k^i],y_k)\bigr\}$, in this example $[\psi_k;\varphi_k^i]=(x_{1k},x_{2k},1)$ (each agent has the bias parameter).

Local model for each agent $i$
$$
y_k^i=w^T\psi_k+\theta_i^T\varphi_k^i+\varepsilon_k,\quad\text{where}\quad
\begin{array}{c}
\psi_k=(x_{1k},\dots,x_{pk}) \\[1ex]
\varphi_k^i=(x_{1k}^i,\dots,x_{p_ik}^i)
\end{array}\text{and}\,\,\,
\varepsilon_k\sim\mathcal{N}(0,\sigma^2)
$$

The agent $i$ solves the problem locally as before:
1. $q_i=\sum_{k\in\mathcal{K}_i}[\psi_k;\phi_k^i]y_k\in\mathbb{R}^{p+p_i}$
2. $\Omega_i=\sum_{k\in\mathcal{K}_i}[\psi_k;\phi_k^i][\psi_k;\phi_k^i]^T\in\mathbb{R}^{(p+p_i)\times(p+p_i)}$
3. $w_i^\ast=\Omega_i^{-1}q_i\in\mathbb{R}^{p+p_i}$ where $w_i=[w;\theta_i]$

The assumption here is $w_i\sim p(w_i)=\mathcal{N}(\mu_i,P_i)$ where $\mu_i=\Omega_i^{-1}q_i$ and $P_i=\Omega_i^{-1}$ as before. This is done with the `fit()` method from the `Agent` class:

```python
agents = [Agent(1, features_1, targets_1), Agent(2, features_2, targets_2)]
for agent in agents:
    agent.fit()
    # automatically prints local solution w_i
    # then RMSE and R2 metrics using sklearn
```

</details> -->

### :file_folder: Custom dataset

We explore two dataset scenarios

Scenario 1 | Scenario 2
---------- | ----------
$x_{1k}\sim\mathcal{U}([-10,10])$ | $x_{2k}\sim\mathcal{U}([-1,1])$
$x_{1k}\sim\mathcal{U}([-1,1])$ | $x_{2k}\sim\mathcal{U}([-10,10])$ 
$\varepsilon_k\sim\mathcal{N}(0,0.8)$ | $\varepsilon_k\sim\mathcal{N}(0,0.8)$

In both scenarios we generate the parameters in the same way, we may want to see the effect of changing the covariates data generating process.

### :busts_in_silhouette: Multi-agent system

We need to define a network topology and specify the number of agents, based on the topology there could be more parameters to set.

There are various possibilities, here we consider these topologies:
- Geometric: generate random 2D coordinates the connect two agents if their distance is below a given threshold (see `network.random_nodes()`)
- Ring: display the nodes in a circle (see `network.ring_nodes`) as the previous the threshold should be provided here too

For example we can set in the YAML file: `topology: random` `n_agents: 10` `dist_thresh: 3.3` that corresponds to the geometric topology with 10 agents and a distance threshold of 3.3 between two agents under which they will be connected (added to the neighbors list respectively, available in `Agent.neighbors`).

When adding agents to the neighbors list, the `.update_neighbors()` method must be called, this also updates the consensus weights for the current agent. We choose to use the Metropolis weights since we only want to exploit local informations available to all agents.

Having $\mathcal{N}_i$ as the list of neighbors for the agent $i$, we can access the list as follows

```python
agent_i = Agent(1, features_i, targets_i)
# get distance matric for the network topology
# connect agents given the network with network.connect_agents()
for neighbor in agent.neighbors:
    print(neighbor)  # see `__str__` method
    print(neighbor.metropolis)  # directly access metropolis weights
```

### :mailbox_with_mail: Consensus algorithm

The full dataset is split between each agent, i.e. each agent has a fraction of indices, this is handled with utilities in `mydata.py` when calling `get_dataset()` and using the `dataset_fun()` output from the main program. The output `agent_splits` is a list of dictionary where each containts local features and targets for each agent.

Once we have the data, we can proceed with solving the local least-squares problem as follows, where the `.fit()` method initializes variables for the consesus algorithm too.

```python
agents = [Agent(1, features_1, targets_1), Agent(2, features_2, targets_2)]
# get distance matrix for the network topology
# connect agents with `connect_agents()` updating neighbors list
for agent in agents:
    agent.fit()
    # automatically prints local solution w_i
    # then RMSE and R2 metrics using sklearn
```

<!-- <details open>
<summary>Consensus algorithm</summary>

Now let's dive deep into the algorithm that allows to align the local solution for each agent. The complexity here is that each agent solution has a common part and an agent-specific part.

##### Bayesian stuffs

We can decompose $p(w_i)=\mathcal{N}(\mu_i,P_i)$ as follows exploiting the Bayesian interpretation for the prior

$$
p(w_i)=p(w,\theta_i)=p(\theta_i\lvert w)p(w)\quad\text{with}\quad
\mu_i=
\begin{bmatrix}
\mu_{1i} \\ \mu_{2i}
\end{bmatrix}\,\,\,\text{and}\,\,\,
P_i=
\begin{bmatrix}
P_{11i} & P_{12i} \\ P_{21i} & P_{22i}
\end{bmatrix}
$$

Staying with the Gaussian prior we can obtain the exact form of the decomposed joint distribution

$$
\begin{array}{c}
p(w)\sim\mathcal{N}(\mu_{1i}, P_{11i}) \\[2ex]
p(\theta_i\lvert w)\sim\mathcal{N}(\mu_{2\lvert1i}, P_{2\lvert1i})
\end{array}\qquad
\begin{array}{c}
\mu_{2\lvert1i}=\mu_{2i}+P_{21i}P_{11i}^{-1}(w-\mu_{1i})
\\[2ex]
P_{2\lvert1i}=P_{22i}-P_{21i}P_{11i}^{-1}P_{12i}
\end{array}
$$

##### Metropolis weights

Since we only want to use local information available to all agents, we use the Metropolis weights defined as follows:

$$
\pi_{ij}=
\begin{cases}
1-\sum_{j\in\mathcal{N}_i}\pi_{ij} & \text{if $j=i$} \\
\frac{1}{1+\max\{d_i,d_j\}} & \text{if $j\in\mathcal{N}_i$} \\
0 & \text{otherwise}
\end{cases}
$$

Having $\mathcal{N}_i$ as the list of neightbors for the agent $i$ available that can be accessed with
```python
agent_i = Agent(1, features_i, targets_i)
# connect agents given the network with network.connect_agents()
for neighbor in agent.neighbors:
    print(neighbor)
```

##### Fusing the common part

Once we have the local solution for each agent (`agent.fit()`), we may proceed with the consensus algorithm for the common part of the weights, starting with

$$
q_{1i}(0)=P_{11i}^{-1}\mu_{1i}\quad\text{and}\quad
\Omega_{1i}(0)=P_{11i}^{-1}
$$

We give the consensus steps $L$ in the YAML configuration file as `maxiter` then `for l=1,...,L`
1. $q_{1i}(l+1)=\pi_{ii}q_{1i}(l)+\sum_{j\in\mathcal{N}_i}\pi_{ij}q_{1j}(l)$
2. $\Omega_{1i}(l+1)=\pi_{ii}\Omega_{1i}(l)+\sum_{j\in\mathcal{N}_i}\pi_{ij}\Omega_{1j}(l)$
3. $w_{1i}(l+1)=\mu_{1i}(l+1)=\bigl[\Omega_{1i}(l+1)\bigr]^{-1}q_{1i}(l+1)$

This will yield $q_{1i}(L)$, $\Omega_{1i}(L)$ and then $w_{1i}(L)$ from the function `train.consensus_algorithm()`:

```python
for l in range(opts.maxiter):
    # single consensus step for each agent
    for agent in agents:
        agent.consensus_step()
        # updates `q_1i_next` `omega_1i_next` (buffer)
    # update consensus variables effectively
    for agent in agents:
        agent.sync()
        # updates `q_1i` `omega_1i` `w_i`
```

Eventually we can update the agent-specific parameters having the common part updated after consensus

$$
\mu_{2i}(L)=\mu_{2i}+P_{21i}P_{11i}^{-1}\bigl(\mu_{1i}(L)-\mu_{1i}\bigr)
$$
where $\mu_{2i}$ is the mean from the local data distribution (`agent.mu_i` and `agent.sigma_i`), so this will results in the global distribution (`agent.mu_i_new` and `agent.sigma_i_new`).

```python
for agent in agents:
    # updates local bias stored in agent.mu_i_new[-1]
    agent.local_consensus()
```

</details> -->

Once we have the local solution for all agents (`agent.fit()`), we may proceed with the consensus algorithm for the common part of the weights, starting with

```python
for l in range(opts.maxiter):
    for agent in agents:
        # single consensus step for each agent
        # stores solution elsewhere util all agents make a step
        agent.consensus_step()
        # updates `q_1i_next` `omega_1i_next` (buffer)
    for agent in agents:
        # update consensus variables effectively
        agent.sync()
        # updates `q_1i` `omega_1i` `w_i`
```

Eventually we can update the agent-specific parameters (just bias here) having the common part updated after consensus

```python
for agent in agents:
    # updates local bias stored in agent.mu_i_new[-1]
    agent.local_consensus()
```

## Results


