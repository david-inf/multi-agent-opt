# Multi-Agent Optimization

Multi-agent optimization for distributed least-squares regression

<details>
<summary>Code organization</summary>

- `configs/` folder with yaml configuration files
- `plots/` folder with plotted results
- `cmd_args.py` main program arguments
- `main.py` main program
- `mydata.py` utilities for dataset creation
- `network.py` utilities for multi-agent network creation
- `train.py` utilities for agents training and sync
- `utils.py` other utilities


</details>

## Learning problem

The learning problem is least-squares regression, so it can be solved in closed form

The complication here is that being a distributed problem, there are global features that are shared between all the agents and local features that only each agent has access to, which are specific of the local data.

### Centralized problem

We start by describing the centralized problem

<details>
<summary>Problem description</summary>
Given a dataset $\mathcal{D}=\{k\mid(\psi_k,y_k),k=1,\dots,K\}$ where $\psi_k$ is the vector of features, in this example $\psi_k=(x_{1k},x_{2k})$, and $y_k$ is a scalar value representing the groundtruth.

The model for the groundtruth is
$$
y_k=w^T\psi_k+\varepsilon_k,\quad\text{where}\quad\psi_k=(x_{1k},\dots,x_{pk})\,\,\,\text{and}\,\,\,\varepsilon\sim\mathcal{N}(0,\sigma^2)
$$

A single agent solves the problem with these steps:
- $q=\sum_{k=1}^K\psi_ky_k\in\mathbb{R}^p$
- $\Omega=\sum_{k=1}^K\psi_k\psi_k^T\in\mathbb{R}^{p\times p}$
- $w^\ast=\Omega^{-1}q\in\mathbb{R}^p$
</details>

### Distributed optimization

Now we move to the distributed problem where we consider the case in which each agent has local data and also local features together with the global ones.

<details>
<summary>Problem description</summary>
So, the full dataset is splitted between each agent, each agent has a fraction of indices $K_i\subset\{1,\dots,K\}$

Multi-agent system: randomly place nodes if the distance is less than a threshold then the two agents will be connected

Local model for each agent $i$
$$
y_k^i=w^T\psi_k+\theta_i^T\varphi_k^i+\varepsilon_k
$$

The agent solves the problem locally as before:
- $q_i=\sum_{k\in K_i}[\psi_k;\phi_k^i]y_k\in\mathbb{R}^{p+p_i}$
- $\Omega_i=\sum_{k\in K_i}[\psi_k;\phi_k^i][\psi_k;\phi_k^i]^T\in\mathbb{R}^{(p+p_i)\times(p+p_i)}$
- $w_i^\ast=\Omega_i^{-1}q_i\in\mathbb{R}^{p+p_i}$
</details>
