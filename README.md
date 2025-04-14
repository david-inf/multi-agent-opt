# Multi-Agent Optimization for Distributed Learning

Multi-agent optimization for distributed least-squares regression with some real-world complications.

<details open>
<summary>Code organization</summary>

Code inside `src/` folder

- `configs/` folder with yaml configuration files
- `plots/` folder with plotted results
- `cmd_args.py` arguments for main programs
- `main.py` main program with arguments, see `python main.py --help` (for now just the configuration file)
- `mydata.py` utilities for dataset creation
- `network.py` utilities for multi-agent network creation, contains `random_nodes()` `connect_agents()` `plot_network()` functions
- `train.py` utilities for agents training and consensus algorithm, contains `Agent` class and `consensus_algorithm()` function
- `utils.py` other utilities

</details>

You can run the main program as follows (also works for `network.py` and `mydata.py` for inspecting agents network and data respectively)

```bash
python main.py --config configs/exp1.yaml
```

## :spider_web: Distributed learning

The learning problem is the least-squares regression, it can be solved in closed form.

The complication here is that being a distributed problem, there are global features that are shared between all the agents and local features that only each agent has access to, which are specific of the local data. In this example we have 2 shared features (coefficients) and a single local feature (bias).

The idea is to solve the local least-squares problems and then align all the solution with the consensus algorithm like in a federated learning setting. See [math.pdf](math.pdf) for the mathematical part (markdown has problems rendering math formulas) and pseudocode too.

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

There are various possibilities, here we consider the **ring** (named `random`, see `network.ring_nodes()`) topology where the nodes are displayed in a circle shape, and between two nodes there will be a connection if their distance is less than a threshold.
<!-- - Geometric: generate random 2D coordinates the connect two agents if their distance is below a given threshold (see `network.random_nodes()`)
- Ring: display the nodes in a circle (see `network.ring_nodes`) as the previous the threshold should be provided here too -->

For example we can set in the YAML file: `topology: random` `n_agents: 10` `dist_thresh: 3.3` that corresponds to the geometric topology with 10 agents and a distance threshold of 3.3 between two agents under which they will be connected (added to the neighbors list respectively, available in `Agent.neighbors`).

When adding agents to the neighbors list, the `Agent.update_neighbors()` method must be called, this also updates the consensus weights for the current agent. We choose to use the Metropolis weights since we only want to exploit local informations available to all agents.

Having $\mathcal{N}_i$ as the list of neighbors for the agent $i$, we can access the neighbors list as follows

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

## :chart_with_downwards_trend: Results

Here we provide results and working examples

### :one: Network with 9 agents and scenario 1

<details>
<summary>Configuration file</summary>

- `seed: 42`
- `dataset: dataset1`
- `n_samples: 12000`
- `n_agents: 9`
- `topology: random`
- `grid_size: 5`
- `dist_thresh: 2.8`
- `maxiter: 60`
- `experiment_name: data1_rand9`
- `log_every: 15`

```
python network.py --config configs/exp1.yaml
```

```
python main.py --config configs/exp1.yaml
```

</details>

Random network | Consensus error
-------------- | ----------------
![network](src/plots/random9.png) | ![error](src/plots/data1_rand9.png)

<details>
<summary>Logging</summary>

```
Agent 0, w_i=[ 0.3924 -0.8844  0.1844]
Agent 1, w_i=[ 0.5646 -1.7176  0.1032]
Agent 2, w_i=[ 0.0105 -1.6111  0.5271]
Agent 3, w_i=[ 0.8012 -1.3788  0.4398]
Agent 4, w_i=[ 0.6964 -1.433  -0.1841]
Agent 5, w_i=[ 0.5154 -1.6344  0.7982]
Agent 6, w_i=[ 0.5105 -1.5002  0.412 ]
Agent 7, w_i=[ 0.8251 -1.4701  0.4211]
Agent 8, w_i=[ 0.7572 -1.572   0.8429]
Synthetic w_i_avg=[ 0.5637 -1.4668  0.3939]

Agent 0 local solution w_i=[ 0.3911 -0.8895  0.1921], RMSE=0.80, R2=0.90
Agent 1 local solution w_i=[ 0.5621 -1.7148  0.0924], RMSE=0.80, R2=0.95
Agent 2 local solution w_i=[ 0.0121 -1.5751  0.5094], RMSE=0.80, R2=0.57
Agent 3 local solution w_i=[ 0.8016 -1.4482  0.4346], RMSE=0.77, R2=0.97
Agent 4 local solution w_i=[ 0.6936 -1.4033 -0.2047], RMSE=0.76, R2=0.97
Agent 5 local solution w_i=[ 0.5184 -1.6283  0.8378], RMSE=0.80, R2=0.94
Agent 6 local solution w_i=[ 0.5038 -1.5415  0.3927], RMSE=0.81, R2=0.93
Agent 7 local solution w_i=[ 0.827  -1.5578  0.3871], RMSE=0.80, R2=0.97
Agent 8 local solution w_i=[ 0.7585 -1.5733  0.8624], RMSE=0.81, R2=0.97
Local w_i_avg=[ 0.5631 -1.4813  0.3893] RMSE_avg=0.79 R2_avg=0.91

Iteration [1/60] cons_err=0.118168
  w_i_avg=[ 0.5763 -1.473   0.3893] RMSE_avg=1.34 R2_avg=0.37
Iteration [16/60] cons_err=0.102429
  w_i_avg=[ 0.5614 -1.447   0.3893] RMSE_avg=1.43 R2_avg=0.10                                                                                               
Iteration [31/60] cons_err=0.102381
  w_i_avg=[ 0.5607 -1.4461  0.3893] RMSE_avg=1.44 R2_avg=0.09                                                                                               
Iteration [46/60] cons_err=0.102381
  w_i_avg=[ 0.5606 -1.446   0.3893] RMSE_avg=1.44 R2_avg=0.09                                                                                               
Iteration [60/60] cons_err=0.102381
  w_i_avg=[ 0.5606 -1.446   0.3893] RMSE_avg=1.44 R2_avg=0.09

Agent 0, w_i=[ 0.5606 -1.446   0.1967], RMSE=1.32, R2=0.72
Agent 1, w_i=[ 0.5606 -1.446   0.0916], RMSE=0.82, R2=0.94
Agent 2, w_i=[ 0.5606 -1.446   0.5685], RMSE=3.27, R2=-6.27
Agent 3, w_i=[ 0.5606 -1.446   0.4015], RMSE=1.58, R2=0.89
Agent 4, w_i=[ 0.5606 -1.446  -0.2087], RMSE=1.08, R2=0.93
Agent 5, w_i=[ 0.5606 -1.446   0.8469], RMSE=0.84, R2=0.93
Agent 6, w_i=[ 0.5606 -1.446   0.3997], RMSE=0.87, R2=0.92
Agent 7, w_i=[ 0.5606 -1.446   0.3883], RMSE=1.72, R2=0.88
Agent 8, w_i=[ 0.5606 -1.446   0.868 ], RMSE=1.42, R2=0.91

w_i_avg=[ 0.5606 -1.446   0.3947] RMSE_avg=1.44 R2_avg=0.09  
```

</details>

### :two: Network with 12 agents and scenario 1

<details>
<summary>Configuration file</summary>

- `seed: 42`
- `dataset: dataset1`
- `n_samples: 20000`
- `n_agents: 12`
- `topology: random`
- `grid_size: 5`
- `dist_thresh: 2.2`
- `maxiter: 100`
- `experiment_name: data1_rand12`
- `log_every: 15`

```
python network.py --config configs/exp1.yaml
```

```
python main.py --config configs/exp1.yaml
```

</details>

Random network | Consensus error
-------------- | ----------------
![network](src/plots/random12.png) | ![error](src/plots/data1_rand12.png)

<details>
<summary>Logging</summary>

```
Agent 0, w_i=[ 0.3207 -1.589   0.1462]
Agent 1, w_i=[ 0.9927 -1.5415  0.3978]
Agent 2, w_i=[ 0.1778 -1.335   0.2539]
Agent 3, w_i=[ 0.5452 -1.6204  0.0626]
Agent 4, w_i=[ 0.4901 -1.2099  0.2077]
Agent 5, w_i=[ 0.2651 -1.1861 -0.4145]
Agent 6, w_i=[ 0.8328 -2.1729  0.2535]
Agent 7, w_i=[ 0.5711 -1.8299  0.0889]
Agent 8, w_i=[ 0.8641 -1.7585  0.2258]
Agent 9, w_i=[ 0.1717 -2.5398 -0.0461]
Agent 10, w_i=[ 1.2962 -1.1216  0.373 ]
Agent 11, w_i=[ 0.9755 -1.6745  0.4874]
Synthetic w_i_avg=[ 0.6253 -1.6316  0.1697]

Agent 0 local solution w_i=[ 0.3204 -1.5821  0.12  ], RMSE=0.82, R2=0.87
Agent 1 local solution w_i=[ 0.9894 -1.5193  0.3872], RMSE=0.83, R2=0.98
Agent 2 local solution w_i=[ 0.1836 -1.3317  0.2344], RMSE=0.80, R2=0.73
Agent 3 local solution w_i=[ 0.54   -1.6427  0.021 ], RMSE=0.75, R2=0.95
Agent 4 local solution w_i=[ 0.4871 -1.2323  0.2445], RMSE=0.81, R2=0.93
Agent 5 local solution w_i=[ 0.2662 -1.244  -0.3933], RMSE=0.79, R2=0.82
Agent 6 local solution w_i=[ 0.8307 -2.1222  0.2752], RMSE=0.83, R2=0.97
Agent 7 local solution w_i=[ 0.5727 -1.8558  0.1021], RMSE=0.77, R2=0.96
Agent 8 local solution w_i=[ 0.8691 -1.7495  0.2292], RMSE=0.79, R2=0.98
Agent 9 local solution w_i=[ 0.1768 -2.5316 -0.0764], RMSE=0.81, R2=0.83
Agent 10 local solution w_i=[ 1.3008 -1.1589  0.3739], RMSE=0.81, R2=0.99
Agent 11 local solution w_i=[ 0.9806 -1.6607  0.4965], RMSE=0.79, R2=0.98
Local w_i_avg=[ 0.6264 -1.6359  0.1679] RMSE_avg=0.80 R2_avg=0.91

Iteration [001/100] cons_err=0.077981
  w_i_avg=[ 0.6327 -1.6719  0.1679] RMSE_avg=1.83 R2_avg=0.43
Iteration [016/100] cons_err=0.061298
  w_i_avg=[ 0.6341 -1.6904  0.1679] RMSE_avg=2.00 R2_avg=0.31
Iteration [031/100] cons_err=0.056201
  w_i_avg=[ 0.6306 -1.6992  0.1679] RMSE_avg=1.99 R2_avg=0.32
Iteration [046/100] cons_err=0.053975
  w_i_avg=[ 0.6284 -1.7049  0.1679] RMSE_avg=1.99 R2_avg=0.32
Iteration [061/100] cons_err=0.053000
  w_i_avg=[ 0.6269 -1.7087  0.1679] RMSE_avg=1.99 R2_avg=0.32
Iteration [076/100] cons_err=0.052573
  w_i_avg=[ 0.6259 -1.7112  0.1679] RMSE_avg=1.99 R2_avg=0.32
Iteration [091/100] cons_err=0.052386
  w_i_avg=[ 0.6253 -1.7128  0.1679] RMSE_avg=1.99 R2_avg=0.32
Iteration [100/100] cons_err=0.052329
  w_i_avg=[ 0.625  -1.7135  0.1679] RMSE_avg=2.00 R2_avg=0.32

Agent 0, w_i=[ 0.6262 -1.7104  0.1277], RMSE=1.97, R2=0.24
Agent 1, w_i=[ 0.622  -1.7213  0.3722], RMSE=2.28, R2=0.85
Agent 2, w_i=[ 0.622  -1.7213  0.284 ], RMSE=2.65, R2=-2.02
Agent 3, w_i=[ 0.6221 -1.7208  0.0259], RMSE=0.89, R2=0.93
Agent 4, w_i=[ 0.6297 -1.7014  0.2741], RMSE=1.18, R2=0.84
Agent 5, w_i=[ 0.629  -1.7032 -0.3493], RMSE=2.21, R2=-0.43
Agent 6, w_i=[ 0.6298 -1.7013  0.3112], RMSE=1.44, R2=0.92
Agent 7, w_i=[ 0.6221 -1.7208  0.1083], RMSE=0.82, R2=0.95
Agent 8, w_i=[ 0.6232 -1.7182  0.2346], RMSE=1.63, R2=0.90
Agent 9, w_i=[ 0.622  -1.7213 -0.0395], RMSE=2.74, R2=-0.92
Agent 10, w_i=[ 0.6298 -1.7013  0.5362], RMSE=3.89, R2=0.73
Agent 11, w_i=[ 0.622  -1.7213  0.5956], RMSE=2.22, R2=0.86

w_i_avg=[ 0.625  -1.7135  0.2067] RMSE_avg=1.99 R2_avg=0.32
```

</details>
