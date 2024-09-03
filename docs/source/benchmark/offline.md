# Offline Algorithms

OmniSafe's Mujoco Velocity Benchmark evaluated the performance of OmniSafe's offline algorithm implementations in SafetyPointCirlce, SafetyPointCirlce from the Safety-Gymnasium task suite. For each algorithm and environment supported, we provide:

- Default hyperparameters used for the benchmark and scripts to reproduce the results.
- A comparison of performance or code-level details with other open-source implementations or classic papers.
- Graphs and raw data that can be used for research purposes.
- Log details obtained during training.

Supported algorithms are listed below:

- **[ICML 2019]** [Batch-Constrained deep Q-learning(BCQ)](https://arxiv.org/pdf/1812.02900.pdf)
- [The Lagrange version of BCQ (BCQ-Lag)](https://arxiv.org/pdf/1812.02900.pdf)
- **[NeurIPS 2020]** [Critic Regularized Regression](https://proceedings.neurips.cc//paper/2020/file/588cb956d6bbe67078f29f8de420a13d-Paper.pdf)
- [The Constrained version of CRR (C-CRR)](https://proceedings.neurips.cc/paper/2020/hash/588cb956d6bbe67078f29f8de420a13d-Abstract.html)
- **[ICLR 2022 (Spotlight)]** [COptiDICE: Offline Constrained Reinforcement Learning via Stationary Distribution Correction Estimation](https://arxiv.org/abs/2204.08957?context=cs.AI)

## Safety-Gymnasium

We highly recommend using ``safety-gymnasium`` to run the following experiments. To install, in a linux machine, type:

```bash
pip install safety_gymnasium
```

## Training agents used to generate data

```bash
omnisafe train --env-id SafetyAntVelocity-v1 --algo PPO
omnisafe train --env-id SafetyAntVelocity-v1 --algo PPOLag
```

## Collect offline data

```python
from omnisafe.common.offline.data_collector import OfflineDataCollector


# please change agent path
env_name = 'SafetyAntVelocity-v1'
size = 1_000_000
agents = [
    ('./runs/PPO', 'epoch-500', 500_000),
    ('./runs/PPOLag', 'epoch-500', 500_000),
]
save_dir = './data'

if __name__ == '__main__':
    col = OfflineDataCollector(size, env_name)
    for agent, model_name, num in agents:
        col.register_agent(agent, model_name, num)
    col.collect(save_dir)
```

## Run the Benchmark

You can set the main function of ``examples/benchmarks/experimrnt_grid.py`` as:

```python
if __name__ == '__main__':
    eg = ExperimentGrid(exp_name='offline-Benchmarks')

    # set up the algorithms.
    offline_policy = ['VAEBC', 'BCQ', 'BCQLag', 'CCR', 'CCRR', 'COptiDICE']

    eg.add('algo', offline_policy)

    # you can use wandb to monitor the experiment.
    eg.add('logger_cfgs:use_wandb', [False])
    # you can use tensorboard to monitor the experiment.
    eg.add('logger_cfgs:use_tensorboard', [True])
    # add dataset path
    eg.add('train_cfgs:dataset', [dataset_path])

    # set up the environment.
    eg.add('env_id', [
        'SafetyAntVelocity-v1',
        ])
    eg.add('seed', [0, 5, 10, 15, 20])

    # total experiment num must can be divided by num_pool
    # meanwhile, users should decide this value according to their machine
    eg.run(train, num_pool=5)
```

After that, you can run the following command to run the benchmark:

```bash
cd examples/benchmarks
python run_experiment_grid.py
```

You can also plot the results by running the following command:

```bash
cd examples
python plot.py --log-dir ALGODIR
```

## OmniSafe Benchmark

### Performance Table

<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  .scrollable-container {
    overflow-x: auto;
    white-space: nowrap;
    width: 100%;
  }
  table {
    border-collapse: collapse;
    width: auto;
    font-size: 12px;
  }
  th, td {
    padding: 8px;
    text-align: center;
    border: 1px solid #ddd;
  }
  th {
    font-weight: bold;
  }
  caption {
    font-size: 12px;
    font-family: 'Times New Roman', Times, serif;
  }
</style>
</head>
<body>

<div class="scrollable-container">
<table id="performance_offline">
<thead>
<tr class="header">
<th style="text-align: left;"></th>
<th colspan="2" style="text-align: center;"><strong>VAE-BC</strong></th>
<th colspan="2" style="text-align: center;"><strong>C-CRR</strong></th>
<th colspan="2" style="text-align: center;"><strong>BCQLag</strong></th>
<th colspan="2" style="text-align: center;"><strong>COptiDICE</strong></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: left;"><strong>Environment</strong></td>
<td style="text-align: center;"><strong>Reward</strong></td>
<td style="text-align: center;"><strong>Cost</strong></td>
<td style="text-align: center;"><strong>Reward</strong></td>
<td style="text-align: center;"><strong>Cost</strong></td>
<td style="text-align: center;"><strong>Reward</strong></td>
<td style="text-align: center;"><strong>Cost</strong></td>
<td style="text-align: center;"><strong>Reward</strong></td>
<td style="text-align: center;"><strong>Cost</strong></td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyPointCircle1-v0(beta=0.25)</span></td>
<td style="text-align: center;">43.66 <span class="math inline">±</span> 0.90</td>
<td style="text-align: center;">109.86 <span class="math inline">±</span> 13.24</td>
<td style="text-align: center;">45.48 <span class="math inline">±</span> 0.87</td>
<td style="text-align: center;">127.30 <span class="math inline">±</span> 12.60</td>
<td style="text-align: center;">43.31 <span class="math inline">±</span> 0.76</td>
<td style="text-align: center;">113.39 <span class="math inline">±</span> 12.81</td>
<td style="text-align: center;">40.68 <span class="math inline">±</span> 0.93</td>
<td style="text-align: center;">67.11 <span class="math inline">±</span> 13.15</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyPointCircle1-v0(beta=0.50)</span></td>
<td style="text-align: center;">42.84 <span class="math inline">±</span> 1.36</td>
<td style="text-align: center;">62.34 <span class="math inline">±</span> 14.84</td>
<td style="text-align: center;">45.99 <span class="math inline">±</span> 1.36</td>
<td style="text-align: center;">97.20 <span class="math inline">±</span> 13.57</td>
<td style="text-align: center;">44.68 <span class="math inline">±</span> 1.97</td>
<td style="text-align: center;">95.06 <span class="math inline">±</span> 33.07</td>
<td style="text-align: center;">39.55 <span class="math inline">±</span> 1.39</td>
<td style="text-align: center;">53.87 <span class="math inline">±</span> 13.27</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyPointCircle1-v0(beta=0.75)</span></td>
  <td style="text-align: center;">40.23 <span class="math inline">±</span> 0.75</td>
  <td style="text-align: center;">41.25 <span class="math inline">±</span> 10.12</td>
  <td style="text-align: center;">40.66 <span class="math inline">±</span> 0.88</td>
  <td style="text-align: center;">49.90 <span class="math inline">±</span> 10.81</td>
  <td style="text-align: center;">42.94 <span class="math inline">±</span> 1.04</td>
  <td style="text-align: center;">85.37 <span class="math inline">±</span> 23.41</td>
  <td style="text-align: center;">40.98 <span class="math inline">±</span> 0.89</td>
  <td style="text-align: center;">70.40 <span class="math inline">±</span> 12.14</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyCarCircle1-v0(beta=0.25)</span></td>
<td style="text-align: center;">19.62 <span class="math inline">±</span> 0.28</td>
<td style="text-align: center;">150.54 <span class="math inline">±</span> 7.63</td>
<td style="text-align: center;">18.53 <span class="math inline">±</span> 0.45</td>
<td style="text-align: center;">122.63 <span class="math inline">±</span> 13.14</td>
<td style="text-align: center;">18.88 <span class="math inline">±</span> 0.61</td>
<td style="text-align: center;">125.44 <span class="math inline">±</span> 15.68</td>
<td style="text-align: center;">17.25 <span class="math inline">±</span> 0.37</td>
<td style="text-align: center;">90.86 <span class="math inline">±</span> 10.75</td>
</tr>
<tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyCarCircle1-v0(beta=0.50)</span></td>
<td style="text-align: center;">18.69 <span class="math inline">±</span> 0.33</td>
<td style="text-align: center;">125.97 <span class="math inline">±</span> 10.36</td>
<td style="text-align: center;">17.24 <span class="math inline">±</span> 0.43</td>
<td style="text-align: center;">89.47 <span class="math inline">±</span> 11.55</td>
<td style="text-align: center;">18.14 <span class="math inline">±</span> 0.96</td>
<td style="text-align: center;">108.07 <span class="math inline">±</span> 20.70</td>
<td style="text-align: center;">16.38 <span class="math inline">±</span> 0.43</td>
<td style="text-align: center;">70.54 <span class="math inline">±</span> 12.36</td>
</tr>
<tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyCarCircle1-v0(beta=0.75)</span></td>
<td style="text-align: center;">17.31 <span class="math inline">±</span> 0.33</td>
<td style="text-align: center;">85.53 <span class="math inline">±</span> 11.33</td>
<td style="text-align: center;">15.74 <span class="math inline">±</span> 0.42</td>
<td style="text-align: center;">48.38 <span class="math inline">±</span> 10.31</td>
<td style="text-align: center;">17.10 <span class="math inline">±</span> 0.84</td>
<td style="text-align: center;">77.54 <span class="math inline">±</span> 14.07</td>
<td style="text-align: center;">15.58 <span class="math inline">±</span> 0.37</td>
<td style="text-align: center;">49.42 <span class="math inline">±</span> 8.70</td>
</tr>
<thead>
</table>
</div>

<caption><p><b>Table 1:</b>The performance of OmniSafe offline algorithms, which was evaluated following 1e6 training steps and under the experimental setting of cost limit=25.00. We introduce a quantization parameter beta from the perspective of safe trajectories and control the trajectory distribution of the mixed dataset. This parameter beta indicates the difficulty of this dataset to a certain extent. When beta is smaller, it means that the number of safe trajectories in the current dataset is smaller, the less safe information can be available for the algorithm to learn.</p></caption>



### Performance Curves

<img style="border-radius: 0.3125em;
box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/offline/benchmarks/SafetyPointCircle1-v0-0.25.png?raw=True">
<br>
<div>SafetyPointCircle1-v0(beta=0.25)</div>

<img style="border-radius: 0.3125em;
box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/offline/benchmarks/SafetyPointCircle1-v0-0.5.png?raw=True">
<br>
<div>SafetyPointCircle1-v0(beta=0.50)</div>

<img style="border-radius: 0.3125em;
box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/offline/benchmarks/SafetyPointCircle1-v0-0.75.png?raw=True">
<br>
<div>SafetyPointCircle1-v0(beta=0.75)</div>

<img style="border-radius: 0.3125em;
box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/offline/benchmarks/SafetyCarCircle1-v0-0.25.png?raw=true">
<br><div>SafetyCarCircle1-v0(beta=0.25)</div>

<img style="border-radius: 0.3125em;
box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/offline/benchmarks/SafetyCarCircle1-v0-0.5.png?raw=True">
<br>
<div>SafetyCarCircle1-v0(beta=0.5)</div>

<img style="border-radius: 0.3125em;
box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/offline/benchmarks/SafetyCarCircle1-v0-0.75.png?raw=True">
<br>
<div>SafetyCarCircle1-v0(beta=0.75)</div>
