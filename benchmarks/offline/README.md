# OmniSafe's Safety-Gymnasium Benchmark for Offline Algorithms

The OmniSafe Safety-Gymnasium Benchmark for offline algorithms evaluates the effectiveness of OmniSafe's offline algorithms across multiple environments from the [Safety-Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium) task suite. For each algorithm and environment supported, we provide:

- Default hyperparameters used for the benchmark and scripts to reproduce the results.
- Graphs and raw data that can be used for research purposes.
- Log details obtained during training.
- Some hints on how to fine-tune the algorithm for optimal results.

Supported algorithms are listed below:

- **[ICML 2019]** [Batch-Constrained deep Q-learning(BCQ)](https://arxiv.org/pdf/1812.02900.pdf)
- [The Lagrange version of BCQ (BCQ-Lag)](https://arxiv.org/pdf/1812.02900.pdf)
- **[NeurIPS 2020]** [Critic Regularized Regression](https://proceedings.neurips.cc//paper/2020/file/588cb956d6bbe67078f29f8de420a13d-Paper.pdf)
- [The Constrained version of CRR (C-CRR)](https://proceedings.neurips.cc/paper/2020/hash/588cb956d6bbe67078f29f8de420a13d-Abstract.html)
- **[ICLR 2022 (Spotlight)]** [COptiDICE: Offline Constrained Reinforcement Learning via Stationary Distribution Correction Estimation](https://arxiv.org/abs/2204.08957?context=cs.AI)

## Safety-Gymnasium

We highly recommend using **Safety-Gymnasium** to run the following experiments. To install, in a linux machine, type:

```bash
pip install safety_gymnasium
```

## Training agents used to generate data

```bash
omnisafe train --env-id SafetyAntVelocity-v1 --algo PPOLag
```

## Collect offline data

The `PATH_TO_AGENT` is the path of the directory containing the `torch_save`.

```python
from omnisafe.common.offline.data_collector import OfflineDataCollector


# please change agent path
env_name = 'SafetyAntVelocity-v1'
size = 1_000_000
agents = [
    ('PATH_TO_AGENT', 'epoch-500.pt', 1_000_000),
]
save_dir = './data'

if __name__ == '__main__':
    col = OfflineDataCollector(size, env_name)
    for agent, model_name, num in agents:
        col.register_agent(agent, model_name, num)
    col.collect(save_dir)
```

## Run the Benchmark

You can set the main function of ``examples/benchmarks/experiment_grid.py`` as:

```python
if __name__ == '__main__':
    eg = ExperimentGrid(exp_name='Offline-Benchmarks')

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
python analyze_experiment_results.py
```

**For a detailed usage of OmniSafe statistics tool, please refer to [this tutorial](https://omnisafe.readthedocs.io/en/latest/common/stastics_tool.html).**

## Example benchmark

### SafetyPointCircle1-v0($\beta = 0.25$)

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/offline/benchmarks/SafetyPointCircle1-v0-0.25.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">SafetyPointCircle1-v0(beta=0.25)</div>
</center>

| Algorithms | Reward (OmniSafe) | Cost (OmniSafe) |
| :--------: | :---------------: | :-------------: |
|   VAE-BC   |    43.66±0.90     |  109.86±13.24   |
|   C-CRR    |    45.48±0.87     |  127.30±12.60   |
|   BCQLag   |    43.31±0.76     |  113.39±12.81   |
| COptiDICE  |    40.68±0.93     |   67.11±13.15   |

### SafetyPointCircle1-v0($\beta = 0.50$)

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/offline/benchmarks/SafetyPointCircle1-v0-0.5.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">SafetyPointCircle1-v0(beta=0.5)</div>
</center>

| Algorithms | Reward (OmniSafe) | Cost (OmniSafe) |
| :--------: | :---------------: | :-------------: |
|   VAE-BC   |    42.84±1.36     |   62.34±14.84   |
|   C-CRR    |    45.99±1.36     |   97.20±13.57   |
|   BCQLag   |    44.68±1.97     |   95.06±33.07   |
| COptiDICE  |    39.55±1.39     |   53.87±13.27   |

### SafetyPointCircle1-v0($\beta = 0.75$)

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/offline/benchmarks/SafetyPointCircle1-v0-0.75.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">SafetyPointCircle1-v0(beta=0.75))</div>
</center>

| Algorithms | Reward (OmniSafe) | Cost (OmniSafe) |
| :--------: | :---------------: | :-------------: |
|   VAE-BC   |    40.23±0.75     |   41.25±10.12   |
|   C-CRR    |    40.66±0.88     |   49.90±10.81   |
|   BCQLag   |    42.94±1.04     |   85.37±23.41   |
| COptiDICE  |    40.98±0.89     |   70.40±12.14   |

### SafetyCarCircle1-v0($\beta = 0.25$)

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/offline/benchmarks/SafetyCarCircle1-v0-0.25.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">SafetyCarCircle1-v0(beta=0.25)</div>
</center>

| Algorithms | Reward (OmniSafe) | Cost (OmniSafe) |
| :--------: | :---------------: | :-------------: |
|   VAE-BC   |    19.62±0.28     |   150.54±7.63   |
|   C-CRR    |    18.53±0.45     |  122.63±13.14   |
|   BCQLag   |    18.88±0.61     |  125.44±15.68   |
| COptiDICE  |    17.25±0.37     |   90.86±10.75   |

### SafetyCarCircle1-v0($\beta = 0.50$)

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/offline/benchmarks/SafetyCarCircle1-v0-0.5.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">SafetyCarCircle1-v0(beta=0.5)</div>
</center>

| Algorithms | Reward (OmniSafe) | Cost (OmniSafe) |
| :--------: | :---------------: | :-------------: |
|   VAE-BC   |    18.69±0.33     |  125.97±10.36   |
|   C-CRR    |    17.24±0.43     |   89.47±11.55   |
|   BCQLag   |    18.14±0.96     |  108.07±20.70   |
| COptiDICE  |    16.38±0.43     |   70.54±12.36   |

### SafetyCarCircle1-v0($\beta = 0.75$)

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/offline/benchmarks/SafetyCarCircle1-v0-0.75.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">SafetyCarCircle1-v0(beta=0.75)</div>
</center>

| Algorithms | Reward (OmniSafe) | Cost (OmniSafe) |
| :--------: | :---------------: | :-------------: |
|   VAE-BC   |    17.31±0.33     |   85.53±11.33   |
|   C-CRR    |    15.74±0.42     |   48.38±10.31   |
|   BCQLag   |    17.10±0.84     |   77.54±14.07   |
| COptiDICE  |    15.58±0.37     |   49.42±8.699   |


## Some Insights

1. The Lagrange method is widely recognized as the simplest approach for implementing safe algorithms. However, its efficacy is limited in offline settings. This is due to the fact that the q-critic, which is based on TD-learning, cannot ensure the absolute accuracy of the q value. Since the policy relies on the q value, relative accuracy is sufficient for learning the appropriate state. However, for the Lagrange-based algorithm to acquire meaningful Lagrange multipliers, it requires q values that are absolutely accurate.
2. Currently, there exists no standardized data set for safe offline evaluation algorithms. This lack of uniformity renders the existing algorithm sensitive to the data set employed. When the data set comprises a significant number of unsafe trajectories, the algorithm struggles to learn the safe performance. We advocate for a construction method that involves training two online policies: one offering a high reward but lacking safety assurances, and the other offering a slightly lower reward but ensuring safety. The two policies are then used to collect data sets in a 1:1 ratio.
