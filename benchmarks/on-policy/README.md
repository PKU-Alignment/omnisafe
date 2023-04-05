# OmniSafe's Mujoco Velocity Benchmark on On-Policy Algorithms

OmniSafe's Mujoco Velocity Benchmark evaluated the performance of OmniSafe algorithm implementations in 6 environments from the Safety-Gymnasium task suite For each algorithm and environment supported, we provide:
- Default hyperparameters used for the benchmark and scripts to reproduce the results
- A comparison of performance or code-level details with other open-source implementations or classic papers.
- Graphs and raw data that can be used for research purposes, - Log details obtained during training
- Some hints on how to fine-tune the algorithm for optimal results.

Supported algorithms are listed below:

**First-Order**

- **[NIPS 1999]** [Policy Gradient(PG)](https://papers.nips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)
- [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf)
- [The Lagrange version of PPO (PPO-Lag)](https://cdn.openai.com/safexp-short.pdf)
- **[IJCAI 2022]** [Penalized Proximal Policy Optimization for Safe Reinforcement Learning(P3O)]( https://arxiv.org/pdf/2205.11814.pdf)
- **[NeurIPS 2020]** [First Order Constrained Optimization in Policy Space (FOCOPS)](https://arxiv.org/abs/2002.06506)
- **[NeurIPS 2022]**  [Constrained Update Projection Approach to Safe Policy Optimization (CUP)](https://arxiv.org/abs/2209.07089)

**Second-Order**

- **[NeurIPS 2001]** [A Natural Policy Gradient (NaturalPG))](https://proceedings.neurips.cc/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf)
- **[PMLR 2015]** [Trust Region Policy Optimization (TRPO)](https://arxiv.org/abs/1502.05477)
- [The Lagrange version of TRPO (TRPO-Lag)](https://cdn.openai.com/safexp-short.pdf)
- **[ICML 2017]** [Constrained Policy Optimization (CPO)](https://proceedings.mlr.press/v70/achiam17a)
- **[ICML 2017]** [Proximal Constrained Policy Optimization (PCPO)](https://proceedings.mlr.press/v70/achiam17a)
- **[ICLR 2019]** [Reward Constrained Policy Optimization (RCPO)](https://openreview.net/forum?id=SkfrvsA9FX)

## Safety-Gymnasium

We highly recommend using ``safety-gymnasium`` to run the following experiments. To install, in a linux machine, type:

```bash
pip install safety_gymnasium
```

## Run the Benchmark

You can set the main function of ``examples/benchmarks/experimrnt_grid.py`` as:

```python
if __name__ == '__main__':
    eg = ExperimentGrid(exp_name='On-Policy-Benchmarks')

    # set up the algorithms.
    base_policy = ['PolicyGradient', 'NaturalPG', 'TRPO', 'PPO']
    naive_lagrange_policy = ['PPOLag', 'TRPOLag', 'RCPO', 'OnCRPO', 'PDO']
    first_order_policy = ['CUP', 'FOCOPS', 'P3O']
    second_order_policy = ['CPO', 'PCPO']

    eg.add('algo', base_policy + naive_lagrange_policy + first_order_policy + second_order_policy)

    # you can use wandb to monitor the experiment.
    eg.add('logger_cfgs:use_wandb', [False])
    # you can use tensorboard to monitor the experiment.
    eg.add('logger_cfgs:use_tensorboard', [True])

    # set up the environment.
    eg.add('env_id', [
        'SafetyHopperVelocity-v1',
        'SafetyWalker2dVelocity-v1',
        'SafetySwimmerVelocity-v1',
        'SafetyAntVelocity-v1',
        'SafetyHalfCheetahVelocity-v1',
        'SafetyHumanoidVelocity-v1'
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

e.g. ALGODIR can be ``examples/runs/SafetyHopperVelocity-v1``.
Then you can compare different algorithms in ``SafetyHopperVelocity-v1`` environments.

Logs is saved in `examples/benchmarks/runs` and can be monitored with tensorboard or wandb.

```bash
$ tensorboard --logdir examples/benchmarks/runs
```

After the experiment is finished, you can use the following command to generate the video of the trained agent:

```bash
cd examples
python evaluate_saved_policy.py
```
Please note that before you evaluate, please set the ``LOG_DIR`` in ``evaluate_saved_policy.py``.

For example, if I train ``PPOLag`` in ``SafetyHumanoidVelocity-v1``

```python
    LOG_DIR = '~/omnisafe/examples/runs/PPOLag-<SafetyHumanoidVelocity-v1>/seed-000-2023-03-07-20-25-48'
    play = True
    save_replay = True
    if __name__ == '__main__':
        evaluator = omnisafe.Evaluator(play=play, save_replay=save_replay)
        for item in os.scandir(os.path.join(LOG_DIR, 'torch_save')):
            if item.is_file() and item.name.split('.')[-1] == 'pt':
                evaluator.load_saved(
                    save_dir=LOG_DIR, model_name=item.name, camera_name='track', width=256, height=256
                )
                evaluator.render(num_episodes=1)
                evaluator.evaluate(num_episodes=1)
```

## Example benchmark

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/first_order_ant.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">SafetyAntVelocity-v1</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/second_order_ant.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">SafetyAntVelocity-v1</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/first_order_halfcheetah.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">SafetyHalfCheetahVelocity-v1</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/second_order_halfcheetah.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">SafetyHalfCheetahVelocity-v1</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/first_order_hopper.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">SafetyHopperVelocity-v1</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/second_order_hopper.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">SafetyHopperVelocity-v1</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/first_order_humanoid.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">SafetyHumanoidVelocity-v1</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/second_order_humanoid.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">SafetyHumanoidVelocity-v1</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/first_order_walker2d.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">SafetyWalker2dVelocity-v1</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/second_order_walker2d.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">SafetyWalker2dVelocity-v1</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/first_order_swimmer.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">SafetySwimmerVelocity-v1</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/second_order_swimmer.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">SafetySwimmerVelocity-v1</div>
</center>


## Experiment Analysis

### Hyperparameters

#### First-Order Methods Specific Hyperparameters

**We are continuously improving performance for first-order algorithms and finding better hyperparameters and will release an ultimate version as soon as possible. Meanwhile, we are happy to receive any advice from users, feel free for opening PRs or issues.**

#### Second-Order Methods Specific Hyperparameters

- ``algo_cfgs:kl_early_stop``: Whether to use early stop for KL divergence. In the second-order methods, we use line search to find the proper step size. If the KL divergence is too large, we will stop the line search and use the previous step size. So we always set this hyperparameter to ``False``.

- ``model_cfgs:actor:lr``: The learning rate of the actor network. The second-order methods use the actor network update the policy by directly setting the parameters of the policy network. So we do not need to set the learning rate of the policy network, which is set to ``None``.

### Some Hints

In our experiments, we found that somehyperparameters are important for the performance of the algorithm:

- ``obs_normlize``: Whether to normalize the observation.
- ``rew_normlize``: Whether to normalize the reward.
- ``cost_normlize``: Whether to normalize the cost.

We have done some experiments to show the effect of these hyperparameters, and we log the best configuration for each algorithm in each environment. You can check it in the ``omnisafe/configs/on_policy``.

In experiments, we found that the ``obs_normlize=True`` always performs better than ``obs_normlize=False`` in the second-order methods. That means the reward would increase quicker if we normalize the observation. So we set ``obs_normlize=True`` in almost all the second-order methods.

Importantly, we found that the ``rew_normlize=True`` not always performs better than ``rew_normlize=False``, especially in the ``SafetyHopperVelocity-v1`` and ``SafetyWalker2dVelocity`` environment.

The Lagrangian method often has the phenomenon of unstable update and easy overshoot。 We found that with ``cost_normlize=True``, the Lagrangian method can update more stable and avoid overshoot to some degree. So we set ``cost_normlize=True`` in the Lagrangian method.

Besides, the hyperparameter ``train_cfgs:torch_num_threads`` is also important. on-policy algorithms always use more time to update policy than to sample data. So we use ``torch_num_threads`` to speed up the update process.

This hyperparamter depens on the number of CPU cores. We set it to 8 in our experiments. You can set it to some other porper value according to your CPU cores.

If you find that other hyperparameters perform better, please feel free to open an issue or pull request.

### First-Order Algorithms Experiment Results

### PG(1M)

|         Environment          | Reward (OmniSafe) | Cost (Omnisafe) |
| :--------------------------: | :---------------: | :-------------: |
|     SafetyAntVelocity-v1     |   1128.4±654.6    |   155.0±96.5    |
| SafetyHalfCheetahVelocity-v1 |   1700.2±902.4    |   422.2±234.1   |
|   SafetyHopperVelocity-v1    |    674.4±127.2    |   180.5±26.4    |
|  SafetyWalker2dVelocity-v1   |    624.2±301.4    |   125.8±67.5    |
|   SafetySwimmerVelocity-v1   |     37.7±8.2      |   695.0±230.3   |
|  SafetyHumanoidVelocity-v1   |    612.7±131.6    |    38.9±17.8    |

### PPO(1M)

|         Environment          | Reward (OmniSafe) | Cost (Omnisafe) |
| :--------------------------: | :---------------: | :-------------: |
|     SafetyAntVelocity-v1     |   3012.2±1167.0   |   618.3±255.0   |
| SafetyHalfCheetahVelocity-v1 |   3641.1±1202.3   |   812.8±219.1   |
|   SafetyHopperVelocity-v1    |    685.2±132.8    |   170.2±25.7    |
|  SafetyWalker2dVelocity-v1   |    723.0±175.3    |   141.0±30.8    |
|   SafetySwimmerVelocity-v1   |     52.4±19.9     |   472.9±300.3   |
|  SafetyHumanoidVelocity-v1   |    633.3±128.7    |    45.9±16.1    |

### PPOLag(1M)

|         Environment          | Reward (OmniSafe) | Cost (Omnisafe) |
| :--------------------------: | :---------------: | :-------------: |
|     SafetyAntVelocity-v1     |   2256.6±315.1    |    29.8±54.7    |
| SafetyHalfCheetahVelocity-v1 |   2065.5±234.5    |     4.7±5.1     |
|   SafetyHopperVelocity-v1    |    415.8±367.9    |    47.2±28.4    |
|  SafetyWalker2dVelocity-v1   |    310.4±44.7     |    19.9±9.9     |
|   SafetySwimmerVelocity-v1   |     22.0± 7.8     |    63.2±16.3    |
|  SafetyHumanoidVelocity-v1   |    623.0±173.7    |    17.0±19.7    |

### P3O(1M)

|         Environment          | Reward (OmniSafe) | Cost (Omnisafe) |
| :--------------------------: | :---------------: | :-------------: |
|     SafetyAntVelocity-v1     |   1837.5±331.2    |    35.5±28.2    |
| SafetyHalfCheetahVelocity-v1 |   1251.2±117.4    |    14.7±15.3    |
|   SafetyHopperVelocity-v1    |    779.0±383.2    |    21.4±13.9    |
|  SafetyWalker2dVelocity-v1   |   1493.1±515.5    |    27.9±26.7    |
|   SafetySwimmerVelocity-v1   |     -8.8±14.3     |   125.0±58.5    |
|  SafetyHumanoidVelocity-v1   |   1027.3±404.7    |     0.4±2.0     |

### FOCOPS(1M)

|         Environment          | Reward (OmniSafe) | Cost (Omnisafe) |
| :--------------------------: | :---------------: | :-------------: |
|     SafetyAntVelocity-v1     |   2022.0±226.6    |     4.5±5.6     |
| SafetyHalfCheetahVelocity-v1 |   1759.8±414.4    |    31.3±55.2    |
|   SafetyHopperVelocity-v1    |    255.4±190.0    |    10.2±12.4    |
|  SafetyWalker2dVelocity-v1   |    346.3±100.2    |    22.1±16.1    |
|   SafetySwimmerVelocity-v1   |     9.0±17.1      |   86.6 ±80.8    |
|  SafetyHumanoidVelocity-v1   |    703.5±188.0    |    14.4±16.5    |

### CUP(1M)

|         Environment          | Reward (OmniSafe) | Cost (Omnisafe) |
| :--------------------------: | :---------------: | :-------------: |
|     SafetyAntVelocity-v1     |   1530.4±723.3    |    60.6±48.4    |
| SafetyHalfCheetahVelocity-v1 |   1217.6±288.0    |    15.2±14.6    |
|   SafetyHopperVelocity-v1    |    249.8±308.5    |    32.2±21.1    |
|  SafetyWalker2dVelocity-v1   |    673.3±608.6    |    22.2±21.6    |
|   SafetySwimmerVelocity-v1   |     1.2±19.3      |   113.9±57.0    |
|  SafetyHumanoidVelocity-v1   |    535.0±78.2     |    16.3±13.6    |



### Second-Order Algorithms Experiment Results

#### NaturalPG(1M)

| Environment | Reward （OmniSafe）| Cost （OmniSafe） |
| :---------: | :-----------: |  :-----------: |
|     SafetyAntVelocity-v1    | **3420.48±729.72** | **593.73±239.85** |
| SafetyHalfCheetahVelocity-v1 | **3031.78±239.26** | **728.80±128.80** |
|    SafetyHopperVelocity-v1   | **2879.188±788.27** | **814.40±252.71** |
|   SafetyWalker2dVelocity-v1  | **4093.80±243.35** | **898.43±43.18** |
|   SafetySwimmerVelocity-v1   | **67.13±45.06** | **458.8±382.83** |
|   SafetyHumanoidVelocity-v1  | **914.64±225.15** | **75.10±26.77** |

| Environment | obs_normlize | rew_normlize | cost_normlize |
| :---------: | :-----------: |  :-----------: |  :-----------: |
|     SafetyAntVelocity-v1    | True | True | False |
| SafetyHalfCheetahVelocity-v1 | True | True | False  |
|    SafetyHopperVelocity-v1   | True | False | False  |
|    SafetyWalker2dVelocity-v1 | True | False | False  |
|    SafetySwimmerVelocity-v1  | True | True | False  |
|    SafetyHumanoidVelocity-v1 | True | True | False  |

#### RCPO(1M)

| Environment | Reward （OmniSafe）| Cost （OmniSafe） |
| :---------: | :-----------: |  :-----------: |
|     SafetyAntVelocity-v1    | **1981.23±156.10** | **2.26±3.63** |
| SafetyHalfCheetahVelocity-v1 | **1636.01±193.70** | **11.9±8.79** |
|    SafetyHopperVelocity-v1   | **996.26±8.47** | **0.0±0.0** |
|   SafetyWalker2dVelocity-v1  | **1842.13±204.86** | **11.66±24.40** |
|   SafetySwimmerVelocity-v1   | **26.48±5.03** | **55.20±9.93** |
|   SafetyHumanoidVelocity-v1  | **807.00±52.35** | **35.4±10.13** |

| Environment | obs_normlize | rew_normlize | cost_normlize |
| :---------: | :-----------: |  :-----------: |  :-----------: |
|     SafetyAntVelocity-v1    | True | False | True |
| SafetyHalfCheetahVelocity-v1 | True | False | True |
|    SafetyHopperVelocity-v1   | True | False | True |
|    SafetyWalker2dVelocity-v1 | True | False | True |
|    SafetySwimmerVelocity-v1  | True | False | True |
|    SafetyHumanoidVelocity-v1   | True | True | True |

#### TRPO(1M)

| Environment | Reward （OmniSafe）| Cost （OmniSafe） |
| :---------: | :-----------: |  :-----------: |
|     SafetyAntVelocity-v1    | **4673.13±874.97** | **888.83±160.12** |
| SafetyHalfCheetahVelocity-v1 | **4321.24±721.59** | **922.166±69.77** |**759.4±329.8** |
|    SafetyHopperVelocity-v1   | **3483.73±169.43** | **993.06±0.24** |
|   SafetyWalker2dVelocity-v1  | **3856.61±901.99** | **787.8±165.37** |
|   SafetySwimmerVelocity-v1   | **40.32±7.64** | **590.2±314.85** |
|   SafetyHumanoidVelocity-v1  | **1617.75±663.87** | **155.46±75.08** |

| Environment | obs_normlize | rew_normlize | cost_normlize |
| :---------: | :-----------: |  :-----------: |  :-----------: |
|     SafetyAntVelocity-v1    | True | True | False |
| SafetyHalfCheetahVelocity-v1 | False | True | False |
|    SafetyHopperVelocity-v1   | True | False | False  |
|    SafetyWalker2dVelocity-v1 | True | False | False  |
|    SafetySwimmerVelocity-v1  | True | True | False  |
|    SafetyHumanoidVelocity-v1 | True | True | False  |

#### TRPOLag(1M)

| Environment | Reward （OmniSafe）| Cost （OmniSafe） |
| :---------: | :-----------: |  :-----------: |
|     SafetyAntVelocity-v1    | **2407.27±131.24** | **3.03±4.98** |
| SafetyHalfCheetahVelocity-v1 | **1892.77±505.02** | **9.7±8.51** |
|    SafetyHopperVelocity-v1   | **1013.93±18.01** | **1.63±2.31** |
|   SafetyWalker2dVelocity-v1  | **1821.26±98.57** | **0.0±0.0** |
|   SafetySwimmerVelocity-v1   | **21.91±8.44** | **58.23±336.05** |
|   SafetyHumanoidVelocity-v1  | **799.91±60.31** | **34.56±9.56** |

| Environment | obs_normlize | rew_normlize | cost_normlize |
| :---------: | :-----------: |  :-----------: |  :-----------: |
|     SafetyAntVelocity-v1    | True | True | True |
| SafetyHalfCheetahVelocity-v1 | True | True | True |
|    SafetyHopperVelocity-v1   | True | False | True |
|    SafetyWalker2dVelocity-v1 | True | False | True |
|    SafetySwimmerVelocity-v1  | True | True | True |
|    SafetyHumanoidVelocity-v1 | True | True | True |

#### CPO(1M)

| Environment | Reward （OmniSafe）| Cost （OmniSafe） |
| :---------: | :-----------: |  :-----------: |
|     SafetyAntVelocity-v1    | **1904.25±337.31** | **36.20±23.66** |
| SafetyHalfCheetahVelocity-v1 | **1527.98±179.73** | **87.73±17.04** |
|    SafetyHopperVelocity-v1   | **115.82±37.05** | **50.6±16.05** |
|   SafetyWalker2dVelocity-v1  | **276.20±11.77** | **41.36±4.51** |
|   SafetySwimmerVelocity-v1   | **21.35±14.84** | **107.26±58.01** |
|   SafetyHumanoidVelocity-v1  | **574.79±75.83** | **35.1±5.04** |

| Environment | obs_normlize | rew_normlize | cost_normlize |
| :---------: | :-----------: |  :-----------: |  :-----------: |
|     SafetyAntVelocity-v1    | True | True | True |
| SafetyHalfCheetahVelocity-v1 | True | True | True |
|    SafetyHopperVelocity-v1   | True | False | True |
|    SafetyWalker2dVelocity-v1 | True | False | True |
|    SafetySwimmerVelocity-v1  | True | True | True |
|    SafetyHumanoidVelocity-v1 | True | True | True |

#### PCPO(1M)

| Environment | Reward （OmniSafe）| Cost （OmniSafe） |
| :---------: | :-----------: |  :-----------: |
|     SafetyAntVelocity-v1    | **2270.54±41.63** | **62.4±55.40** |
| SafetyHalfCheetahVelocity-v1 | **1233.64±412.36** | **23.80±21.55** |
|    SafetyHopperVelocity-v1   | **829.42±256.74** | **41.6±57.75** |
|   SafetyWalker2dVelocity-v1  | **373.74±53.45** | **29.1±3.81** |
|   SafetySwimmerVelocity-v1   | **24.49±10.58** | **139.3±39.82** |
|   SafetyHumanoidVelocity-v1  | **668.26±199.91** | **7.63±14.33** |

| Environment | obs_normlize | rew_normlize | cost_normlize |
| :---------: | :-----------: |  :-----------: |  :-----------: |
|     SafetyAntVelocity-v1    | True | True | True |
| SafetyHalfCheetahVelocity-v1 | True | True | True |
|    SafetyHopperVelocity-v1   | True | False | True |
|    SafetyWalker2dVelocity-v1 | True | False | True |
|    SafetySwimmerVelocity-v1  | True | True | True |
|    SafetyHumanoidVelocity-v1 | True | True | True |
