# OmniSafe's Mujoco Velocity Benchmark on On-Policy Algorithms

OmniSafe's Mujoco Velocity Benchmark evaluated the performance of OmniSafe algorithm implementations in 6 environments from the Safety-Gymnasium task suite For each algorithm and environment supported, we provide:
- Default hyperparameters used for the benchmark and scripts to reproduce the results
- A comparison of performance or code-level details with other open-source implementations or classic papers.
- Graphs and raw data that can be used for research purposes, - Log details obtained during training
- Some hints on how to fine-tune the algorithm for optimal results.

Supported algorithms are listed below:

- [X] **[NeurIPS 2001]** [A Natural Policy Gradient (NaturalPG))](https://proceedings.neurips.cc/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf)
- [X] **[PMLR 2015]** [Trust Region Policy Optimization (TRPO)](https://arxiv.org/abs/1502.05477)
- [X] [The Lagrange version of TRPO (TRPO-Lag)](https://cdn.openai.com/safexp-short.pdf)
- [X] **[ICML 2017]** [Constrained Policy Optimization (CPO)](https://proceedings.mlr.press/v70/achiam17a)
- [X] **[ICML 2017]** [Proximal Constrained Policy Optimization (PCPO)](https://proceedings.mlr.press/v70/achiam17a)
- [X] **[ICLR 2019]** [Reward Constrained Policy Optimization (RCPO)](https://openreview.net/forum?id=SkfrvsA9FX)

## Safety-Gymnasium

We highly recommend using ``safety-gymnasium`` to run the following experiments. To install, in a linux machine, type:

```bash
pip install safety_gymnasium
```

## Run the Benchmark
You can set the main function of ``examples/benchmarks/experimrnt_grid.py`` as:

```python
if __name__ == '__main__':
    eg = ExperimentGrid(exp_name='Safety_Gymnasium_Goal')

    # set up the algorithms
    base_policy = ['PolicyGradient', 'NaturalPG', 'TRPO', 'PPO']
    naive_lagrange_policy = ['PPOLag', 'TRPOLag', 'RCPO', 'OnCRPO', 'PDO']
    first_order_policy = ['CUP', 'FOCOPS']
    second_order_policy = ['CPO', 'PCPO']

    eg.add('algo', base_policy + naive_lagrange_policy + first_order_policy + second_order_policy)

     # You can use wandb to monitor the experiment.
    eg.add('logger_cfgs:use_wandb', [True])
    # you can use tensorboard to monitor the experiment.
    eg.add('logger_cfgs:use_tensorboard', [True])

    # set up the environment
    eg.add('env_id', [
        'SafetyHopperVelocity-v4',
        'SafetyWalker2dVelocity-v4',
        'SafetySwimmerVelocity-v4',
        'SafetyAntVelocity-v4',
        'SafetyHalfCheetahVelocity-v4',
        'SafetyHumanoidVelocity-v4'
        ])
    eg.add('seed', [0, 5, 10, 15, 20])
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

e.g. ALGODIR can be ``examples/runs/SafetyHopperVelocity-v4``.
Then you can compare different algorithms in ``SafetyHopperVelocity-v4`` environments.

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

For example, if I train ``PPOLag`` in ``SafetyHumanoidVelocity-v4``

```python
    LOG_DIR = '~/omnisafe/examples/runs/PPOLag-<SafetyHumanoidVelocity-v4>/seed-000-2023-03-07-20-25-48'
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
    src="./benchmarks/second_order_ant.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">SafetyAntVelocity-v4</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/second_order_halfcheetah.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">SafetyHalfCheetahVelocity-v4</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/second_order_hopper.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">SafetyHopperVelocity-v4</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/second_order_humanoid.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">SafetyHumanoidVelocity-v4</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/second_order_walker2d.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">SafetyWalker2dVelocity-v4</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/second_order_swimmer.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">SafetySwimmerVelocity-v4</div>
</center>

## Experiment Analysis

### Hyperparameters

#### Second Order Methods Specific Hyperparameters

- ``algo_cfgs:kl_early_stop``: Whether to use early stop for KL divergence. In the second order methods, we use line search to find the proper step size. If the KL divergence is too large, we will stop the line search and use the previous step size. So we always set this hyperparameter to ``False``.

- ``model_cfgs:actor:lr``: The learning rate of the actor network. The second order methods use the actor network update the policy by directly setting the parameters of the policy network. So we do not need to set the learning rate of the policy network, which is set to ``None``.

### Some Hints

In our experiments, we found that somehyperparameters are important for the performance of the algorithm:

- ``obs_normlize``: Whether to normalize the observation.
- ``rew_normlize``: Whether to normalize the reward.
- ``cost_normlize``: Whether to normalize the cost.

We have done some experiments to show the effect of these hyperparameters, and we log the best configuration for each algorithm in each environment. You can check it in the ``omnisafe/configs/on_policy``.

In experiments, we found that the ``obs_normlize=True`` always performs better than ``obs_normlize=False`` in the second order methods. That means the reward would increase quicker if we normalize the observation. So we set ``obs_normlize=True`` in almost all the second order methods.

Importantly, we found that the ``rew_normlize=True`` not always performs better than ``rew_normlize=False``, especially in the ``SafetyHopperVelocity-v4`` and ``SafetyWalker2dVelocity`` environment.

The Lagrangian method often has the phenomenon of unstable update and easy overshoot。 We found that with ``cost_normlize=True``, the Lagrangian method can update more stable and avoid overshoot to some degree. So we set ``cost_normlize=True`` in the Lagrangian method.

Besides, the hyperparameter ``train_cfgs:torch_num_threads`` is also important. on-policy algorithms always use more time to update policy than to sample data. So we use ``torch_num_threads`` to speed up the update process.

This hyperparamter depens on the number of CPU cores. We set it to 8 in our experiments. You can set it to some other porper value according to your CPU cores.

If you find that other hyperparameters perform better, please feel free to open an issue or pull request.

### Experiment Results

#### NaturalPG(1M)

| Environment | Reward （OmniSafe）| Cost （OmniSafe） |
| :---------: | :-----------: |  :-----------: |
|     SafetyAntVelocity-v4    | **3420.48±729.72** | **593.73±239.85** |
| SafetyHalfCheetahVelocity-v4 | **3031.78±239.26** | **728.80±128.80** |
|    SafetyHopperVelocity-v4   | **2879.188±788.27** | **814.40±252.71** |
|   SafetyWalker2dVelocity-v4  | **4093.80±243.35** | **898.43±43.18** |
|   SafetySwimmerVelocity-v4   | **67.13±45.06** | **458.8±382.83** |
|   SafetyHumanoidVelocity-v4  | **914.64±225.15** | **75.10±26.77** |

| Environment | obs_normlize | rew_normlize | cost_normlize |
| :---------: | :-----------: |  :-----------: |  :-----------: |
|     SafetyAntVelocity-v4    | True | True | False |
| SafetyHalfCheetahVelocity-v4 | True | True | False  |
|    SafetyHopperVelocity-v4   | True | False | False  |
|    SafetyWalker2dVelocity-v4 | True | False | False  |
|    SafetySwimmerVelocity-v4  | True | True | False  |
|    SafetyHumanoidVelocity-v4 | True | True | False  |

#### RCPO(1M)

| Environment | Reward （OmniSafe）| Cost （OmniSafe） |
| :---------: | :-----------: |  :-----------: |
|     SafetyAntVelocity-v4    | **1981.23±156.10** | **2.26±3.63** |
| SafetyHalfCheetahVelocity-v4 | **1636.01±193.70** | **11.9±8.79** |
|    SafetyHopperVelocity-v4   | **996.26±8.47** | **0.0±0.0** |
|   SafetyWalker2dVelocity-v4  | **1842.13±204.86** | **11.66±24.40** |
|   SafetySwimmerVelocity-v4   | **26.48±5.03** | **55.20±9.93** |
|   SafetyHumanoidVelocity-v4  | **807.00±52.35** | **35.4±10.13** |

| Environment | obs_normlize | rew_normlize | cost_normlize |
| :---------: | :-----------: |  :-----------: |  :-----------: |
|     SafetyAntVelocity-v4    | True | False | True |
| SafetyHalfCheetahVelocity-v4 | True | False | True |
|    SafetyHopperVelocity-v4   | True | False | True |
|    SafetyWalker2dVelocity-v4 | True | False | True |
|    SafetySwimmerVelocity-v4  | True | False | True |
|    SafetyHumanoidVelocity-v4   | True | True | True |

#### TRPO(1M)

| Environment | Reward （OmniSafe）| Cost （OmniSafe） |
| :---------: | :-----------: |  :-----------: |
|     SafetyAntVelocity-v4    | **4673.13±874.97** | **888.83±160.12** |
| SafetyHalfCheetahVelocity-v4 | **4321.24±721.59** | **922.166±69.77** |**759.4±329.8** |
|    SafetyHopperVelocity-v4   | **3483.73±169.43** | **993.06±0.24** |
|   SafetyWalker2dVelocity-v4  | **3856.61±901.99** | **787.8±165.37** |
|   SafetySwimmerVelocity-v4   | **40.32±7.64** | **590.2±314.85** |
|   SafetyHumanoidVelocity-v4  | **1617.75±663.87** | **155.46±75.08** |

| Environment | obs_normlize | rew_normlize | cost_normlize |
| :---------: | :-----------: |  :-----------: |  :-----------: |
|     SafetyAntVelocity-v4    | True | True | False |
| SafetyHalfCheetahVelocity-v4 | False | True | False |
|    SafetyHopperVelocity-v4   | True | False | False  |
|    SafetyWalker2dVelocity-v4 | True | False | False  |
|    SafetySwimmerVelocity-v4  | True | True | False  |
|    SafetyHumanoidVelocity-v4 | True | True | False  |

#### TRPOLag(1M)

| Environment | Reward （OmniSafe）| Cost （OmniSafe） |
| :---------: | :-----------: |  :-----------: |
|     SafetyAntVelocity-v4    | **2407.27±131.24** | **3.03±4.98** |
| SafetyHalfCheetahVelocity-v4 | **1892.77±505.02** | **9.7±8.51** |
|    SafetyHopperVelocity-v4   | **1013.93±18.01** | **1.63±2.31** |
|   SafetyWalker2dVelocity-v4  | **1821.26±98.57** | **0.0±0.0** |
|   SafetySwimmerVelocity-v4   | **21.91±8.44** | **58.23±336.05** |
|   SafetyHumanoidVelocity-v4  | **799.91±60.31** | **34.56±9.56** |

| Environment | obs_normlize | rew_normlize | cost_normlize |
| :---------: | :-----------: |  :-----------: |  :-----------: |
|     SafetyAntVelocity-v4    | True | True | True |
| SafetyHalfCheetahVelocity-v4 | True | True | True |
|    SafetyHopperVelocity-v4   | True | False | True |
|    SafetyWalker2dVelocity-v4 | True | False | True |
|    SafetySwimmerVelocity-v4  | True | True | True |
|    SafetyHumanoidVelocity-v4 | True | True | True |

#### CPO(1M)

| Environment | Reward （OmniSafe）| Cost （OmniSafe） |
| :---------: | :-----------: |  :-----------: |
|     SafetyAntVelocity-v4    | **1904.25±337.31** | **36.20±23.66** |
| SafetyHalfCheetahVelocity-v4 | **1527.98±179.73** | **87.73±17.04** |
|    SafetyHopperVelocity-v4   | **115.82±37.05** | **50.6±16.05** |
|   SafetyWalker2dVelocity-v4  | **276.20±11.77** | **41.36±4.51** |
|   SafetySwimmerVelocity-v4   | **21.35±14.84** | **107.26±58.01** |
|   SafetyHumanoidVelocity-v4  | **574.79±75.83** | **35.1±5.04** |

| Environment | obs_normlize | rew_normlize | cost_normlize |
| :---------: | :-----------: |  :-----------: |  :-----------: |
|     SafetyAntVelocity-v4    | True | True | True |
| SafetyHalfCheetahVelocity-v4 | True | True | True |
|    SafetyHopperVelocity-v4   | True | False | True |
|    SafetyWalker2dVelocity-v4 | True | False | True |
|    SafetySwimmerVelocity-v4  | True | True | True |
|    SafetyHumanoidVelocity-v4 | True | True | True |

#### PCPO(1M)

| Environment | Reward （OmniSafe）| Cost （OmniSafe） |
| :---------: | :-----------: |  :-----------: |
|     SafetyAntVelocity-v4    | **2270.54±41.63** | **62.4±55.40** |
| SafetyHalfCheetahVelocity-v4 | **1233.64±412.36** | **23.80±21.55** |
|    SafetyHopperVelocity-v4   | **829.42±256.74** | **41.6±57.75** |
|   SafetyWalker2dVelocity-v4  | **373.74±53.45** | **29.1±3.81** |
|   SafetySwimmerVelocity-v4   | **24.49±10.58** | **139.3±39.82** |
|   SafetyHumanoidVelocity-v4  | **668.26±199.91** | **7.63±14.33** |

| Environment | obs_normlize | rew_normlize | cost_normlize |
| :---------: | :-----------: |  :-----------: |  :-----------: |
|     SafetyAntVelocity-v4    | True | True | True |
| SafetyHalfCheetahVelocity-v4 | True | True | True |
|    SafetyHopperVelocity-v4   | True | False | True |
|    SafetyWalker2dVelocity-v4 | True | False | True |
|    SafetySwimmerVelocity-v4  | True | True | True |
|    SafetyHumanoidVelocity-v4 | True | True | True |
