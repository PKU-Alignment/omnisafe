# OmniSafe's Navigation Benchmark on Model-based Algorithms

OmniSafe's navigation Benchmark evaluated the performance of OmniSafe algorithm implementations in 2 environments from the Safety-Gymnasium task suite For each algorithm and environment supported, we provide:

- Default hyperparameters used for the benchmark and scripts to reproduce the results.
- A comparison of performance or code-level details with other open-source implementations or classic papers.
- Graphs and raw data that can be used for research purposes.
- Log details obtained during training.
- Some hints on how to fine-tune the algorithm for optimal results.

Supported algorithms are listed below:

- **[NeurIPS 2001]** [Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models (PETS))](https://arxiv.org/abs/1805.12114)
- **[CoRL 2021]** [Learning Off-Policy with Online Planning (LOOP and SafeLOOP)](https://arxiv.org/abs/2008.10066)
- **[AAAI 2022]** [Conservative and Adaptive Penalty for Model-Based Safe Reinforcement Learning (CAP)](https://arxiv.org/abs/2112.07701)
- **[ICML 2022 Workshop]** [Constrained Model-based Reinforcement Learning with Robust Cross-Entropy Method (RCE)](https://arxiv.org/abs/2010.07968)
- **[NeurIPS 2018]** [Constrained Cross-Entropy Method for Safe Reinforcement Learning (CCE)](https://proceedings.neurips.cc/paper/2018/hash/34ffeb359a192eb8174b6854643cc046-Abstract.html)

## Safety-Gymnasium

We highly recommend using ``safety-gymnasium`` to run the following experiments. To install, in a linux machine, type:

```bash
pip install safety_gymnasium
```

## Run the Benchmark

You can set the main function of ``examples/benchmarks/experimrnt_grid.py`` as:

```python
if __name__ == '__main__':
    eg = ExperimentGrid(exp_name='Model-Based-Benchmarks')

    # set up the algorithms.
    model_based_base_policy = ['LOOP', 'PETS']
    model_based_safe_policy = ['SafeLOOP', 'CCEPETS', 'CAPPETS', 'RCEPETS']
    eg.add('algo', model_based_base_policy + model_based_safe_policy)

    # you can use wandb to monitor the experiment.
    eg.add('logger_cfgs:use_wandb', [False])
    # you can use tensorboard to monitor the experiment.
    eg.add('logger_cfgs:use_tensorboard', [True])
    eg.add('train_cfgs:total_steps', [1000000])

    # set up the environment.
    eg.add('env_id', [
        'SafetyPointGoal1-v0-modelbased',
        'SafetyCarGoal1-v0-modelbased',
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

You can set the path of ``examples/benchmarks/experimrnt_grid.py`` :
example:

```python
path ='/home/username/omnisafe/omnisafe/examples/benchmarks/exp-x/Model-Based-Benchmarks'
```

You can also plot the results by running the following command:

```bash
cd examples
python analyze_experiment_results.py
```

## Example benchmark

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/SafetyCarGoal1-v0-modelbased.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">SafetyCarGoal1-v0-modelbased</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    src="./benchmarks/SafetyPointGoal1-v0-modelbased.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">SafetyPointGoal1-v0-modelbased</div>
</center>

## Some Hints

In our experiments, we found that somehyperparameters are important for the performance of the algorithm:

- ``action_repeat``: The time of action repeat.
- ``init_var``: The initial variance of gaussian distribution for sampling actions.
- ``temperature``: The temperature factor for rescaling reward in planning.
- ``cost_temperature``: : The temperature factor for rescaling cost in planning
- ``plan_horizon``: Planning horizon.

We have done some experiments to show the effect of these hyperparameters, and we log the best configuration for each algorithm in each environment. You can check it in the ``omnisafe/configs/model_based``.

In experiments, we found that the ``action_repeat=5`` always performs better than ``action_repeat=1`` in the navigation task for the cem-based methods. That means the change in reward or observation per action performed in a navigation task may be too small, the ``action_repeat=5`` will enlarge these variable and make the dynamics model more trainable.


Importantly, we found that the high variance like ``init_var=4.0`` performs better than low variance like ``init_var=0.01`` in pets-based algorithms, but we found that the situation is the opposite in policy-guided algorithms like LOOP, LOOP need the low variance like ``init_var=0.01`` to make the planning policy more similar to the neural policy.

Besides, the hyperparameter ``temperature`` and ``cost_temperature`` are also important. LOOP and SafeLOOP should fine tune these two parameters in in different environments. This affects the contribution of reward size to action mean and variance.

Moreover, No policy-guided like pets need the high ``plan_horizon``, and policy-guided algorithms like loop only need low ``plan_horizon` in mujoco environments, but for a fair comparison, we use the planning horizon in navigation tasks.

If you find that other hyperparameters perform better, please feel free to open an issue or pull request.

| Algorithm | action_repeat | init_var | plan_horizon |
| :-------: | :-----------: | :------: | :----------: |
|   PETS    |       5       |   4.0    |      7       |
|   LOOP    |       5       |   0.01   |      7       |
| SafeLOOP  |       5       |  0.075   |      7       |
|  CCEPETS  |       5       |   4.0    |      7       |
|  CAPPETS  |       5       |   4.0    |      7       |
|  RCEPETS  |       5       |   4.0    |      7       |

However, there are some differences between these algorithms. We list the differences below:

### LOOP

|     Environment     | temperature |
| :-----------------: | :---------: |
| SafetyPointGoal1-v0 |    10.0     |
|  SafetyCarGoal1-v0  |    10.0     |

### SafeLOOP

|     Environment     | temperature | cost_temperature |
| :-----------------: | :---------: | :--------------: |
| SafetyPointGoal1-v0 |    10.0     |      100.0       |
|  SafetyCarGoal1-v0  |    10.0     |      100.0       |
