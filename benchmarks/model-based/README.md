# OmniSafe's Navigation Benchmark for Model-based Algorithms

The OmniSafe Navigation Benchmark for model-based algorithms evaluates the effectiveness of OmniSafe's model-based algorithms across two different environments from the [Safety-Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium) task suite. For each supported algorithm and environment, we offer the following:

- Default hyperparameters used for the benchmark and scripts that enable result replication.
- Graphs and raw data that can be utilized for research purposes.
- Detailed logs obtained during training.
- Suggestions and hints on fine-tuning the algorithm for achieving optimal results.

Supported algorithms are listed below:

- **[NeurIPS 2001]** [Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models (PETS))](https://arxiv.org/abs/1805.12114)
- **[CoRL 2021]** [Learning Off-Policy with Online Planning (LOOP and SafeLOOP)](https://arxiv.org/abs/2008.10066)
- **[AAAI 2022]** [Conservative and Adaptive Penalty for Model-Based Safe Reinforcement Learning (CAP)](https://arxiv.org/abs/2112.07701)
- **[ICML 2022 Workshop]** [Constrained Model-based Reinforcement Learning with Robust Cross-Entropy Method (RCE)](https://arxiv.org/abs/2010.07968)
- **[NeurIPS 2018]** [Constrained Cross-Entropy Method for Safe Reinforcement Learning (CCE)](https://proceedings.neurips.cc/paper/2018/hash/34ffeb359a192eb8174b6854643cc046-Abstract.html)

## Safety-Gymnasium

We highly recommend using **Safety-Gymnasium** to run the following experiments. To install, in a linux machine, type:

```bash
pip install safety_gymnasium
```

## Run the Benchmark

You can set the main function of ``examples/benchmarks/experiment_grid.py`` as:

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

You can set the path of ``examples/benchmarks/experiment_grid.py`` :
example:

```python
path ='/home/username/omnisafe/omnisafe/examples/benchmarks/exp-x/Model-Based-Benchmarks'
```

You can also plot the results by running the following command:

```bash
cd examples
python analyze_experiment_results.py
```

**For a detailed usage of OmniSafe statistics tool, please refer to [this tutorial](https://omnisafe.readthedocs.io/en/latest/common/stastics_tool.html).**

## OmniSafe Benchmark

To demonstrate the high reliability of the algorithms implemented, OmniSafe offers performance insights within the Safety-Gymnasium environment. It should be noted that all data is procured under the constraint of `cost_limit=1.00`. The results are presented in <a href="#performance_model_based">Table 1</a> and <a href="#curve_model_based">Figure 1</a>.

### Performance Table

<table id="performance_model_based">
<thead>
<tr class="header">
<th style="text-align: left;"></th>
<th colspan="2" style="text-align: center;"><strong>PETS</strong></th>
<th colspan="2" style="text-align: center;"><strong>LOOP</strong></th>
<th colspan="2"
style="text-align: center;"><strong>SafeLOOP</strong></th>
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
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyCarGoal1-v0</span></td>
<td style="text-align: center;">33.07 <span class="math inline">±</span>
1.33</td>
<td style="text-align: center;">61.20 <span class="math inline">±</span>
7.23</td>
<td style="text-align: center;">25.41 <span class="math inline">±</span>
1.23</td>
<td style="text-align: center;">62.64 <span class="math inline">±</span>
8.34</td>
<td style="text-align: center;">22.09 <span class="math inline">±</span>
0.30</td>
<td style="text-align: center;">0.16 <span class="math inline">±</span>
0.15</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyPointGoal1-v0</span></td>
<td style="text-align: center;">27.66 <span class="math inline">±</span>
0.07</td>
<td style="text-align: center;">49.16 <span class="math inline">±</span>
2.69</td>
<td style="text-align: center;">25.08 <span class="math inline">±</span>
1.47</td>
<td style="text-align: center;">55.23 <span class="math inline">±</span>
2.64</td>
<td style="text-align: center;">22.94 <span class="math inline">±</span>
0.72</td>
<td style="text-align: center;">0.04 <span class="math inline">±</span>
0.07</td>
</tr>
<thead>
<tr class="header">
<th style="text-align: left;"></th>
<th colspan="2" style="text-align: center;"><strong>CCEPETS</strong></th>
<th colspan="2" style="text-align: center;"><strong>RCEPETS</strong></th>
<th colspan="2"
style="text-align: center;"><strong>CAPPETS</strong></th>
</tr>
</thead>
<tr class="odd">
<td style="text-align: left;"><strong>Environment</strong></td>
<td style="text-align: center;"><strong>Reward</strong></td>
<td style="text-align: center;"><strong>Cost</strong></td>
<td style="text-align: center;"><strong>Reward</strong></td>
<td style="text-align: center;"><strong>Cost</strong></td>
<td style="text-align: center;"><strong>Reward</strong></td>
<td style="text-align: center;"><strong>Cost</strong></td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyCarGoal1-v0</span></td>
<td style="text-align: center;">27.60 <span class="math inline">±</span>
1.21</td>
<td style="text-align: center;">1.03 <span class="math inline">±</span>
0.29</td>
<td style="text-align: center;">29.08 <span class="math inline">±</span>
1.63</td>
<td style="text-align: center;">1.02 <span class="math inline">±</span>
0.88</td>
<td style="text-align: center;">23.33 <span class="math inline">±</span>
6.34</td>
<td style="text-align: center;">0.48 <span class="math inline">±</span>
0.17</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyPointGoal1-v0</span></td>
<td style="text-align: center;">24.98 <span class="math inline">±</span>
0.05</td>
<td style="text-align: center;">1.87 <span class="math inline">±</span>
1.27</td>
<td style="text-align: center;">25.39 <span class="math inline">±</span>
0.28</td>
<td style="text-align: center;">2.46 <span class="math inline">±</span>
0.58</td>
<td style="text-align: center;">9.45 <span class="math inline">±</span>
8.62</td>
<td style="text-align: center;">0.64 <span class="math inline">±</span>
0.77</td>
</tr>
</tbody>
  <caption><p style="font-family: 'Times New Roman', Times, serif;"><b>Table 1:</b> The performance of OmniSafe model-based algorithms, encompassing both reward and cost, was assessed within the Safety-Gymnasium environments. It is crucial to highlight that all model-based algorithms underwent evaluation following 1e6 training steps.</p></caption>
</table>

### Performance Curves

<table id="curve_model_based">
  <tr>
    <td style="text-align:center">
      <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/model-based/benchmarks/SafetyCarGoal1-v0-modelbased.png">
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
        SafetyCarGoal1-v0
      </div>
    </td>
  </tr>
  <tr>
    <td style="text-align:center">
      <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/model-based/benchmarks/SafetyPointGoal1-v0-modelbased.png">
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
        SafetyPointGoal1-v0
      </div>
    </td>
  </tr>
  <caption><p style="font-family: 'Times New Roman', Times, serif;"><b>Figure 1:</b> Training curves in Safety-Gymnasium environments, covering classical reinforcement learning algorithms and safe learning algorithms mentioned in <a href="#performance_model_based">Table 1</a>.</p></caption>
</table>

## Some Hints

In our experiments, we found that some hyperparameters are important for the performance of the algorithm:

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
