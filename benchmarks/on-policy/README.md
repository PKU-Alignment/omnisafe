# OmniSafe's Safety-Gymnasium Benchmark for On-Policy Algorithms

The OmniSafe Safety-Gymnasium Benchmark for on-policy algorithms evaluates the effectiveness of OmniSafe's on-policy algorithms across multiple environments from the [Safety-Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium) task suite. For each supported algorithm and environment, we offer the following:

- Default hyperparameters used for the benchmark and scripts that enable result replication.
- Performance comparison with other open-source implementations.
- Graphs and raw data that can be utilized for research purposes.
- Detailed logs obtained during training.
- Suggestions and hints on fine-tuning the algorithm for achieving optimal results.

Supported algorithms are listed below:

**First-Order**

- **[NIPS 1999]** [Policy Gradient (PG)](https://papers.nips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)
- **[Preprint 2017]**[Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf)
- [The Lagrange version of PPO (PPOLag)](https://cdn.openai.com/safexp-short.pdf)
- **[IJCAI 2022]** [Penalized Proximal Policy Optimization for Safe Reinforcement Learning (P3O)]( https://arxiv.org/pdf/2205.11814.pdf)
- **[NeurIPS 2020]** [First Order Constrained Optimization in Policy Space (FOCOPS)](https://arxiv.org/abs/2002.06506)
- **[NeurIPS 2022]**  [Constrained Update Projection Approach to Safe Policy Optimization (CUP)](https://arxiv.org/abs/2209.07089)

**Second-Order**

- **[NeurIPS 2001]** [A Natural Policy Gradient (NaturalPG))](https://proceedings.neurips.cc/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf)
- **[PMLR 2015]** [Trust Region Policy Optimization (TRPO)](https://arxiv.org/abs/1502.05477)
- [The Lagrange version of TRPO (TRPOLag)](https://cdn.openai.com/safexp-short.pdf)
- **[ICML 2017]** [Constrained Policy Optimization (CPO)](https://proceedings.mlr.press/v70/achiam17a)
- **[ICML 2017]** [Proximal Constrained Policy Optimization (PCPO)](https://proceedings.mlr.press/v70/achiam17a)
- **[ICLR 2019]** [Reward Constrained Policy Optimization (RCPO)](https://openreview.net/forum?id=SkfrvsA9FX)

**Saute RL**

- **[ICML 2022]** [Sauté RL: Almost Surely Safe Reinforcement Learning Using State Augmentation (PPOSaute, TRPOSaute)](https://arxiv.org/abs/2202.06558)

**Simmer**

- **[NeurIPS 2022]** [Effects of Safety State Augmentation on Safe Exploration (PPOSimmerPID, TRPOSimmerPID)](https://arxiv.org/abs/2206.02675)

**PID-Lagrangian**

- **[ICML 2020]** [Responsive Safety in Reinforcement Learning by PID Lagrangian Methods (CPPOPID, TRPOPID)](https://arxiv.org/abs/2007.03964)

**Early Terminated MDP**

- **[Preprint 2021]** [Safe Exploration by Solving Early Terminated MDP (PPOEarlyTerminated, TRPOEarlyTerminated)](https://arxiv.org/pdf/2107.04200.pdf)




## Safety-Gymnasium

We highly recommend using **Safety-Gymnasium** to run the following experiments. To install, in a linux machine, type:

```bash
pip install safety_gymnasium
```

## Run the Benchmark

You can set the main function of `examples/benchmarks/experiment_grid.py` as:

```python
if __name__ == '__main__':
    eg = ExperimentGrid(exp_name='On-Policy-Benchmarks')

    # set up the algorithms.
    base_policy = ['PolicyGradient', 'NaturalPG', 'TRPO', 'PPO']
    naive_lagrange_policy = ['PPOLag', 'TRPOLag', 'RCPO']
    first_order_policy = ['CUP', 'FOCOPS', 'P3O']
    second_order_policy = ['CPO', 'PCPO']
    saute_policy = ['PPOSaute', 'TRPOSaute']
    simmer_policy = ['PPOSimmerPID', 'TRPOSimmerPID']
    pid_policy = ['CPPOPID', 'TRPOPID']
    early_mdp_policy = ['PPOEarlyTerminated', 'TRPOEarlyTerminated']

    eg.add(
        'algo',
        base_policy +
        naive_lagrange_policy +
        first_order_policy +
        second_order_policy +
        saute_policy +
        simmer_policy +
        pid_policy +
        early_mdp_policy
    )

    # you can use wandb to monitor the experiment.
    eg.add('logger_cfgs:use_wandb', [False])
    # you can use tensorboard to monitor the experiment.
    eg.add('logger_cfgs:use_tensorboard', [True])

    # the default configs here are as follows:
    # eg.add('algo_cfgs:steps_per_epoch', [20000])
    # eg.add('train_cfgs:total_steps', [20000 * 500])
    # which can reproduce results of 1e7 steps.

    # if you want to reproduce results of 1e6 steps, using
    # eg.add('algo_cfgs:steps_per_epoch', [2048])
    # eg.add('train_cfgs:total_steps', [2048 * 500])

    # set the device.
    avaliable_gpus = list(range(torch.cuda.device_count()))
    # if you want to use GPU, please set gpu_id like follows:
    # gpu_id = [0, 1, 2, 3]
    # if you want to use CPU, please set gpu_id = None
    # we recommends using CPU to obtain results as consistent
    # as possible with our publicly available results,
    # since the performance of all on-policy algorithms
    # in OmniSafe is tested on CPU.
    gpu_id = None

    if gpu_id and not set(gpu_id).issubset(avaliable_gpus):
        warnings.warn('The GPU ID is not available, use CPU instead.', stacklevel=1)
        gpu_id = None

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

    # total experiment num must can be divided by num_pool.
    # meanwhile, users should decide this value according to their machine.
    eg.run(train, num_pool=5, gpu_id=gpu_id)
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

Logs is saved in `examples/benchmarks/exp-x` and can be monitored with tensorboard or wandb.

```bash
tensorboard --logdir examples/benchmarks/exp-x
```

After the experiment is finished, you can use the following command to generate the video of the trained agent:

```bash
cd examples
python evaluate_saved_policy.py
```

Please note that before you evaluate, set the `LOG_DIR` in `evaluate_saved_policy.py`.

For example, if I train `PPOLag` in `SafetyHumanoidVelocity-v1`

```python
LOG_DIR = '~/omnisafe/examples/runs/PPOLag-<SafetyHumanoidVelocity-v1>/seed-000'
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

## OmniSafe Benchmark

### Classic Reinforcement Learning Algorithms
To ascertain the credibility of OmniSafe ’s algorithmic implementation, a comparative assessment was conducted, juxtaposing the performance of classical reinforcement learning algorithms. Such as Policy Gradient, Natural Policy Gradient, TRPO and PPO. The performance table is provided in <a
href="#compare_on_policy">Table 1</a>. with well-established open-source implementations, specifically [Tianshou](https://github.com/thu-ml/tianshou) and [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).

<table id="compare_on_policy">
<thead>
<tr class="header">
<th style="text-align: left;"></th>
<th colspan="3" style="text-align: center;"><strong>Policy
Gradient</strong></th>
<th colspan="3" style="text-align: center;"><strong>PPO</strong></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: left;"><strong>Environment</strong></td>
<td style="text-align: center;"><strong>OmniSafe (Ours)</strong></td>
<td style="text-align: center;"><strong>Tianshou</strong></td>
<td style="text-align: center;"><strong>Stable-Baselines3</strong></td>
<td style="text-align: center;"><strong>OmniSafe (Ours)</strong></td>
<td style="text-align: center;"><strong>Tianshou</strong></td>
<td style="text-align: center;"><strong>Stable-Baselines3</strong></td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyAntVelocity-v1</span></td>
<td style="text-align: center;"><strong>2769.45 <span
class="math inline">±</span> 550.71</strong></td>
<td style="text-align: center;">145.33 <span
class="math inline">±</span> 127.55</td>
<td style="text-align: center;">- <span class="math inline">±</span>
-</td>
<td style="text-align: center;"><strong>4295.96 <span
class="math inline">±</span> 658.2</strong></td>
<td style="text-align: center;">2607.48 <span
class="math inline">±</span> 1415.78</td>
<td style="text-align: center;">1780.61 <span
class="math inline">±</span> 780.65</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyHalfCheetahVelocity-v1</span></td>
<td style="text-align: center;"><strong>2625.44 <span
class="math inline">±</span> 1079.04</strong></td>
<td style="text-align: center;">707.56 <span
class="math inline">±</span> 158.59</td>
<td style="text-align: center;">- <span class="math inline">±</span>
-</td>
<td style="text-align: center;">3507.47 <span
class="math inline">±</span> 1563.69</td>
<td style="text-align: center;"><strong>6299.27 <span
class="math inline">±</span> 1692.38</strong></td>
<td style="text-align: center;">5074.85 <span
class="math inline">±</span> 2225.47</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyHopperVelocity-v1</span></td>
<td style="text-align: center;"><strong>1884.38 <span
class="math inline">±</span> 825.13</strong></td>
<td style="text-align: center;">343.88 <span
class="math inline">±</span> 51.85</td>
<td style="text-align: center;">- <span class="math inline">±</span>
-</td>
<td style="text-align: center;"><strong>2679.98 <span
class="math inline">±</span> 921.96</strong></td>
<td style="text-align: center;">1834.7 <span
class="math inline">±</span> 862.06</td>
<td style="text-align: center;">838.96 <span
class="math inline">±</span> 351.10</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyHumanoidVelocity-v1</span></td>
<td style="text-align: center;"><strong>647.52 <span
class="math inline">±</span> 154.82</strong></td>
<td style="text-align: center;">438.97 <span
class="math inline">±</span> 123.68</td>
<td style="text-align: center;">- <span class="math inline">±</span>
-</td>
<td style="text-align: center;"><strong>1106.09 <span
class="math inline">±</span> 607.6</strong></td>
<td style="text-align: center;">677.43 <span
class="math inline">±</span> 189.96</td>
<td style="text-align: center;">762.73 <span
class="math inline">±</span> 170.22</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetySwimmerVelocity-v1</span></td>
<td style="text-align: center;"><strong>47.31 <span
class="math inline">±</span> 16.19</strong></td>
<td style="text-align: center;">27.12 <span class="math inline">±</span>
7.47</td>
<td style="text-align: center;">- <span class="math inline">±</span>
-</td>
<td style="text-align: center;">113.28 <span
class="math inline">±</span> 20.22</td>
<td style="text-align: center;">37.93 <span class="math inline">±</span>
8.68</td>
<td style="text-align: center;"><strong>273.86 <span
class="math inline">±</span> 87.76</strong></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyWalker2dVelocity-v1</span></td>
<td style="text-align: center;"><strong>1665 .00 <span
class="math inline">±</span> 930.18</strong></td>
<td style="text-align: center;">373.63 <span
class="math inline">±</span> 129.2</td>
<td style="text-align: center;">- <span class="math inline">±</span>
-</td>
<td style="text-align: center;"><strong>3806.39 <span
class="math inline">±</span> 1547.48</strong></td>
<td style="text-align: center;">3748.26 <span
class="math inline">±</span> 1832.83</td>
<td style="text-align: center;">3304.35 <span
class="math inline">±</span> 706.13</td>
</tr>
<thead>
<tr class="header">
<th style="text-align: left;"></th>
<th colspan="3" style="text-align: center;"><strong>NaturalPG</strong></th>
<th colspan="3" style="text-align: center;"><strong>TRPO</strong></th>
</tr>
</thead>
<tr class="odd">
<td style="text-align: left;"><strong>Environment</strong></td>
<td style="text-align: center;"><strong>OmniSafe (Ours)</strong></td>
<td style="text-align: center;"><strong>Tianshou</strong></td>
<td style="text-align: center;"><strong>Stable-Baselines3</strong></td>
<td style="text-align: center;"><strong>OmniSafe (Ours)</strong></td>
<td style="text-align: center;"><strong>Tianshou</strong></td>
<td style="text-align: center;"><strong>Stable-Baselines3</strong></td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyAntVelocity-v1</span></td>
<td style="text-align: center;"><strong>3793.70 <span
class="math inline">±</span> 583.66</strong></td>
<td style="text-align: center;">2062.45 <span
class="math inline">±</span> 876.43</td>
<td style="text-align: center;">- <span class="math inline">±</span>
-</td>
<td style="text-align: center;"><strong>4362.43 <span
class="math inline">±</span> 640.54</strong></td>
<td style="text-align: center;">2521.36 <span
class="math inline">±</span> 1442.10</td>
<td style="text-align: center;">3233.58 <span
class="math inline">±</span> 1437.16</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyHalfCheetahVelocity-v1</span></td>
<td style="text-align: center;"><strong>4096.77 <span
class="math inline">±</span> 1223.70</strong></td>
<td style="text-align: center;">3430.9 <span
class="math inline">±</span> 239.38</td>
<td style="text-align: center;">- <span class="math inline">±</span>
-</td>
<td style="text-align: center;">3313.31 <span
class="math inline">±</span> 1048.78</td>
<td style="text-align: center;">4255.73 <span
class="math inline">±</span> 1053.82</td>
<td style="text-align: center;"><strong>7185.06 <span
class="math inline">±</span> 3650.82</strong></td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyHopperVelocity-v1</span></td>
<td style="text-align: center;"><strong>2590.54 <span
class="math inline">±</span> 631.05</strong></td>
<td style="text-align: center;">993.63 <span
class="math inline">±</span> 489.42</td>
<td style="text-align: center;">- <span class="math inline">±</span>
-</td>
<td style="text-align: center;"><strong>2698.19 <span
class="math inline">±</span> 568.80</strong></td>
<td style="text-align: center;">1346.94 <span
class="math inline">±</span> 984.09</td>
<td style="text-align: center;">2467.10 <span
class="math inline">±</span> 1160.25</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyHumanoidVelocity-v1</span></td>
<td style="text-align: center;"><strong>3838.67 <span
class="math inline">±</span> 1654.79</strong></td>
<td style="text-align: center;">810.76 <span
class="math inline">±</span> 270.69</td>
<td style="text-align: center;">- <span class="math inline">±</span>
-</td>
<td style="text-align: center;">1461.51 <span
class="math inline">±</span> 602.23</td>
<td style="text-align: center;">749.42 <span
class="math inline">±</span> 149.81</td>
<td style="text-align: center;"><strong>2828.18 <span
class="math inline">±</span> 2256.38</strong></td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetySwimmerVelocity-v1</span></td>
<td style="text-align: center;"><strong>116.33 <span
class="math inline">±</span> 5.97</strong></td>
<td style="text-align: center;">29.75 <span class="math inline">±</span>
12.00</td>
<td style="text-align: center;">- <span class="math inline">±</span>
-</td>
<td style="text-align: center;">105.08 <span
class="math inline">±</span> 31.00</td>
<td style="text-align: center;">37.21 <span class="math inline">±</span>
4.04</td>
<td style="text-align: center;"><strong>258.62 <span
class="math inline">±</span> 124.91</strong></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyWalker2dVelocity-v1</span></td>
<td style="text-align: center;"><strong>4054.62 <span
class="math inline">±</span> 1266.76</strong></td>
<td style="text-align: center;">3372.59 <span
class="math inline">±</span> 1049.14</td>
<td style="text-align: center;">- <span class="math inline">±</span>
-</td>
<td style="text-align: center;">4099.97 <span
class="math inline">±</span> 409.05</td>
<td style="text-align: center;">3372.59 <span
class="math inline">±</span> 961.74</td>
<td style="text-align: center;"><strong>4227.91 <span
class="math inline">±</span> 760.93</strong></td>
</tr>
</tbody>
  <caption><p style="font-family: 'Times New Roman', Times, serif;"><b>Table 1:</b>The performance of OmniSafe, which was evaluated in relation to published baselines within the Safety-Gymnasium MuJoCo Velocity environments. Experimental outcomes, comprising mean and standard deviation, were derived from 10 assessment iterations encompassing multiple random seeds. A noteworthy distinction lies in the fact that Stable-Baselines3 employs distinct parameters tailored to each environment, while OmniSafe maintains a consistent parameter set across all environments.</p></caption>
</table>

### Safe Reinforcement Learning Algorithms

To demonstrate the high reliability of the algorithms implemented, OmniSafe offers performance insights within the Safety-Gymnasium environment. It should be noted that all data is procured under the constraint of `cost_limit=25.00`. The results are presented in <a href="#performance_on_policy">Table 2</a> and the training curves are in the following sections (Please click the triangle button to see the training curves).

#### Performance Table

<table id="performance_on_policy">
<thead>
<tr class="header">
<th style="text-align: left;"></th>
<th colspan="2" style="text-align: center;"><strong>Policy
Gradient</strong></th>
<th colspan="2" style="text-align: center;"><strong>Natural
PG</strong></th>
<th colspan="2" style="text-align: center;"><strong>TRPO</strong></th>
<th colspan="2" style="text-align: center;"><strong>PPO</strong></th>
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
class="smallcaps">SafetyAntVelocity-v1</span></td>
<td style="text-align: center;">5292.29 <span
class="math inline">±</span> 913.44</td>
<td style="text-align: center;">919.42 <span
class="math inline">±</span> 158.61</td>
<td style="text-align: center;">5547.20 <span
class="math inline">±</span> 807.89</td>
<td style="text-align: center;">895.56 <span
class="math inline">±</span> 77.13</td>
<td style="text-align: center;">6026.79 <span
class="math inline">±</span> 314.98</td>
<td style="text-align: center;">933.46 <span
class="math inline">±</span> 41.28</td>
<td style="text-align: center;">5977.73 <span
class="math inline">±</span> 885.65</td>
<td style="text-align: center;">958.13 <span
class="math inline">±</span> 134.5</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyHalfCheetahVelocity-v1</span></td>
<td style="text-align: center;">5188.46 <span
class="math inline">±</span> 1202.76</td>
<td style="text-align: center;">896.55 <span
class="math inline">±</span> 184.7</td>
<td style="text-align: center;">5878.28 <span
class="math inline">±</span> 2012.24</td>
<td style="text-align: center;">847.74 <span
class="math inline">±</span> 249.02</td>
<td style="text-align: center;">6490.76 <span
class="math inline">±</span> 2507.18</td>
<td style="text-align: center;">734.26 <span
class="math inline">±</span> 321.88</td>
<td style="text-align: center;">6921.83 <span
class="math inline">±</span> 1721.79</td>
<td style="text-align: center;">919.2 <span class="math inline">±</span>
173.08</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyHopperVelocity-v1</span></td>
<td style="text-align: center;">3218.17 <span
class="math inline">±</span> 672.88</td>
<td style="text-align: center;">881.76 <span
class="math inline">±</span> 198.46</td>
<td style="text-align: center;">2613.95 <span
class="math inline">±</span> 866.13</td>
<td style="text-align: center;">587.78 <span
class="math inline">±</span> 220.97</td>
<td style="text-align: center;">2047.35 <span
class="math inline">±</span> 447.33</td>
<td style="text-align: center;">448.12 <span
class="math inline">±</span> 103.87</td>
<td style="text-align: center;">2337.11 <span
class="math inline">±</span> 942.06</td>
<td style="text-align: center;">550.02 <span
class="math inline">±</span> 237.70</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyHumanoidVelocity-v1</span></td>
<td style="text-align: center;">7001.78 <span
class="math inline">±</span> 419.67</td>
<td style="text-align: center;">834.11 <span
class="math inline">±</span> 212.43</td>
<td style="text-align: center;">8055.20 <span
class="math inline">±</span> 641.67</td>
<td style="text-align: center;">946.40 <span
class="math inline">±</span> 9.11</td>
<td style="text-align: center;">8681.24 <span
class="math inline">±</span> 3934.08</td>
<td style="text-align: center;">718.42 <span
class="math inline">±</span> 323.30</td>
<td style="text-align: center;">9115.93 <span
class="math inline">±</span> 596.88</td>
<td style="text-align: center;">960.44 <span
class="math inline">±</span> 7.06</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetySwimmerVelocity-v1</span></td>
<td style="text-align: center;">77.05 <span class="math inline">±</span>
33.44</td>
<td style="text-align: center;">107.1 <span class="math inline">±</span>
60.58</td>
<td style="text-align: center;">120.19 <span
class="math inline">±</span> 7.74</td>
<td style="text-align: center;">161.78 <span
class="math inline">±</span> 17.51</td>
<td style="text-align: center;">124.91 <span
class="math inline">±</span> 6.13</td>
<td style="text-align: center;">176.56 <span
class="math inline">±</span> 15.95</td>
<td style="text-align: center;">119.77 <span
class="math inline">±</span> 13.8</td>
<td style="text-align: center;">165.27 <span
class="math inline">±</span> 20.15</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyWalker2dVelocity-v1</span></td>
<td style="text-align: center;">4832.34 <span
class="math inline">±</span> 685.76</td>
<td style="text-align: center;">866.59 <span
class="math inline">±</span> 93.47</td>
<td style="text-align: center;">5347.35 <span
class="math inline">±</span> 436.86</td>
<td style="text-align: center;">914.74 <span
class="math inline">±</span> 32.61</td>
<td style="text-align: center;">6096.67 <span
class="math inline">±</span> 723.06</td>
<td style="text-align: center;">914.46 <span
class="math inline">±</span> 27.85</td>
<td style="text-align: center;">6239.52 <span
class="math inline">±</span> 879.99</td>
<td style="text-align: center;">902.68 <span
class="math inline">±</span> 100.93</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyCarGoal1-v0</span></td>
<td style="text-align: center;">35.86 <span class="math inline">±</span>
1.97</td>
<td style="text-align: center;">57.46 <span class="math inline">±</span>
48.34</td>
<td style="text-align: center;">36.07 <span class="math inline">±</span>
1.25</td>
<td style="text-align: center;">58.06 <span class="math inline">±</span>
10.03</td>
<td style="text-align: center;">36.60 <span class="math inline">±</span>
0.22</td>
<td style="text-align: center;">55.58 <span class="math inline">±</span>
12.68</td>
<td style="text-align: center;">33.41 <span class="math inline">±</span>
2.89</td>
<td style="text-align: center;">58.06 <span class="math inline">±</span>
42.06</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyCarButton1-v0</span></td>
<td style="text-align: center;">19.76 <span class="math inline">±</span>
10.15</td>
<td style="text-align: center;">353.26 <span
class="math inline">±</span> 177.08</td>
<td style="text-align: center;">22.16 <span class="math inline">±</span>
4.48</td>
<td style="text-align: center;">333.98 <span
class="math inline">±</span> 67.49</td>
<td style="text-align: center;">21.98 <span class="math inline">±</span>
2.06</td>
<td style="text-align: center;">343.22 <span
class="math inline">±</span> 24.60</td>
<td style="text-align: center;">17.51 <span class="math inline">±</span>
9.46</td>
<td style="text-align: center;">373.98 <span
class="math inline">±</span> 156.64</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyCarGoal2-v0</span></td>
<td style="text-align: center;">29.43 <span class="math inline">±</span>
4.62</td>
<td style="text-align: center;">179.2 <span class="math inline">±</span>
84.86</td>
<td style="text-align: center;">30.26 <span class="math inline">±</span>
0.38</td>
<td style="text-align: center;">209.62 <span
class="math inline">±</span> 29.97</td>
<td style="text-align: center;">32.17 <span class="math inline">±</span>
1.24</td>
<td style="text-align: center;">190.74 <span
class="math inline">±</span> 21.05</td>
<td style="text-align: center;">29.88 <span class="math inline">±</span>
4.55</td>
<td style="text-align: center;">194.16 <span
class="math inline">±</span> 106.2</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyCarButton2-v0</span></td>
<td style="text-align: center;">18.06 <span class="math inline">±</span>
10.53</td>
<td style="text-align: center;">349.82 <span
class="math inline">±</span> 187.07</td>
<td style="text-align: center;">20.85 <span class="math inline">±</span>
3.14</td>
<td style="text-align: center;">313.88 <span
class="math inline">±</span> 58.20</td>
<td style="text-align: center;">20.51 <span class="math inline">±</span>
3.34</td>
<td style="text-align: center;">316.42 <span
class="math inline">±</span> 35.28</td>
<td style="text-align: center;">21.35 <span class="math inline">±</span>
8.22</td>
<td style="text-align: center;">312.64 <span
class="math inline">±</span> 138.4</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyPointGoal1-v0</span></td>
<td style="text-align: center;">26.19 <span class="math inline">±</span>
3.44</td>
<td style="text-align: center;">201.22 <span
class="math inline">±</span> 80.4</td>
<td style="text-align: center;">26.92 <span class="math inline">±</span>
0.58</td>
<td style="text-align: center;">57.92 <span class="math inline">±</span>
9.97</td>
<td style="text-align: center;">27.20 <span class="math inline">±</span>
0.44</td>
<td style="text-align: center;">45.88 <span class="math inline">±</span>
11.27</td>
<td style="text-align: center;">25.44 <span class="math inline">±</span>
5.43</td>
<td style="text-align: center;">55.72 <span class="math inline">±</span>
35.55</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyPointButton1-v0</span></td>
<td style="text-align: center;">29.98 <span class="math inline">±</span>
5.24</td>
<td style="text-align: center;">141.74 <span
class="math inline">±</span> 75.13</td>
<td style="text-align: center;">31.95 <span class="math inline">±</span>
1.53</td>
<td style="text-align: center;">123.98 <span
class="math inline">±</span> 32.05</td>
<td style="text-align: center;">30.61 <span class="math inline">±</span>
0.40</td>
<td style="text-align: center;">134.38 <span
class="math inline">±</span> 22.06</td>
<td style="text-align: center;">27.03 <span class="math inline">±</span>
6.14</td>
<td style="text-align: center;">152.48 <span
class="math inline">±</span> 80.39</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyPointGoal2-v0</span></td>
<td style="text-align: center;">25.18 <span class="math inline">±</span>
3.62</td>
<td style="text-align: center;">204.96 <span
class="math inline">±</span> 104.97</td>
<td style="text-align: center;">26.19 <span class="math inline">±</span>
0.84</td>
<td style="text-align: center;">193.60 <span
class="math inline">±</span> 18.54</td>
<td style="text-align: center;">25.61 <span class="math inline">±</span>
0.89</td>
<td style="text-align: center;">202.26 <span
class="math inline">±</span> 15.15</td>
<td style="text-align: center;">25.49 <span class="math inline">±</span>
2.46</td>
<td style="text-align: center;">159.28 <span
class="math inline">±</span> 87.13</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyPointButton2-v0</span></td>
<td style="text-align: center;">26.88 <span class="math inline">±</span>
4.38</td>
<td style="text-align: center;">153.88 <span
class="math inline">±</span> 65.54</td>
<td style="text-align: center;">28.45 <span class="math inline">±</span>
1.49</td>
<td style="text-align: center;">160.40 <span
class="math inline">±</span> 20.08</td>
<td style="text-align: center;">28.78 <span class="math inline">±</span>
2.05</td>
<td style="text-align: center;">170.30 <span
class="math inline">±</span> 30.59</td>
<td style="text-align: center;">25.91 <span class="math inline">±</span>
6.15</td>
<td style="text-align: center;">166.6 <span class="math inline">±</span>
111.21</td>
</tr>
<thead>
<tr class="header">
<th style="text-align: left;"></th>
<th colspan="2" style="text-align: center;"><strong>RCPO</strong></th>
<th colspan="2" style="text-align: center;"><strong>TRPOLag</strong></th>
<th colspan="2" style="text-align: center;"><strong>PPOLag</strong></th>
<th colspan="2" style="text-align: center;"><strong>P3O</strong></th>
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
<td style="text-align: center;"><strong>Reward</strong></td>
<td style="text-align: center;"><strong>Cost</strong></td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyAntVelocity-v1</span></td>
<td style="text-align: center;">3139.52 <span
class="math inline">±</span> 110.34</td>
<td style="text-align: center;">12.34 <span class="math inline">±</span>
3.11</td>
<td style="text-align: center;">3041.89 <span
class="math inline">±</span> 180.77</td>
<td style="text-align: center;">19.52 <span class="math inline">±</span>
20.21</td>
<td style="text-align: center;">3261.87 <span
class="math inline">±</span> 80.00</td>
<td style="text-align: center;">12.05 <span class="math inline">±</span>
6.57</td>
<td style="text-align: center;">2636.62 <span
class="math inline">±</span> 181.09</td>
<td style="text-align: center;">20.69 <span class="math inline">±</span>
10.23</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyHalfCheetahVelocity-v1</span></td>
<td style="text-align: center;">2440.97 <span
class="math inline">±</span> 451.88</td>
<td style="text-align: center;">9.02 <span class="math inline">±</span>
9.34</td>
<td style="text-align: center;">2884.68 <span
class="math inline">±</span> 77.47</td>
<td style="text-align: center;">9.04 <span class="math inline">±</span>
11.83</td>
<td style="text-align: center;">2946.15 <span
class="math inline">±</span> 306.35</td>
<td style="text-align: center;">3.44 <span class="math inline">±</span>
4.77</td>
<td style="text-align: center;">2117.84 <span
class="math inline">±</span> 313.55</td>
<td style="text-align: center;">27.6 <span class="math inline">±</span>
8.36</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyHopperVelocity-v1</span></td>
<td style="text-align: center;">1428.58 <span
class="math inline">±</span> 199.87</td>
<td style="text-align: center;">11.12 <span class="math inline">±</span>
12.66</td>
<td style="text-align: center;">1391.79 <span
class="math inline">±</span> 269.07</td>
<td style="text-align: center;">11.22 <span class="math inline">±</span>
9.97</td>
<td style="text-align: center;">961.92 <span
class="math inline">±</span> 752.87</td>
<td style="text-align: center;">13.96 <span class="math inline">±</span>
19.33</td>
<td style="text-align: center;">1231.52 <span
class="math inline">±</span> 465.35</td>
<td style="text-align: center;">16.33 <span class="math inline">±</span>
11.38</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyHumanoidVelocity-v1</span></td>
<td style="text-align: center;">6286.51 <span
class="math inline">±</span> 151.03</td>
<td style="text-align: center;">19.47 <span class="math inline">±</span>
7.74</td>
<td style="text-align: center;">6551.30 <span
class="math inline">±</span> 58.42</td>
<td style="text-align: center;">59.56 <span class="math inline">±</span>
117.37</td>
<td style="text-align: center;">6624.46 <span
class="math inline">±</span> 25.9</td>
<td style="text-align: center;">5.87 <span class="math inline">±</span>
9.46</td>
<td style="text-align: center;">6342.47 <span
class="math inline">±</span> 82.45</td>
<td style="text-align: center;">126.4 <span class="math inline">±</span>
193.76</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetySwimmerVelocity-v1</span></td>
<td style="text-align: center;">61.29 <span class="math inline">±</span>
18.12</td>
<td style="text-align: center;">22.60 <span class="math inline">±</span>
1.16</td>
<td style="text-align: center;">81.18 <span class="math inline">±</span>
16.33</td>
<td style="text-align: center;">22.24 <span class="math inline">±</span>
3.91</td>
<td style="text-align: center;">64.74 <span class="math inline">±</span>
17.67</td>
<td style="text-align: center;">28.02 <span class="math inline">±</span>
4.09</td>
<td style="text-align: center;">38.02 <span class="math inline">±</span>
34.18</td>
<td style="text-align: center;">18.4 <span class="math inline">±</span>
12.13</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyWalker2dVelocity-v1</span></td>
<td style="text-align: center;">3064.43 <span
class="math inline">±</span> 218.83</td>
<td style="text-align: center;">3.02 <span class="math inline">±</span>
1.48</td>
<td style="text-align: center;">3207.10 <span
class="math inline">±</span> 7.88</td>
<td style="text-align: center;">14.98 <span class="math inline">±</span>
9.27</td>
<td style="text-align: center;">2982.27 <span
class="math inline">±</span> 681.55</td>
<td style="text-align: center;">13.49 <span class="math inline">±</span>
14.55</td>
<td style="text-align: center;">2713.57 <span
class="math inline">±</span> 313.2</td>
<td style="text-align: center;">20.51 <span class="math inline">±</span>
14.09</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyCarGoal1-v0</span></td>
<td style="text-align: center;">18.71 <span class="math inline">±</span>
2.72</td>
<td style="text-align: center;">23.10 <span class="math inline">±</span>
12.57</td>
<td style="text-align: center;">27.04 <span class="math inline">±</span>
1.82</td>
<td style="text-align: center;">26.80 <span class="math inline">±</span>
5.64</td>
<td style="text-align: center;">13.27 <span class="math inline">±</span>
9.26</td>
<td style="text-align: center;">21.72 <span class="math inline">±</span>
32.06</td>
<td style="text-align: center;">-1.10 <span class="math inline">±</span>
6.851</td>
<td style="text-align: center;">50.58 <span class="math inline">±</span>
99.24</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyCarButton1-v0</span></td>
<td style="text-align: center;">-2.04 <span class="math inline">±</span>
2.98</td>
<td style="text-align: center;">43.48 <span class="math inline">±</span>
31.52</td>
<td style="text-align: center;">-0.38 <span class="math inline">±</span>
0.85</td>
<td style="text-align: center;">37.54 <span class="math inline">±</span>
31.72</td>
<td style="text-align: center;">0.33 <span class="math inline">±</span>
1.96</td>
<td style="text-align: center;">55.5 <span class="math inline">±</span>
89.64</td>
<td style="text-align: center;">-2.06 <span class="math inline">±</span>
7.2</td>
<td style="text-align: center;">43.78 <span class="math inline">±</span>
98.01</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyCarGoal2-v0</span></td>
<td style="text-align: center;">2.30 <span class="math inline">±</span>
1.76</td>
<td style="text-align: center;">22.90 <span class="math inline">±</span>
16.22</td>
<td style="text-align: center;">3.65 <span class="math inline">±</span>
1.09</td>
<td style="text-align: center;">39.98 <span class="math inline">±</span>
20.29</td>
<td style="text-align: center;">1.58 <span class="math inline">±</span>
2.49</td>
<td style="text-align: center;">13.82 <span class="math inline">±</span>
24.62</td>
<td style="text-align: center;">-0.07 <span class="math inline">±</span>
1.62</td>
<td style="text-align: center;">43.86 <span class="math inline">±</span>
99.58</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyCarButton2-v0</span></td>
<td style="text-align: center;">-1.35 <span class="math inline">±</span>
2.41</td>
<td style="text-align: center;">42.02 <span class="math inline">±</span>
31.77</td>
<td style="text-align: center;">-1.68 <span class="math inline">±</span>
2.55</td>
<td style="text-align: center;">20.36 <span class="math inline">±</span>
13.67</td>
<td style="text-align: center;">0.76 <span class="math inline">±</span>
2.52</td>
<td style="text-align: center;">47.86 <span class="math inline">±</span>
103.27</td>
<td style="text-align: center;">0.11 <span class="math inline">±</span>
0.72</td>
<td style="text-align: center;">85.94 <span class="math inline">±</span>
122.01</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyPointGoal1-v0</span></td>
<td style="text-align: center;">15.27 <span class="math inline">±</span>
4.05</td>
<td style="text-align: center;">30.56 <span class="math inline">±</span>
19.15</td>
<td style="text-align: center;">18.51 <span class="math inline">±</span>
3.83</td>
<td style="text-align: center;">22.98 <span class="math inline">±</span>
8.45</td>
<td style="text-align: center;">12.96 <span class="math inline">±</span>
6.95</td>
<td style="text-align: center;">25.80 <span class="math inline">±</span>
34.99</td>
<td style="text-align: center;">1.6 <span class="math inline">±</span>
3.01</td>
<td style="text-align: center;">31.1 <span class="math inline">±</span>
80.03</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyPointButton1-v0</span></td>
<td style="text-align: center;">3.65 <span class="math inline">±</span>
4.47</td>
<td style="text-align: center;">26.30 <span class="math inline">±</span>
9.22</td>
<td style="text-align: center;">6.93 <span class="math inline">±</span>
1.84</td>
<td style="text-align: center;">31.16 <span class="math inline">±</span>
20.58</td>
<td style="text-align: center;">4.60 <span class="math inline">±</span>
4.73</td>
<td style="text-align: center;">20.8 <span class="math inline">±</span>
35.78</td>
<td style="text-align: center;">-0.34 <span class="math inline">±</span>
1.53</td>
<td style="text-align: center;">52.86 <span class="math inline">±</span>
85.62</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyPointGoal2-v0</span></td>
<td style="text-align: center;">2.17 <span class="math inline">±</span>
1.46</td>
<td style="text-align: center;">33.82 <span class="math inline">±</span>
21.93</td>
<td style="text-align: center;">4.64 <span class="math inline">±</span>
1.43</td>
<td style="text-align: center;">26.00 <span class="math inline">±</span>
4.70</td>
<td style="text-align: center;">1.98 <span class="math inline">±</span>
3.86</td>
<td style="text-align: center;">41.20 <span class="math inline">±</span>
61.03</td>
<td style="text-align: center;">0.34 <span class="math inline">±</span>
2.2</td>
<td style="text-align: center;">65.84 <span class="math inline">±</span>
195.76</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyPointButton2-v0</span></td>
<td style="text-align: center;">7.18 <span class="math inline">±</span>
1.93</td>
<td style="text-align: center;">45.02 <span class="math inline">±</span>
25.28</td>
<td style="text-align: center;">5.43 <span class="math inline">±</span>
3.44</td>
<td style="text-align: center;">25.10 <span class="math inline">±</span>
8.98</td>
<td style="text-align: center;">0.93 <span class="math inline">±</span>
3.69</td>
<td style="text-align: center;">33.72 <span class="math inline">±</span>
58.75</td>
<td style="text-align: center;">0.33 <span class="math inline">±</span>
2.44</td>
<td style="text-align: center;">28.5 <span class="math inline">±</span>
49.79</td>
</tr>
<thead>
<tr class="header">
<th style="text-align: left;"></th>
<th colspan="2" style="text-align: center;"><strong>CUP</strong></th>
<th colspan="2" style="text-align: center;"><strong>PCPO</strong></th>
<th colspan="2" style="text-align: center;"><strong>FOCOPS</strong></th>
<th colspan="2" style="text-align: center;"><strong>CPO</strong></th>
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
<td style="text-align: center;"><strong>Reward</strong></td>
<td style="text-align: center;"><strong>Cost</strong></td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyAntVelocity-v1</span></td>
<td style="text-align: center;">3215.79 <span
class="math inline">±</span> 346.68</td>
<td style="text-align: center;">18.25 <span class="math inline">±</span>
17.12</td>
<td style="text-align: center;">2257.07 <span
class="math inline">±</span> 47.97</td>
<td style="text-align: center;">10.44 <span class="math inline">±</span>
5.22</td>
<td style="text-align: center;">3184.48 <span
class="math inline">±</span> 305.59</td>
<td style="text-align: center;">14.75 <span class="math inline">±</span>
6.36</td>
<td style="text-align: center;">3098.54 <span
class="math inline">±</span> 78.90</td>
<td style="text-align: center;">14.12 <span class="math inline">±</span>
3.41</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyHalfCheetahVelocity-v1</span></td>
<td style="text-align: center;">2850.6 <span
class="math inline">±</span> 244.65</td>
<td style="text-align: center;">4.27 <span class="math inline">±</span>
4.46</td>
<td style="text-align: center;">1677.93 <span
class="math inline">±</span> 217.31</td>
<td style="text-align: center;">19.06 <span class="math inline">±</span>
15.26</td>
<td style="text-align: center;">2965.2 <span
class="math inline">±</span> 290.43</td>
<td style="text-align: center;">2.37 <span class="math inline">±</span>
3.5</td>
<td style="text-align: center;">2786.48 <span
class="math inline">±</span> 173.45</td>
<td style="text-align: center;">4.70 <span class="math inline">±</span>
6.72</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyHopperVelocity-v1</span></td>
<td style="text-align: center;">1716.08 <span
class="math inline">±</span> 5.93</td>
<td style="text-align: center;">7.48 <span class="math inline">±</span>
5.535</td>
<td style="text-align: center;">1551.22 <span
class="math inline">±</span> 85.16</td>
<td style="text-align: center;">15.46 <span class="math inline">±</span>
9.83</td>
<td style="text-align: center;">1437.75 <span
class="math inline">±</span> 446.87</td>
<td style="text-align: center;">10.13 <span class="math inline">±</span>
8.87</td>
<td style="text-align: center;">1713.71 <span
class="math inline">±</span> 18.26</td>
<td style="text-align: center;">13.40 <span class="math inline">±</span>
5.82</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyHumanoidVelocity-v1</span></td>
<td style="text-align: center;">6109.94 <span
class="math inline">±</span> 497.56</td>
<td style="text-align: center;">24.69 <span class="math inline">±</span>
20.54</td>
<td style="text-align: center;">5852.25 <span
class="math inline">±</span> 78.01</td>
<td style="text-align: center;">0.24 <span class="math inline">±</span>
0.48</td>
<td style="text-align: center;">6489.39 <span
class="math inline">±</span> 35.1</td>
<td style="text-align: center;">13.86 <span class="math inline">±</span>
39.33</td>
<td style="text-align: center;">6465.34 <span
class="math inline">±</span> 79.87</td>
<td style="text-align: center;">0.18 <span class="math inline">±</span>
0.36</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetySwimmerVelocity-v1</span></td>
<td style="text-align: center;">63.83 <span class="math inline">±</span>
46.45</td>
<td style="text-align: center;">21.95 <span class="math inline">±</span>
11.04</td>
<td style="text-align: center;">54.42 <span class="math inline">±</span>
38.65</td>
<td style="text-align: center;">17.34 <span class="math inline">±</span>
1.57</td>
<td style="text-align: center;">53.87 <span class="math inline">±</span>
17.9</td>
<td style="text-align: center;">29.75 <span class="math inline">±</span>
7.33</td>
<td style="text-align: center;">65.30 <span class="math inline">±</span>
43.25</td>
<td style="text-align: center;">18.22 <span class="math inline">±</span>
8.01</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyWalker2dVelocity-v1</span></td>
<td style="text-align: center;">2466.95 <span
class="math inline">±</span> 1114.13</td>
<td style="text-align: center;">6.63 <span class="math inline">±</span>
8.25</td>
<td style="text-align: center;">1802.86 <span
class="math inline">±</span> 714.04</td>
<td style="text-align: center;">18.82 <span class="math inline">±</span>
5.57</td>
<td style="text-align: center;">3117.05 <span
class="math inline">±</span> 53.60</td>
<td style="text-align: center;">8.78 <span class="math inline">±</span>
12.38</td>
<td style="text-align: center;">2074.76 <span
class="math inline">±</span> 962.45</td>
<td style="text-align: center;">21.90 <span class="math inline">±</span>
9.41</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyCarGoal1-v0</span></td>
<td style="text-align: center;">6.14 <span class="math inline">±</span>
6.97</td>
<td style="text-align: center;">36.12 <span class="math inline">±</span>
89.56</td>
<td style="text-align: center;">21.56 <span class="math inline">±</span>
2.87</td>
<td style="text-align: center;">38.42 <span class="math inline">±</span>
8.36</td>
<td style="text-align: center;">15.23 <span class="math inline">±</span>
10.76</td>
<td style="text-align: center;">31.66 <span class="math inline">±</span>
93.51</td>
<td style="text-align: center;">25.52 <span class="math inline">±</span>
2.65</td>
<td style="text-align: center;">43.32 <span class="math inline">±</span>
14.35</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyCarButton1-v0</span></td>
<td style="text-align: center;">1.49 <span class="math inline">±</span>
2.84</td>
<td style="text-align: center;">103.24 <span
class="math inline">±</span> 123.12</td>
<td style="text-align: center;">0.36 <span class="math inline">±</span>
0.85</td>
<td style="text-align: center;">40.52 <span class="math inline">±</span>
21.25</td>
<td style="text-align: center;">0.21 <span class="math inline">±</span>
2.27</td>
<td style="text-align: center;">31.78 <span class="math inline">±</span>
47.03</td>
<td style="text-align: center;">0.82 <span class="math inline">±</span>
1.60</td>
<td style="text-align: center;">37.86 <span class="math inline">±</span>
27.41</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyCarGoal2-v0</span></td>
<td style="text-align: center;">1.78 <span class="math inline">±</span>
4.03</td>
<td style="text-align: center;">95.4 <span class="math inline">±</span>
129.64</td>
<td style="text-align: center;">1.62 <span class="math inline">±</span>
0.56</td>
<td style="text-align: center;">48.12 <span class="math inline">±</span>
31.19</td>
<td style="text-align: center;">2.09 <span class="math inline">±</span>
4.33</td>
<td style="text-align: center;">31.56 <span class="math inline">±</span>
58.93</td>
<td style="text-align: center;">3.56 <span class="math inline">±</span>
0.92</td>
<td style="text-align: center;">32.66 <span class="math inline">±</span>
3.31</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyCarButton2-v0</span></td>
<td style="text-align: center;">1.49 <span class="math inline">±</span>
2.64</td>
<td style="text-align: center;">173.68 <span
class="math inline">±</span> 163.77</td>
<td style="text-align: center;">0.66 <span class="math inline">±</span>
0.42</td>
<td style="text-align: center;">49.72 <span class="math inline">±</span>
36.50</td>
<td style="text-align: center;">1.14 <span class="math inline">±</span>
3.18</td>
<td style="text-align: center;">46.78 <span class="math inline">±</span>
57.47</td>
<td style="text-align: center;">0.17 <span class="math inline">±</span>
1.19</td>
<td style="text-align: center;">48.56 <span class="math inline">±</span>
29.34</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyPointGoal1-v0</span></td>
<td style="text-align: center;">14.42 <span class="math inline">±</span>
6.74</td>
<td style="text-align: center;">19.02 <span class="math inline">±</span>
20.08</td>
<td style="text-align: center;">18.57 <span class="math inline">±</span>
1.71</td>
<td style="text-align: center;">22.98 <span class="math inline">±</span>
6.56</td>
<td style="text-align: center;">14.97 <span class="math inline">±</span>
9.01</td>
<td style="text-align: center;">33.72 <span class="math inline">±</span>
42.24</td>
<td style="text-align: center;">20.46 <span class="math inline">±</span>
1.38</td>
<td style="text-align: center;">28.84 <span class="math inline">±</span>
7.76</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyPointButton1-v0</span></td>
<td style="text-align: center;">3.5 <span class="math inline">±</span>
7.07</td>
<td style="text-align: center;">39.56 <span class="math inline">±</span>
54.26</td>
<td style="text-align: center;">2.66 <span class="math inline">±</span>
1.83</td>
<td style="text-align: center;">49.40 <span class="math inline">±</span>
36.76</td>
<td style="text-align: center;">5.89 <span class="math inline">±</span>
7.66</td>
<td style="text-align: center;">38.24 <span class="math inline">±</span>
42.96</td>
<td style="text-align: center;">4.04 <span class="math inline">±</span>
4.54</td>
<td style="text-align: center;">40.00 <span class="math inline">±</span>
4.52</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyPointGoal2-v0</span></td>
<td style="text-align: center;">1.06 <span class="math inline">±</span>
2.67</td>
<td style="text-align: center;">107.3 <span class="math inline">±</span>
204.26</td>
<td style="text-align: center;">1.06 <span class="math inline">±</span>
0.69</td>
<td style="text-align: center;">51.92 <span class="math inline">±</span>
47.40</td>
<td style="text-align: center;">2.21 <span class="math inline">±</span>
4.15</td>
<td style="text-align: center;">37.92 <span class="math inline">±</span>
111.81</td>
<td style="text-align: center;">2.50 <span class="math inline">±</span>
1.25</td>
<td style="text-align: center;">40.84 <span class="math inline">±</span>
23.31</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyPointButton2-v0</span></td>
<td style="text-align: center;">2.88 <span class="math inline">±</span>
3.65</td>
<td style="text-align: center;">54.24 <span class="math inline">±</span>
71.07</td>
<td style="text-align: center;">1.05 <span class="math inline">±</span>
1.27</td>
<td style="text-align: center;">41.14 <span class="math inline">±</span>
12.35</td>
<td style="text-align: center;">2.43 <span class="math inline">±</span>
3.33</td>
<td style="text-align: center;">17.92 <span class="math inline">±</span>
26.1</td>
<td style="text-align: center;">5.09 <span class="math inline">±</span>
1.83</td>
<td style="text-align: center;">48.92 <span class="math inline">±</span>
17.79</td>
</tr>
<thead>
<tr class="header">
<th style="text-align: left;"></th>
<th colspan="2" style="text-align: center;"><strong>PPOSaute</strong></th>
<th colspan="2" style="text-align: center;"><strong>TRPOSaute</strong></th>
<th colspan="2" style="text-align: center;"><strong>PPOSimmerPID</strong></th>
<th colspan="2" style="text-align: center;"><strong>TRPOSimmerPID</strong></th>
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
<td style="text-align: center;"><strong>Reward</strong></td>
<td style="text-align: center;"><strong>Cost</strong></td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyAntVelocity-v1</span></td>
<td style="text-align: center;">2978.74 <span
class="math inline">±</span> 93.65</td>
<td style="text-align: center;">16.77 <span class="math inline">±</span>
0.92</td>
<td style="text-align: center;">2507.65 <span
class="math inline">±</span> 63.97</td>
<td style="text-align: center;">8.036 <span class="math inline">±</span>
0.39</td>
<td style="text-align: center;">2944.84 <span
class="math inline">±</span> 60.53</td>
<td style="text-align: center;">16.20 <span class="math inline">±</span>
0.66</td>
<td style="text-align: center;">3018.95 <span
class="math inline">±</span> 66.44</td>
<td style="text-align: center;">16.52 <span class="math inline">±</span>
0.23</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyHalfCheetahVelocity-v1</span></td>
<td style="text-align: center;">2901.40 <span
class="math inline">±</span> 25.49</td>
<td style="text-align: center;">16.20 <span
class="math inline">±</span> 0.60</td>
<td style="text-align: center;">2521.80 <span
class="math inline">±</span> 477.29</td>
<td style="text-align: center;">7.61 <span class="math inline">±</span>
0.39</td>
<td style="text-align: center;">2922.17 <span
class="math inline">±</span> 24.84</td>
<td style="text-align: center;">16.14 <span class="math inline">±</span>
0.14</td>
<td style="text-align: center;">2737.79 <span
class="math inline">±</span> 37.53</td>
<td style="text-align: center;">16.44 <span class="math inline">±</span>
0.21</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyHopperVelocity-v1</span></td>
<td style="text-align: center;">1650.91 <span
class="math inline">±</span> 152.65</td>
<td style="text-align: center;">17.87 <span class="math inline">±</span>
1.33</td>
<td style="text-align: center;">1368.28 <span
class="math inline">±</span> 576.08</td>
<td style="text-align: center;">10.38 <span class="math inline">±</span>
4.38</td>
<td style="text-align: center;">1699.94 <span
class="math inline">±</span> 24.25</td>
<td style="text-align: center;">17.04 <span class="math inline">±</span>
0.41</td>
<td style="text-align: center;">1608.41 <span
class="math inline">±</span> 88.23</td>
<td style="text-align: center;">16.30 <span class="math inline">±</span>
0.30</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyHumanoidVelocity-v1</span></td>
<td style="text-align: center;">6401.00 <span
class="math inline">±</span> 32.23</td>
<td style="text-align: center;">17.10 <span class="math inline">±</span>
2.41</td>
<td style="text-align: center;">5759.44 <span
class="math inline">±</span> 75.73</td>
<td style="text-align: center;">15.84 <span class="math inline">±</span>
1.42</td>
<td style="text-align: center;">6401.85 <span
class="math inline">±</span> 57.62</td>
<td style="text-align: center;">11.06 <span class="math inline">±</span>
5.35</td>
<td style="text-align: center;">6411.32 <span
class="math inline">±</span> 44.26</td>
<td style="text-align: center;">13.04 <span class="math inline">±</span>
2.68</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetySwimmerVelocity-v1</span></td>
<td style="text-align: center;">35.61 <span class="math inline">±</span>
4.37</td>
<td style="text-align: center;">3.44 <span class="math inline">±</span>
1.35</td>
<td style="text-align: center;">34.72 <span class="math inline">±</span>
1.37</td>
<td style="text-align: center;">10.19 <span class="math inline">±</span>
2.32</td>
<td style="text-align: center;">77.52 <span class="math inline">±</span>
40.20</td>
<td style="text-align: center;">0.98 <span class="math inline">±</span>
1.91</td>
<td style="text-align: center;">51.39 <span class="math inline">±</span>
40.09</td>
<td style="text-align: center;">0.00 <span class="math inline">±</span>
0.00</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyWalker2dVelocity-v1</span></td>
<td style="text-align: center;">2410.89 <span
class="math inline">±</span> 241.22</td>
<td style="text-align: center;">18.88 <span class="math inline">±</span>
2.38</td>
<td style="text-align: center;">2548.82 <span
class="math inline">±</span> 891.65</td>
<td style="text-align: center;">13.21 <span class="math inline">±</span>
6.09</td>
<td style="text-align: center;">3187.56 <span
class="math inline">±</span> 32.66</td>
<td style="text-align: center;">17.10 <span class="math inline">±</span>
0.49</td>
<td style="text-align: center;">3156.99 <span
class="math inline">±</span> 30.93</td>
<td style="text-align: center;">17.14 <span class="math inline">±</span>
0.54</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyCarGoal1-v0</span></td>
<td style="text-align: center;">7.12 <span class="math inline">±</span>
5.41</td>
<td style="text-align: center;">21.68 <span class="math inline">±</span>
29.11</td>
<td style="text-align: center;">16.67 <span class="math inline">±</span>
10.57</td>
<td style="text-align: center;">23.58 <span class="math inline">±</span>
26.39</td>
<td style="text-align: center;">8.45 <span class="math inline">±</span>
7.16</td>
<td style="text-align: center;">18.98 <span class="math inline">±</span>
25.63</td>
<td style="text-align: center;">15.08 <span class="math inline">±</span>
13.41</td>
<td style="text-align: center;">23.22 <span class="math inline">±</span>
19.80</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyCarButton1-v0</span></td>
<td style="text-align: center;">-1.72 <span class="math inline">±</span>
0.89</td>
<td style="text-align: center;">51.88 <span class="math inline">±</span>
28.18</td>
<td style="text-align: center;">-2.03 <span class="math inline">±</span>
0.40</td>
<td style="text-align: center;">6.24 <span class="math inline">±</span>
6.14</td>
<td style="text-align: center;">-0.57 <span class="math inline">±</span>
0.63</td>
<td style="text-align: center;">49.14 <span class="math inline">±</span>
37.77</td>
<td style="text-align: center;">-1.24 <span class="math inline">±</span>
0.47</td>
<td style="text-align: center;">17.26 <span class="math inline">±</span>
16.13</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyCarGoal2-v0</span></td>
<td style="text-align: center;">0.90 <span class="math inline">±</span>
1.20</td>
<td style="text-align: center;">19.98 <span class="math inline">±</span>
10.12</td>
<td style="text-align: center;">1.76 <span class="math inline">±</span>
5.20</td>
<td style="text-align: center;">31.50 <span class="math inline">±</span>
45.50</td>
<td style="text-align: center;">1.02 <span class="math inline">±</span>
1.41</td>
<td style="text-align: center;">27.32 <span class="math inline">±</span>
60.12</td>
<td style="text-align: center;">0.93 <span class="math inline">±</span>
2.21</td>
<td style="text-align: center;">26.66 <span class="math inline">±</span>
60.07</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyCarButton2-v0</span></td>
<td style="text-align: center;">-1.89 <span class="math inline">±</span>
1.86</td>
<td style="text-align: center;">47.33 <span class="math inline">±</span>
28.90</td>
<td style="text-align: center;">-2.60 <span class="math inline">±</span>
0.40</td>
<td style="text-align: center;">74.57 <span class="math inline">±</span>
84.95</td>
<td style="text-align: center;">-1.31 <span class="math inline">±</span>
0.93</td>
<td style="text-align: center;">52.33 <span class="math inline">±</span>
19.96</td>
<td style="text-align: center;">-0.99 <span class="math inline">±</span>
0.63</td>
<td style="text-align: center;">20.40 <span class="math inline">±</span>
12.77</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyPointGoal1-v0</span></td>
<td style="text-align: center;">7.06 <span class="math inline">±</span>
5.85</td>
<td style="text-align: center;">20.04 <span class="math inline">±</span>
21.91</td>
<td style="text-align: center;">16.18 <span class="math inline">±</span>
9.55</td>
<td style="text-align: center;">29.94 <span class="math inline">±</span>
26.68</td>
<td style="text-align: center;">8.30 <span class="math inline">±</span>
6.03</td>
<td style="text-align: center;">25.32 <span class="math inline">±</span>
31.91</td>
<td style="text-align: center;">11.64 <span class="math inline">±</span>
8.46</td>
<td style="text-align: center;">30.00 <span class="math inline">±</span>
27.67</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyPointButton1-v0</span></td>
<td style="text-align: center;">-1.47 <span class="math inline">±</span>
0.98</td>
<td style="text-align: center;">22.60 <span class="math inline">±</span>
13.91</td>
<td style="text-align: center;">-3.13 <span class="math inline">±</span>
3.51</td>
<td style="text-align: center;">9.04 <span class="math inline">±</span>
3.94</td>
<td style="text-align: center;">-1.97 <span class="math inline">±</span>
1.41</td>
<td style="text-align: center;">12.80 <span class="math inline">±</span>
7.84</td>
<td style="text-align: center;">-1.36 <span class="math inline">±</span>
0.37</td>
<td style="text-align: center;">2.14 <span class="math inline">±</span>
1.73</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyPointGoal2-v0</span></td>
<td style="text-align: center;">0.84 <span class="math inline">±</span>
2.93</td>
<td style="text-align: center;">14.06 <span class="math inline">±</span>
30.21</td>
<td style="text-align: center;">1.64 <span class="math inline">±</span>
4.02</td>
<td style="text-align: center;">19.00 <span class="math inline">±</span>
34.69</td>
<td style="text-align: center;">0.56 <span class="math inline">±</span>
2.52</td>
<td style="text-align: center;">12.36 <span class="math inline">±</span>
43.39</td>
<td style="text-align: center;">1.55 <span class="math inline">±</span>
4.68</td>
<td style="text-align: center;">14.90 <span class="math inline">±</span>
27.82</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyPointButton2-v0</span></td>
<td style="text-align: center;">-1.38 <span class="math inline">±</span>
0.11</td>
<td style="text-align: center;">12.00 <span class="math inline">±</span>
8.60</td>
<td style="text-align: center;">-2.56 <span class="math inline">±</span>
0.67</td>
<td style="text-align: center;">17.27 <span class="math inline">±</span>
10.01</td>
<td style="text-align: center;">-1.70 <span class="math inline">±</span>
0.29</td>
<td style="text-align: center;">7.90 <span class="math inline">±</span>
3.30</td>
<td style="text-align: center;">-1.66 <span class="math inline">±</span>
0.99</td>
<td style="text-align: center;">6.70 <span class="math inline">±</span>
4.74</td>
</tr>
<thead>
<tr class="header">
<th style="text-align: left;"></th>
<th colspan="2" style="text-align: center;"><strong>CPPOPID</strong></th>
<th colspan="2" style="text-align: center;"><strong>TRPOPID</strong></th>
<th colspan="2" style="text-align: center;"><strong>PPOEarlyTerminated</strong></th>
<th colspan="2" style="text-align: center;"><strong>TRPOEarlyTerminated</strong></th>
</tr>
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
class="smallcaps">SafetyAntVelocity-v1</span></td>
<td style="text-align: center;">3213.36 <span
class="math inline">±</span> 146.78</td>
<td style="text-align: center;">14.30 <span class="math inline">±</span>
7.39</td>
<td style="text-align: center;">3052.94 <span
class="math inline">±</span> 139.67</td>
<td style="text-align: center;">15.22 <span class="math inline">±</span>
3.68</td>
<td style="text-align: center;">2801.53 <span
class="math inline">±</span> 19.66</td>
<td style="text-align: center;">0.23 <span class="math inline">±</span>
0.09</td>
<td style="text-align: center;">3052.63 <span
class="math inline">±</span> 58.41</td>
<td style="text-align: center;">0.40 <span class="math inline">±</span>
0.23</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyHalfCheetahVelocity-v1</span></td>
<td style="text-align: center;">2837.89 <span
class="math inline">±</span> 398.52</td>
<td style="text-align: center;">8.06 <span class="math inline">±</span>
9.62</td>
<td style="text-align: center;">2796.75 <span
class="math inline">±</span> 190.84</td>
<td style="text-align: center;">11.16 <span class="math inline">±</span>
9.80</td>
<td style="text-align: center;">2447.25 <span
class="math inline">±</span> 346.84</td>
<td style="text-align: center;">3.47 <span class="math inline">±</span>
4.90</td>
<td style="text-align: center;">2555.70 <span
class="math inline">±</span> 368.17</td>
<td style="text-align: center;">0.06 <span class="math inline">±</span>
0.08</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyHopperVelocity-v1</span></td>
<td style="text-align: center;">1713.29 <span
class="math inline">±</span> 10.21</td>
<td style="text-align: center;">8.96 <span class="math inline">±</span>
4.28</td>
<td style="text-align: center;">1178.59 <span
class="math inline">±</span> 646.71</td>
<td style="text-align: center;">18.76 <span class="math inline">±</span>
8.93</td>
<td style="text-align: center;">1643.39 <span
class="math inline">±</span> 2.58</td>
<td style="text-align: center;">0.77 <span class="math inline">±</span>
0.26</td>
<td style="text-align: center;">1646.47 <span
class="math inline">±</span> 49.95</td>
<td style="text-align: center;">0.42 <span class="math inline">±</span>
0.84</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyHumanoidVelocity-v1</span></td>
<td style="text-align: center;">6579.26 <span
class="math inline">±</span> 55.70</td>
<td style="text-align: center;">3.76 <span class="math inline">±</span>
3.61</td>
<td style="text-align: center;">6407.95 <span
class="math inline">±</span> 254.06</td>
<td style="text-align: center;">7.38 <span class="math inline">±</span>
11.34</td>
<td style="text-align: center;">6321.45 <span
class="math inline">±</span> 35.73</td>
<td style="text-align: center;">0.00 <span class="math inline">±</span>
0.00</td>
<td style="text-align: center;">6332.14 <span
class="math inline">±</span> 89.86</td>
<td style="text-align: center;">0.00 <span class="math inline">±</span>
0.00</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetySwimmerVelocity-v1</span></td>
<td style="text-align: center;">91.05 <span class="math inline">±</span>
62.68</td>
<td style="text-align: center;">19.12 <span class="math inline">±</span>
8.33</td>
<td style="text-align: center;">69.75 <span class="math inline">±</span>
46.52</td>
<td style="text-align: center;">20.48 <span class="math inline">±</span>
9.13</td>
<td style="text-align: center;">33.02 <span class="math inline">±</span>
7.26</td>
<td style="text-align: center;">24.23 <span class="math inline">±</span>
0.54</td>
<td style="text-align: center;">39.24 <span class="math inline">±</span>
5.01</td>
<td style="text-align: center;">23.20 <span class="math inline">±</span>
0.48</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyWalker2dVelocity-v1</span></td>
<td style="text-align: center;">2183.43 <span
class="math inline">±</span> 1300.69</td>
<td style="text-align: center;">14.12 <span class="math inline">±</span>
10.28</td>
<td style="text-align: center;">2707.75 <span
class="math inline">±</span> 980.56</td>
<td style="text-align: center;">9.60 <span class="math inline">±</span>
8.94</td>
<td style="text-align: center;">2195.57 <span
class="math inline">±</span> 1046.29</td>
<td style="text-align: center;">7.63 <span class="math inline">±</span>
10.44</td>
<td style="text-align: center;">2079.64 <span
class="math inline">±</span> 1028.73</td>
<td style="text-align: center;">13.74 <span class="math inline">±</span>
15.94</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyCarGoal1-v0</span></td>
<td style="text-align: center;">10.60 <span class="math inline">±</span>
2.51</td>
<td style="text-align: center;">30.66 <span class="math inline">±</span>
7.53</td>
<td style="text-align: center;">25.49 <span class="math inline">±</span>
1.31</td>
<td style="text-align: center;">28.92 <span class="math inline">±</span>
7.66</td>
<td style="text-align: center;">17.92 <span class="math inline">±</span>
1.54</td>
<td style="text-align: center;">21.60 <span class="math inline">±</span>
0.83</td>
<td style="text-align: center;">22.09 <span class="math inline">±</span>
3.07</td>
<td style="text-align: center;">17.97 <span class="math inline">±</span>
1.35</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyCarButton1-v0</span></td>
<td style="text-align: center;">-1.36 <span class="math inline">±</span>
0.68</td>
<td style="text-align: center;">14.62 <span class="math inline">±</span>
9.40</td>
<td style="text-align: center;">-0.31 <span class="math inline">±</span>
0.49</td>
<td style="text-align: center;">15.24 <span class="math inline">±</span>
17.01</td>
<td style="text-align: center;">4.47 <span class="math inline">±</span>
1.12</td>
<td style="text-align: center;">25.00 <span class="math inline">±</span>
0.00</td>
<td style="text-align: center;">4.34 <span class="math inline">±</span>
0.72</td>
<td style="text-align: center;">25.00 <span class="math inline">±</span>
0.00</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyCarGoal2-v0</span></td>
<td style="text-align: center;">0.13 <span class="math inline">±</span>
1.11</td>
<td style="text-align: center;">23.50 <span class="math inline">±</span>
1.22</td>
<td style="text-align: center;">1.77 <span class="math inline">±</span>
1.20</td>
<td style="text-align: center;">17.43 <span class="math inline">±</span>
12.13</td>
<td style="text-align: center;">6.59 <span class="math inline">±</span>
0.58</td>
<td style="text-align: center;">25.00 <span class="math inline">±</span>
0.00</td>
<td style="text-align: center;">7.12 <span class="math inline">±</span>
4.06</td>
<td style="text-align: center;">23.37 <span class="math inline">±</span>
1.35</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyCarButton2-v0</span></td>
<td style="text-align: center;">-1.59 <span class="math inline">±</span>
0.70</td>
<td style="text-align: center;">39.97 <span class="math inline">±</span>
26.91</td>
<td style="text-align: center;">-2.95 <span class="math inline">±</span>
4.03</td>
<td style="text-align: center;">27.90 <span class="math inline">±</span>
6.37</td>
<td style="text-align: center;">4.86 <span class="math inline">±</span>
1.57</td>
<td style="text-align: center;">25.00 <span class="math inline">±</span>
0.00</td>
<td style="text-align: center;">5.07 <span class="math inline">±</span>
1.24</td>
<td style="text-align: center;">25.00 <span class="math inline">±</span>
0.00</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyPointGoal1-v0</span></td>
<td style="text-align: center;">8.43 <span class="math inline">±</span>
3.43</td>
<td style="text-align: center;">25.74 <span class="math inline">±</span>
7.83</td>
<td style="text-align: center;">19.24 <span class="math inline">±</span>
3.94</td>
<td style="text-align: center;">21.38 <span class="math inline">±</span>
6.96</td>
<td style="text-align: center;">16.03 <span class="math inline">±</span>
8.60</td>
<td style="text-align: center;">19.17 <span class="math inline">±</span>
9.42</td>
<td style="text-align: center;">16.31 <span class="math inline">±</span>
6.99</td>
<td style="text-align: center;">22.10 <span class="math inline">±</span>
6.13</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyPointButton1-v0</span></td>
<td style="text-align: center;">1.18 <span class="math inline">±</span>
1.02</td>
<td style="text-align: center;">29.42 <span class="math inline">±</span>
12.10</td>
<td style="text-align: center;">6.40 <span class="math inline">±</span>
1.43</td>
<td style="text-align: center;">27.90 <span class="math inline">±</span>
13.27</td>
<td style="text-align: center;">7.48 <span class="math inline">±</span>
8.47</td>
<td style="text-align: center;">24.27 <span class="math inline">±</span>
3.95</td>
<td style="text-align: center;">9.52 <span class="math inline">±</span>
7.86</td>
<td style="text-align: center;">25.00 <span class="math inline">±</span>
0.00</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyPointGoal2-v0</span></td>
<td style="text-align: center;">-0.56 <span class="math inline">±</span>
0.06</td>
<td style="text-align: center;">48.43 <span class="math inline">±</span>
40.55</td>
<td style="text-align: center;">1.67 <span class="math inline">±</span>
1.43</td>
<td style="text-align: center;">23.50 <span class="math inline">±</span>
11.17</td>
<td style="text-align: center;">6.09 <span class="math inline">±</span>
5.03</td>
<td style="text-align: center;">25.00 <span class="math inline">±</span>
0.00</td>
<td style="text-align: center;">8.62 <span class="math inline">±</span>
7.13</td>
<td style="text-align: center;">25.00 <span class="math inline">±</span>
0.00</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyPointButton2-v0</span></td>
<td style="text-align: center;">0.42 <span class="math inline">±</span>
0.63</td>
<td style="text-align: center;">28.87 <span class="math inline">±</span>
11.27</td>
<td style="text-align: center;">1.00 <span class="math inline">±</span>
1.00</td>
<td style="text-align: center;">30.00 <span class="math inline">±</span>
9.50</td>
<td style="text-align: center;">6.94 <span class="math inline">±</span>
4.47</td>
<td style="text-align: center;">25.00 <span class="math inline">±</span>
0.00</td>
<td style="text-align: center;">8.35 <span class="math inline">±</span>
10.44</td>
<td style="text-align: center;">25.00 <span class="math inline">±</span>
0.00</td>
</tr>
</tbody>
  <caption><p style="font-family: 'Times New Roman', Times, serif;"><b>Table 2:</b> The performance of OmniSafe on-policy algorithms, encompassing both reward and cost, was assessed within the Safety-Gymnasium environments. It is crucial to highlight that all on-policy algorithms underwent evaluation following 1e7 training steps.</p></caption>
</table>

#### First Order Algorithms

<details>
<summary><b><big>1e6 Steps Velocity Results</big></b></summary>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/first_order_ant_1e6.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyAntVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/first_order_halfcheetah_1e6.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyHalfCheetahVelocity-v1
        </div>
        </td>
    </tr>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/first_order_hopper_1e6.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyHopperVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/first_order_humanoid_1e6.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyHumanoidVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/first_order_walker2d_1e6.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyWalker2dVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/first_order_swimmer_1e6.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetySwimmerVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <caption><p style="font-family: 'Times New Roman', Times, serif;"><b>Figure 1.1:</b> Training curves in Safety-Gymnasium MuJoCo Velocity environments within 1e6 steps
    </table>
</details>

<details>
<summary><b><big>1e7 Steps Velocity Results</big></b></summary>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/first_order_ant_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyAntVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/first_order_halfcheetah_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyHalfCheetahVelocity-v1
        </div>
        </td>
    </tr>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/first_order_hopper_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyHopperVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/first_order_humanoid_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyHumanoidVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/first_order_walker2d_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyWalker2dVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/first_order_swimmer_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetySwimmerVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <caption><p style="font-family: 'Times New Roman', Times, serif;"><b>Figure 1.2:</b> Training curves in Safety-Gymnasium MuJoCo Velocity environments within 1e7 steps
    </table>
</details>

<details>
<summary><b><big>1e7 Steps Navigation Results</big></b></summary>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/first_order_carbutton1_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyCarButton1-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/first_order_carbutton2_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyCarButton2-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/first_order_cargoal1_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyCarGoal1-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/first_order_cargoal2_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyCarGoal2-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/first_order_pointbutton1_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyPointButton1-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/first_order_pointbutton2_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyPointButton2-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/first_order_pointgoal1_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyPointGoal1-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/first_order_pointgoal2_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyPointGoal2-v0
        </div>
        </td>
    </tr>
    </table>
    <caption><p style="font-family: 'Times New Roman', Times, serif;"><b>Figure 1.3:</b> Training curves in Safety-Gymnasium MuJoCo Navigation environments within 1e7 steps
    </table>
</details>

#### Second Order Algorithms

<details>
<summary><b><big>1e6 Steps Velocity Results</big></b></summary>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/second_order_ant_1e6.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyAntVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/second_order_halfcheetah_1e6.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyHalfCheetahVelocity-v1
        </div>
        </td>
    </tr>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/second_order_hopper_1e6.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyHopperVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/second_order_humanoid_1e6.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyHumanoidVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/second_order_walker2d_1e6.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyWalker2dVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/second_order_swimmer_1e6.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetySwimmerVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <caption><p style="font-family: 'Times New Roman', Times, serif;"><b>Figure 2.1:</b> Training curves of second order algorithms in Safety-Gymnasium MuJoCo Velocity environments within 1e6 steps
    </table>
</details>

<details>
<summary><b><big>1e7 Steps Velocity Results</big></b></summary>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/second_order_ant_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyAntVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/second_order_halfcheetah_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyHalfCheetahVelocity-v1
        </div>
        </td>
    </tr>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/second_order_hopper_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyHopperVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/second_order_humanoid_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyHumanoidVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/second_order_walker2d_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyWalker2dVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/second_order_swimmer_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetySwimmerVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <caption><p style="font-family: 'Times New Roman', Times, serif;"><b>Figure 2.2:</b>  Training curves of second order algorithms in Safety-Gymnasium MuJoCo Velocity environments within 1e7 steps
    </table>
</details>

<details>
<summary><b><big>1e7 Steps Navigation Results</big></b></summary>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/second_order_carbutton1_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyCarButton1-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/second_order_carbutton2_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyCarButton2-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/second_order_cargoal1_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyCarGoal1-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/second_order_cargoal2_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyCarGoal2-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/second_order_pointbutton1_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyPointButton1-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/second_order_pointbutton2_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyPointButton2-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/second_order_pointgoal1_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyPointGoal1-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/second_order_pointgoal2_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyPointGoal2-v0
        </div>
        </td>
    </tr>
    </table>
    <caption><p style="font-family: 'Times New Roman', Times, serif;"><b>Figure 2.3:</b> Training curves of second order algorithms in Safety-Gymnasium MuJoCo Navigation environments within 1e7 steps
    </table>
</details>

#### Saute Algorithms

<details>
<summary><b><big>1e6 Steps Velocity Results</big></b></summary>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/saute_ant_1e6.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyAntVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/saute_halfcheetah_1e6.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyHalfCheetahVelocity-v1
        </div>
        </td>
    </tr>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/saute_hopper_1e6.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyHopperVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/saute_humanoid_1e6.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyHumanoidVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/saute_walker2d_1e6.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyWalker2dVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/saute_swimmer_1e6.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetySwimmerVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <caption><p style="font-family: 'Times New Roman', Times, serif;"><b>Figure 3.1:</b> Training curves of Saute MDP algorithms in Safety-Gymnasium MuJoCo Velocity environments within 1e6 steps
    </table>
</details>

<details>
<summary><b><big>1e7 Steps Velocity Results</big></b></summary>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/saute_ant_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyAntVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/saute_halfcheetah_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyHalfCheetahVelocity-v1
        </div>
        </td>
    </tr>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/saute_hopper_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyHopperVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/saute_humanoid_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyHumanoidVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/saute_walker2d_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyWalker2dVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/saute_swimmer_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetySwimmerVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <caption><p style="font-family: 'Times New Roman', Times, serif;"><b>Figure 3.2:</b> Training curves of Saute MDP algorithms in Safety-Gymnasium MuJoCo Velocity environments within 1e7 steps
    </table>
</details>

<details>
<summary><b><big>1e7 Steps Navigation Results</big></b></summary>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/saute_carbutton1_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyCarButton1-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/saute_carbutton2_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyCarButton2-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/saute_carcircle1_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyCarCircle1-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/saute_carcircle2_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyCarCircle2-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/saute_cargoal1_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyCarGoal1-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/saute_cargoal2_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyCarGoal2-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/saute_pointbutton1_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyPointButton1-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/saute_pointbutton2_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyPointButton2-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/saute_pointcircle1_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyPointCircle1-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/saute_pointcircle2_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyPointCircle2-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/saute_pointgoal1_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyPointGoal1-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/saute_pointgoal2_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyPointGoal2-v0
        </div>
        </td>
    </tr>
    </table>
    <caption><p style="font-family: 'Times New Roman', Times, serif;"><b>Figure 3.3:</b> Training curves of Saute MDP algorithms in Safety-Gymnasium MuJoCo Navigation environments within 1e7 steps
    </table>
</details>

#### Simmer Algorithms

<details>
<summary><b><big>1e6 Steps Velocity Results</big></b></summary>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/simmer_ant_1e6.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyAntVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/simmer_halfcheetah_1e6.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyHalfCheetahVelocity-v1
        </div>
        </td>
    </tr>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/simmer_hopper_1e6.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyHopperVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/simmer_humanoid_1e6.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyHumanoidVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/simmer_walker2d_1e6.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyWalker2dVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/simmer_swimmer_1e6.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetySwimmerVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <caption><p style="font-family: 'Times New Roman', Times, serif;"><b>Figure 4.1:</b> Training curves of Simmer MDP algorithms in Safety-Gymnasium MuJoCo Velocity environments within 1e6 steps
    </table>
</details>

<details>
<summary><b><big>1e7 Steps Velocity Results</big></b></summary>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/simmer_ant_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyAntVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/simmer_halfcheetah_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyHalfCheetahVelocity-v1
        </div>
        </td>
    </tr>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/simmer_hopper_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyHopperVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/simmer_humanoid_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyHumanoidVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/simmer_walker2d_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyWalker2dVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/simmer_swimmer_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetySwimmerVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <caption><p style="font-family: 'Times New Roman', Times, serif;"><b>Figure 4.2:</b> Training curves of Simmer MDP algorithms in Safety-Gymnasium MuJoCo Velocity environments within 1e7 steps
    </table>
</details>

<details>
<summary><b><big>1e7 Steps Navigation Results</big></b></summary>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/simmer_carbutton1_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyCarButton1-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/simmer_carbutton2_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyCarButton2-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/simmer_cargoal1_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyCarGoal1-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/simmer_cargoal2_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyCarGoal2-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/simmer_pointbutton1_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyPointButton1-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/simmer_pointbutton2_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyPointButton2-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/simmer_pointgoal1_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyPointGoal1-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/simmer_pointgoal2_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyPointGoal2-v0
        </div>
        </td>
    </tr>
    </table>
    <caption><p style="font-family: 'Times New Roman', Times, serif;"><b>Figure 4.3:</b> Training curves of Simmer MDP algorithms in Safety-Gymnasium MuJoCo Navigation environments within 1e7 steps
    </table>
</details>

#### PID-Lagrangian Algorithms

<details>
<summary><b><big>1e6 Steps Velocity Results</big></b></summary>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/pid_ant_1e6.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyAntVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/pid_halfcheetah_1e6.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyHalfCheetahVelocity-v1
        </div>
        </td>
    </tr>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/pid_hopper_1e6.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyHopperVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/pid_humanoid_1e6.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyHumanoidVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/pid_walker2d_1e6.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyWalker2dVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/pid_swimmer_1e6.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetySwimmerVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <caption><p style="font-family: 'Times New Roman', Times, serif;"><b>Figure 5.1:</b> Training curves of PID-Lagrangian algorithms in Safety-Gymnasium MuJoCo Velocity environments within 1e6 steps
    </table>
</details>

<details>
<summary><b><big>1e7 Steps Velocity Results</big></b></summary>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/pid_ant_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyAntVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/pid_halfcheetah_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyHalfCheetahVelocity-v1
        </div>
        </td>
    </tr>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/pid_hopper_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyHopperVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/pid_humanoid_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyHumanoidVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/pid_walker2d_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyWalker2dVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/pid_swimmer_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetySwimmerVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <caption><p style="font-family: 'Times New Roman', Times, serif;"><b>Figure 5.2:</b> Training curves of PID-Lagrangian algorithms in Safety-Gymnasium MuJoCo Velocity environments within 1e7 steps
    </table>
</details>

<details>
<summary><b><big>1e7 Steps Navigation Results</big></b></summary>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/pid_carbutton1_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyCarButton1-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/pid_carbutton2_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyCarButton2-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/pid_cargoal1_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyCarGoal1-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/pid_cargoal2_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyCarGoal2-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/pid_pointbutton1_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyPointButton1-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/pid_pointbutton2_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyPointButton2-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/pid_pointgoal1_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyPointGoal1-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/pid_pointgoal2_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyPointGoal2-v0
        </div>
        </td>
    </tr>
    </table>
    <caption><p style="font-family: 'Times New Roman', Times, serif;"><b>Figure 5.3:</b> Training curves of PID-Lagrangian algorithms in Safety-Gymnasium MuJoCo Navigation environments within 1e7 steps.
    </table>
</details>

#### Early Terminated MDP Algorithms

<details>
<summary><b><big>1e6 Steps Velocity Results</big></b></summary>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/early_ant_1e6.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyAntVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/early_halfcheetah_1e6.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyHalfCheetahVelocity-v1
        </div>
        </td>
    </tr>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/early_hopper_1e6.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyHopperVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/early_humanoid_1e6.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyHumanoidVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/early_walker2d_1e6.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyWalker2dVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/early_swimmer_1e6.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetySwimmerVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <caption><p style="font-family: 'Times New Roman', Times, serif;"><b>Figure 6.1:</b> Training curves of early terminated MDP algorithms in Safety-Gymnasium MuJoCo Velocity environments within 1e6 steps.
    </table>
</details>

<details>
<summary><b><big>1e7 Steps Velocity Results</big></b></summary>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/early_ant_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyAntVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/early_halfcheetah_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyHalfCheetahVelocity-v1
        </div>
        </td>
    </tr>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/early_hopper_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyHopperVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/early_humanoid_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyHumanoidVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/early_walker2d_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyWalker2dVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/early_swimmer_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetySwimmerVelocity-v1
        </div>
        </td>
    </tr>
    </table>
    <caption><p style="font-family: 'Times New Roman', Times, serif;"><b>Figure 6.2:</b> Training curves of early terminated MDP algorithms in Safety-Gymnasium MuJoCo Velocity environments within 1e7 steps.
    </table>
</details>

<details>
<summary><b><big>1e7 Steps Navigation Results</big></b></summary>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/early_carbutton1_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyCarButton1-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/early_carbutton2_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyCarButton2-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/early_cargoal1_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyCarGoal1-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/early_cargoal2_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyCarGoal2-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/early_pointbutton1_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyPointButton1-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/early_pointbutton2_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyPointButton2-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/early_pointgoal1_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyPointGoal1-v0
        </div>
        </td>
    </tr>
    </table>
    <table>
    <tr>
        <td style="text-align:center">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/early_pointgoal2_1e7.png">
        <br>
        <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
            SafetyPointGoal2-v0
        </div>
        </td>
    </tr>
    </table>
    <caption><p style="font-family: 'Times New Roman', Times, serif;"><b>Figure 6.3:</b> Training curves of early terminated MDP algorithms in Safety-Gymnasium MuJoCo Navigation environments within 1e7 steps.
    </table>
</details>

## Experiment Analysis

### Hyperparameters

**We are continuously improving performance for on-policy algorithms and finding better hyperparameters. So we are happy to receive any advice from users, feel free for opening an [issue](https://github.com/PKU-Alignment/omnisafe/issues/new/choose) or [pull request](https://github.com/PKU-Alignment/omnisafe/pulls).**

#### First-Order Methods Specific Hyperparameters

- `algo_cfgs:kl_early_stop`: Whether to use the `early stop` trick for KL divergence. In first-order methods, this parameter is set to `True`. If the KL divergence is too large, we will stop the updating iteration.

#### Second-Order Methods Specific Hyperparameters

- `algo_cfgs:kl_early_stop`: Whether to use early stop for KL divergence. In second-order methods, we use line search to find the proper step size. If the KL divergence is too large, we will stop the line search and use the previous step size. So it is not necessary to use the `early stop` trick for KL divergence in second-order methods. We set `kl_early_stop=False` in second-order methods.

- `model_cfgs:actor:lr`: The learning rate of the policy network. Second-order methods update the policy by directly setting the parameters. So we just set the learning rate of the policy network to `None`.

You may find that in some environments, Natural PG performs nearly the same as TRPO. This is because, in the Mujoco
Velocity environment series, the TRPO search update step size is always 1. Additionally, since all algorithms were
tested under the same series of random seeds, there is an occurrence of TRPO and Natural PG training curves overlapping.

#### Saute RL Methods Specific Hyperparameters
- `saute_gamma`: In the experiment we found that `saute_gamma` impacts the performance of Saute RL methods a lot. We found that 0.999 is a good value for this hyperparameter.

#### Simmer RL Methods Specific Hyperparameters

- `saute_gamma`: Since the Simmer RL methods are based on Saute RL methods, we also set `saute_gamma` to 0.999.
- `control_cfgs`: The controller parameters of the Simmer RL methods. While Simmer uses a PID controller to control the safety budget, and PID is known as a parameter-sensitive controller. So we need to tune the control parameters (`Kp`, `Ki` and `Kd`) for different environments. We have done some experiments to find control parameters generally suitable for all environments, that is:

| Parameters | Descriptions| Values |
| -----------| ------------| ------ |
|`kp`|The proportional gain of the PID controller|1.0|
|`ki`|The derivative gain of the PID controller|0.001|
|`kd`|The integral gain of the PID controller|0.01|
|`polyak`|The learning rate for soft update|0.995|

#### PID-Lagrangian Methods Specific Hyperparameters

PID-Lagrangian methods use a PID controller to control the lagrangian multiplier, The `pid_kp`, `pid_kd` and `pid_ki` count for the proportional gain, derivative gain and integral gain of the PID controller respectively. As PID-Lagrangian methods use a PID controller to control the lagrangian multiplier, the hyperparameters of the PID controller are important for the performance of the algorithm.

- `pid_kp`: The proportional gain of the PID controller, determines how much the output responds to changes in the `ep_costs` signal. If the `pid_kp` is too large, the lagrangian multiplier will oscillate and the performance will be bad. If the `pid_kp` is too small, the lagrangian multiplier will update slowly and the performance will also be bad.
- `pid_kd`: The derivative gain of the PID controller, determines how much the output responds to changes in the `ep_costs` signal. If the `pid_kd` is too large, the lagrangian multiplier may be too sensitive to noise or changes in the `ep_costs` signal, leading to instability or oscillations. If the `pid_kd` is too small, the lagrangian multiplier may not respond quickly or accurately enough to changes in the `ep_costs`.
- `pid_ki`: The integral gain of the PID controller, determines the controller's ability to eliminate the steady-state error, by integrating the `ep_costs` signal over time. If the `pid_ki` is too large, the lagrangian multiplier may become too responsive to errors before.

We have done some experiments to find relatively good `pid_kp`, `pid_ki`, and `pid_kd` for all environments, and we found that the following value is a good value for this hyperparameter.

| Parameters | Descriptions| Values |
| -----------| ------------| ------ |
|`pid_kp`|The proportional gain of the PID controller|0.1|
|`pid_ki`|The derivative gain of the PID controller|0.01|
|`pid_kd`|The integral gain of the PID controller|0.01|

#### Early Terminated MDP Methods Specific Hyperparameters

- `vector_num_envs`: Though vectorized environments can speed up the training process, we found that the early terminated MDP will reset all the environments when one of the agents violates the safety constraint. So we set `vector_num_envs` to 1 in the early terminated MDP methods.

#### Lagragian

The lagrangian versions of on-policy algorithms share the same set of lagrangian hyperparameters (Except for PID-Lagrangian). The hyperparameters are listed below:

|      Hyperparameter      | Value |
| :----------------------: | :---: |
| `cost_limit` | 25.0  |
| `lagrangian_multiplier_init` | 0.001  |
| `lambda_lr` | 0.035  |
| `lambda_optimizer` | Adam |

### Some Hints

In our experiments, we found that some hyperparameters are important for the performance of the algorithm:

- `obs_normalize`: Whether to normalize the observation.
- `reward_normalize`: Whether to normalize the reward.
- `cost_normalize`: Whether to normalize the cost.

We have done some experiments to show the effect of these hyperparameters, and we log the best configuration for each algorithm in each environment. You can check it in the `omnisafe/configs/on_policy`.

In experiments, we found that the `obs_normalize=True` always performs better than `obs_normalize=False` in on-policy algorithms. That means the reward would increase quicker while the safety constraint also maintained if we normalize the observation. So we set `obs_normalize=True` in almost all on-policy algorithms.

Importantly, we found that the `reward_normalize=True` does not always perform better than `reward_normalize=False`, especially in the `SafetyHopperVelocity-v1` and `SafetyWalker2dVelocity` environments.

**To improve the overall performance stability, we use the following unified setting in all of OmniSafe on-policy algorithms**

|      Hyperparameter      | Value |
| :----------------------: | :---: |
| `obs_normalize`   | `True` |
| `reward_normalize`|  `False`  |
| `cost_normalize`  |  `False`  |

Besides, the hyperparameter `torch_num_threads` in `train_cfgs` is also important. In a single training session, a larger value for `torch_num_threads` often means faster training speed. However, we found in experiments that setting `torch_num_threads` too high can cause resource contention between parallel training sessions, resulting in slower overall experiment speed. In the configs file, we set the default value for `torch_num_threads` to 16, which ensures faster training speed for a single session. If you need to launch multiple training sessions in parallel, please consider your device configuration. For example, suppose your CPU has 2 physical threads per core and has a total of 32 cores, if you are launching 16 training scripts in parallel, you had better not set `torch_num_threads` to a value greater than 4.

If you find that other hyperparameters perform better, please feel free to open an [issue](https://github.com/PKU-Alignment/omnisafe/issues/new/choose) or [pull request](https://github.com/PKU-Alignment/omnisafe/pulls).
