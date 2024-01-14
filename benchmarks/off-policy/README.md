# OmniSafe's Safety-Gymnasium Benchmark for Off-Policy Algorithms

The OmniSafe Safety-Gymnasium Benchmark for off-policy algorithms evaluates the effectiveness of OmniSafe's off-policy algorithms across multiple environments from the [Safety-Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium) task suite. For each supported algorithm and environment, we offer the following:

- Default hyperparameters used for the benchmark and scripts that enable result replication.
- Performance comparison with other open-source implementations.
- Graphs and raw data that can be utilized for research purposes.
- Detailed logs obtained during training.
- Suggestions and hints on fine-tuning the algorithm for achieving optimal results.

Supported algorithms are listed below:

- **[ICLR 2016]** [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/pdf/1509.02971.pdf)
- **[ICML 2018]** [Twin Delayed DDPG (TD3)](https://arxiv.org/pdf/1802.09477.pdf)
- **[ICML 2018]** [Soft Actor-Critic (SAC)](https://arxiv.org/pdf/1812.05905.pdf)
- **[Preprint 2019]<sup>[[1]](#footnote1)</sup>** [The Lagrangian version of DDPG (DDPGLag)](https://cdn.openai.com/safexp-short.pdf)
- **[Preprint 2019]<sup>[[1]](#footnote1)</sup>** [The Lagrangian version of TD3 (TD3Lag)](https://cdn.openai.com/safexp-short.pdf)
- **[Preprint 2019]<sup>[[1]](#footnote1)</sup>** [The Lagrangian version of SAC (SACLag)](https://cdn.openai.com/safexp-short.pdf)
- **[ICML 2020]** [Responsive Safety in Reinforcement Learning by PID Lagrangian Methods (DDPGPID)](https://arxiv.org/abs/2007.03964)
- **[ICML 2020]** [Responsive Safety in Reinforcement Learning by PID Lagrangian Methods (TD3PID)](https://arxiv.org/abs/2007.03964)
- **[ICML 2020]** [Responsive Safety in Reinforcement Learning by PID Lagrangian Methods (SACPID)](https://arxiv.org/abs/2007.03964)

## Safety-Gymnasium

We highly recommend using **Safety-Gymnasium** to run the following experiments. To install, in a linux machine, type:

```bash
pip install safety_gymnasium
```

## Run the Benchmark
You can set the main function of `examples/benchmarks/experiment_grid.py` as:

```python
if __name__ == '__main__':
    eg = ExperimentGrid(exp_name='Off-Policy-Benchmarks')

    # set up the algorithms.
    off_policy = ['DDPG', 'SAC', 'TD3', 'DDPGLag', 'TD3Lag', 'SACLag', 'DDPGPID', 'TD3PID', 'SACPID']
    eg.add('algo', off_policy)

    # you can use wandb to monitor the experiment.
    eg.add('logger_cfgs:use_wandb', [False])
    # you can use tensorboard to monitor the experiment.
    eg.add('logger_cfgs:use_tensorboard', [True])

    # the default configs here are as follows:
    # eg.add('algo_cfgs:steps_per_epoch', [2000])
    # eg.add('train_cfgs:total_steps', [2000 * 500])
    # which can reproduce results of 1e6 steps.

    # if you want to reproduce results of 3e6 steps, using
    # eg.add('algo_cfgs:steps_per_epoch', [2000])
    # eg.add('train_cfgs:total_steps', [2000 * 1500])

    # set the device.
    avaliable_gpus = list(range(torch.cuda.device_count()))
    gpu_id = [0, 1, 2, 3]
    # if you want to use CPU, please set gpu_id = None
    # gpu_id = None

    if gpu_id and not set(gpu_id).issubset(avaliable_gpus):
        warnings.warn('The GPU ID is not available, use CPU instead.', stacklevel=1)
        gpu_id = None

    # set up the environments.
    eg.add('env_id', [
        'SafetyHopperVelocity-v1',
        'SafetyWalker2dVelocity-v1',
        'SafetySwimmerVelocity-v1',
        'SafetyAntVelocity-v1',
        'SafetyHalfCheetahVelocity-v1',
        'SafetyHumanoidVelocity-v1'
        ])
    eg.add('seed', [0, 5, 10, 15, 20])
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

Logs are saved in `examples/benchmarks/exp-x` and can be monitored with tensorboard or wandb.

```bash
tensorboard --logdir examples/benchmarks/exp-x
```

After the experiment is finished, you can use the following command to generate the video of the trained agent:

```bash
cd examples
python evaluate_saved_policy.py
```
Please note that before you evaluate, please set the `LOG_DIR` in `evaluate_saved_policy.py`.

For example, if I train `DDPG` in `SafetyHumanoidVelocity-v1`

```python
LOG_DIR = '~/omnisafe/examples/runs/DDPG-<SafetyHumanoidVelocity-v1>/seed-000'
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

In an effort to ascertain the credibility of OmniSafe’s algorithmic implementation, a comparative assessment was conducted, juxtaposing the performance of classical reinforcement
learning algorithms, such as DDPG, TD3 and SAC. The performance table is provided in <a href="#compare_off_policy">Table 1</a>, with
well-established open-source implementations, specifically [Tianshou](https://github.com/thu-ml/tianshou) and
[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).

<table id="compare_off_policy">
<thead>
<tr class="header">
<th style="text-align: left;"></th>
<th colspan="3" style="text-align: center;"><strong>DDPG</strong></th>
<th colspan="3" style="text-align: center;"><strong>TD3</strong></th>
<th colspan="3" style="text-align: center;"><strong>SAC</strong></th>
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
<td style="text-align: center;"><strong>OmniSafe (Ours)</strong></td>
<td style="text-align: center;"><strong>Tianshou</strong></td>
<td style="text-align: center;"><strong>Stable-Baselines3</strong></td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyAntVelocity-v1</span></td>
<td style="text-align: center;">860.86 <span
class="math inline">±</span> 198.03</td>
<td style="text-align: center;">308.60 <span
class="math inline">±</span> 318.60</td>
<td style="text-align: center;"><strong>2654.58 <span
class="math inline">±</span> 1738.21</strong></td>
<td style="text-align: center;">5246.86 <span
class="math inline">±</span> 580.50</td>
<td style="text-align: center;"><strong>5379.55 <span
class="math inline">±</span> 224.69</strong></td>
<td style="text-align: center;">3079.45 <span
class="math inline">±</span> 1456.81</td>
<td style="text-align: center;">5456.31 <span
class="math inline">±</span> 156.04</td>
<td style="text-align: center;"><strong>6012.30 <span
class="math inline">±</span> 102.64</strong></td>
<td style="text-align: center;">2404.50 <span
class="math inline">±</span> 1152.65</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyHalfCheetahVelocity-v1</span></td>
<td style="text-align: center;">11377.10 <span
class="math inline">±</span> 75.29</td>
<td style="text-align: center;"><strong>12493.55 <span
class="math inline">±</span> 437.54</strong></td>
<td style="text-align: center;">7796.63 <span
class="math inline">±</span> 3541.64</td>
<td style="text-align: center;"><strong>11246.12 <span
class="math inline">±</span> 488.62</strong></td>
<td style="text-align: center;">10246.77 <span
class="math inline">±</span> 908.39</td>
<td style="text-align: center;">8631.27 <span
class="math inline">±</span> 2869.15</td>
<td style="text-align: center;">11488.86 <span
class="math inline">±</span> 513.09</td>
<td style="text-align: center;"><strong>12083.89 <span
class="math inline">±</span> 564.51</strong></td>
<td style="text-align: center;">7767.74 <span
class="math inline">±</span> 3159.07</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyHopperVelocity-v1</span></td>
<td style="text-align: center;">1462.56 <span
class="math inline">±</span> 591.14</td>
<td style="text-align: center;">2018.97 <span
class="math inline">±</span> 1045.20</td>
<td style="text-align: center;"><strong>2214.06 <span
class="math inline">±</span> 1219.57</strong></td>
<td style="text-align: center;"><strong>3404.41 <span
class="math inline">±</span> 82.57</strong></td>
<td style="text-align: center;">2682.53 <span
class="math inline">±</span> 1004.84</td>
<td style="text-align: center;">2542.67 <span
class="math inline">±</span> 1253.33</td>
<td style="text-align: center;"><strong>3597.70 <span
class="math inline">±</span> 32.23</strong></td>
<td style="text-align: center;">3546.59 <span
class="math inline">±</span> 76 .00</td>
<td style="text-align: center;">2158.54 <span
class="math inline">±</span> 1343.24</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyHumanoidVelocity-v1</span></td>
<td style="text-align: center;">1537.39 <span
class="math inline">±</span> 335.62</td>
<td style="text-align: center;">124.96 <span
class="math inline">±</span> 61.68</td>
<td style="text-align: center;"><strong>2276.92 <span
class="math inline">±</span> 2299.68</strong></td>
<td style="text-align: center;"><strong>5798.01 <span
class="math inline">±</span> 160.7</strong>2</td>
<td style="text-align: center;">3838.06 <span
class="math inline">±</span> 1832.90</td>
<td style="text-align: center;">3511.06 <span
class="math inline">±</span> 2214.12</td>
<td style="text-align: center;"><strong>6039.77 <span
class="math inline">±</span> 167.82</strong></td>
<td style="text-align: center;">5424.55 <span
class="math inline">±</span> 118.52</td>
<td style="text-align: center;">2713.60 <span
class="math inline">±</span> 2256.89</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetySwimmerVelocity-v1</span></td>
<td style="text-align: center;">139.39 <span
class="math inline">±</span> 11.74</td>
<td style="text-align: center;">138.98 <span
class="math inline">±</span> 8.60</td>
<td style="text-align: center;"><strong>210.40 <span
class="math inline">±</span> 148.01</strong></td>
<td style="text-align: center;"><strong>98.39 <span
class="math inline">±</span> 32.28</strong></td>
<td style="text-align: center;">94.43 <span class="math inline">±</span>
9.63</td>
<td style="text-align: center;">247.09 <span
class="math inline">±</span> 131.69</td>
<td style="text-align: center;">46.44 <span class="math inline">±</span>
1.23</td>
<td style="text-align: center;">44.34 <span class="math inline">±</span>
2.01</td>
<td style="text-align: center;"><strong>247.33 <span
class="math inline">±</span> 122.02</strong></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyWalker2dVelocity-v1</span></td>
<td style="text-align: center;">1911.70 <span
class="math inline">±</span> 395.97</td>
<td style="text-align: center;">543.23 <span
class="math inline">±</span> 316.10</td>
<td style="text-align: center;"><strong>3917.46 <span
class="math inline">±</span> 1077.38</strong></td>
<td style="text-align: center;">3034.83 <span
class="math inline">±</span> 1374.72</td>
<td style="text-align: center;"><strong>4267.05 <span
class="math inline">±</span> 678.65</strong></td>
<td style="text-align: center;">4087.94 <span
class="math inline">±</span> 755.10</td>
<td style="text-align: center;">4419.29 <span
class="math inline">±</span> 232.06</td>
<td style="text-align: center;"><strong>4619.34 <span
class="math inline">±</span> 274.43</strong></td>
<td style="text-align: center;">3906.78 <span
class="math inline">±</span> 795.48</td>
</tr>
</tbody>
  <caption><p style="font-family: 'Times New Roman', Times, serif;"><b>Table 1:</b> The performance of OmniSafe, which was evaluated in relation to published baselines within the Safety-Gymnasium environments. Experimental outcomes, comprising mean and standard deviation, were derived from 10 assessment iterations encompassing multiple random seeds. A noteworthy distinction lies in the fact that Stable-Baselines3 employs distinct parameters tailored to each environment, while OmniSafe maintains a consistent parameter set across all environments.</p></caption>
</table>

### Safe Reinforcement Learning Algorithms

To demonstrate the high reliability of the algorithms implemented, OmniSafe offers performance insights within the Safety-Gymnasium environment. It should be noted that all data is procured under the constraint of `cost_limit=25.00`. The results are presented in <a href="#performance_off_policy">Table 2</a>, <a href="#curve_off_base">Figure 1</a>, <a href="#curve_off_lag">Figure 2</a>, <a href="#curve_off_pid">Figure 3</a>.

#### Performance Table

<table id="performance_off_policy">
<thead>
<tr class="header">
<th style="text-align: left;"></th>
<th colspan="2" style="text-align: center;"><strong>DDPG</strong></th>
<th colspan="2" style="text-align: center;"><strong>TD3</strong></th>
<th colspan="2" style="text-align: center;"><strong>SAC</strong></th>
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
class="smallcaps">SafetyAntVelocity-v1</span></td>
<td style="text-align: center;">860.86 <span
class="math inline">±</span> 198.03</td>
<td style="text-align: center;">234.80 <span
class="math inline">±</span> 40.63</td>
<td style="text-align: center;">5246.86 <span
class="math inline">±</span> 580.50</td>
<td style="text-align: center;">912.90 <span
class="math inline">±</span> 93.73</td>
<td style="text-align: center;">5456.31 <span
class="math inline">±</span> 156.04</td>
<td style="text-align: center;">943.10 <span
class="math inline">±</span> 47.51</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyHalfCheetahVelocity-v1</span></td>
<td style="text-align: center;">11377.10 <span
class="math inline">±</span> 75.29</td>
<td style="text-align: center;">980.93 <span
class="math inline">±</span> 1.05</td>
<td style="text-align: center;">11246.12 <span
class="math inline">±</span> 488.62</td>
<td style="text-align: center;">981.27 <span
class="math inline">±</span> 0.31</td>
<td style="text-align: center;">11488.86 <span
class="math inline">±</span> 513.09</td>
<td style="text-align: center;">981.93 <span
class="math inline">±</span> 0.33</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyHopperVelocity-v1</span></td>
<td style="text-align: center;">1462.56 <span
class="math inline">±</span> 591.14</td>
<td style="text-align: center;">429.17 <span
class="math inline">±</span> 220.05</td>
<td style="text-align: center;">3404.41 <span
class="math inline">±</span> 82.57</td>
<td style="text-align: center;">973.80 <span
class="math inline">±</span> 4.92</td>
<td style="text-align: center;">3537.70 <span
class="math inline">±</span> 32.23</td>
<td style="text-align: center;">975.23 <span
class="math inline">±</span> 2.39</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyHumanoidVelocity-v1</span></td>
<td style="text-align: center;">1537.39 <span
class="math inline">±</span> 335.62</td>
<td style="text-align: center;">48.79 <span class="math inline">±</span>
13.06</td>
<td style="text-align: center;">5798.01 <span
class="math inline">±</span> 160.72</td>
<td style="text-align: center;">255.43 <span
class="math inline">±</span> 437.13</td>
<td style="text-align: center;">6039.77 <span
class="math inline">±</span> 167.82</td>
<td style="text-align: center;">41.42 <span class="math inline">±</span>
49.78</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetySwimmerVelocity-v1</span></td>
<td style="text-align: center;">139.39 <span
class="math inline">±</span> 11.74</td>
<td style="text-align: center;">200.53 <span
class="math inline">±</span> 43.28</td>
<td style="text-align: center;">98.39 <span class="math inline">±</span>
32.28</td>
<td style="text-align: center;">115.27 <span
class="math inline">±</span> 44.90</td>
<td style="text-align: center;">46.44 <span class="math inline">±</span>
1.23</td>
<td style="text-align: center;">40.97 <span class="math inline">±</span>
0.47</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyWalker2dVelocity-v1</span></td>
<td style="text-align: center;">1911.70 <span
class="math inline">±</span> 395.97</td>
<td style="text-align: center;">318.10 <span
class="math inline">±</span> 71.03</td>
<td style="text-align: center;">3034.83 <span
class="math inline">±</span> 1374.72</td>
<td style="text-align: center;">606.47 <span
class="math inline">±</span> 337.33</td>
<td style="text-align: center;">4419.29 <span
class="math inline">±</span> 232.06</td>
<td style="text-align: center;">877.70 <span
class="math inline">±</span> 8.95</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyCarCircle1-v0</span></td>
<td style="text-align: center;">44.64 <span class="math inline">±</span>
2.15</td>
<td style="text-align: center;">371.93 <span
class="math inline">±</span> 38.75</td>
<td style="text-align: center;">44.57 <span class="math inline">±</span>
2.71</td>
<td style="text-align: center;">383.37 <span
class="math inline">±</span> 62.03</td>
<td style="text-align: center;">43.46 <span class="math inline">±</span>
4.39</td>
<td style="text-align: center;">406.87 <span
class="math inline">±</span> 78.78</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyCarGoal1-v0</span></td>
<td style="text-align: center;">36.99 <span class="math inline">±</span>
1.66</td>
<td style="text-align: center;">57.13 <span class="math inline">±</span>
38.40</td>
<td style="text-align: center;">36.26 <span class="math inline">±</span>
2.35</td>
<td style="text-align: center;">69.70 <span class="math inline">±</span>
52.18</td>
<td style="text-align: center;">35.71 <span class="math inline">±</span>
2.24</td>
<td style="text-align: center;">54.73 <span class="math inline">±</span>
46.74</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyPointCircle1-v0</span></td>
<td style="text-align: center;">113.67 <span
class="math inline">±</span> 1.33</td>
<td style="text-align: center;">421.53 <span
class="math inline">±</span> 142.66</td>
<td style="text-align: center;">115.15 <span
class="math inline">±</span> 2.24</td>
<td style="text-align: center;">391.07 <span
class="math inline">±</span> 38.34</td>
<td style="text-align: center;">115.06 <span
class="math inline">±</span> 2.04</td>
<td style="text-align: center;">403.43 <span
class="math inline">±</span> 44.78</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyPointGoal1-v0</span></td>
<td style="text-align: center;">25.55 <span class="math inline">±</span>
2.62</td>
<td style="text-align: center;">41.60 <span class="math inline">±</span>
37.17</td>
<td style="text-align: center;">27.28 <span class="math inline">±</span>
1.21</td>
<td style="text-align: center;">51.43 <span class="math inline">±</span>
33.05</td>
<td style="text-align: center;">27.04 <span class="math inline">±</span>
1.49</td>
<td style="text-align: center;">67.57 <span class="math inline">±</span>
32.13</td>
</tr>
</tbody>
<thead>
<tr class="header">
<th style="text-align: left;"></th>
<th colspan="2"
style="text-align: center;"><strong>DDPGLag</strong></th>
<th colspan="2" style="text-align: center;"><strong>TD3Lag</strong></th>
<th colspan="2" style="text-align: center;"><strong>SACLag</strong></th>
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
class="smallcaps">SafetyAntVelocity-v1</span></td>
<td style="text-align: center;">1271.48 <span
class="math inline">±</span> 581.71</td>
<td style="text-align: center;">33.27 <span class="math inline">±</span>
13.34</td>
<td style="text-align: center;">1944.38 <span
class="math inline">±</span> 759.20</td>
<td style="text-align: center;">63.27 <span class="math inline">±</span>
46.89</td>
<td style="text-align: center;">1897.32 <span
class="math inline">±</span> 1213.74</td>
<td style="text-align: center;">5.73 <span class="math inline">±</span>
7.83</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyHalfCheetahVelocity-v1</span></td>
<td style="text-align: center;">2743.06 <span
class="math inline">±</span> 21.77</td>
<td style="text-align: center;">0.33 <span class="math inline">±</span>
0.12</td>
<td style="text-align: center;">2741.08 <span
class="math inline">±</span> 49.13</td>
<td style="text-align: center;">10.47 <span class="math inline">±</span>
14.45</td>
<td style="text-align: center;">2833.72 <span
class="math inline">±</span> 3.62</td>
<td style="text-align: center;">0.00 <span class="math inline">±</span>
0.00</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyHopperVelocity-v1</span></td>
<td style="text-align: center;">1093.25 <span
class="math inline">±</span> 81.55</td>
<td style="text-align: center;">15.00 <span class="math inline">±</span>
21.21</td>
<td style="text-align: center;">928.79 <span
class="math inline">±</span> 389.48</td>
<td style="text-align: center;">40.67 <span class="math inline">±</span>
30.99</td>
<td style="text-align: center;">963.49 <span
class="math inline">±</span> 291.64</td>
<td style="text-align: center;">20.23 <span class="math inline">±</span>
28.47</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyHumanoidVelocity-v1</span></td>
<td style="text-align: center;">2059.96 <span
class="math inline">±</span> 485.68</td>
<td style="text-align: center;">19.71 <span class="math inline">±</span>
4.05</td>
<td style="text-align: center;">5751.99 <span
class="math inline">±</span> 157.28</td>
<td style="text-align: center;">10.71 <span class="math inline">±</span>
23.60</td>
<td style="text-align: center;">5940.04 <span
class="math inline">±</span> 121.93</td>
<td style="text-align: center;">17.59 <span class="math inline">±</span>
6.24</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetySwimmerVelocity-v1</span></td>
<td style="text-align: center;">13.18 <span class="math inline">±</span>
20.31</td>
<td style="text-align: center;">28.27 <span class="math inline">±</span>
32.27</td>
<td style="text-align: center;">15.58 <span class="math inline">±</span>
16.97</td>
<td style="text-align: center;">13.27 <span class="math inline">±</span>
17.64</td>
<td style="text-align: center;">11.03 <span class="math inline">±</span>
11.17</td>
<td style="text-align: center;">22.70 <span class="math inline">±</span>
32.10</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyWalker2dVelocity-v1</span></td>
<td style="text-align: center;">2238.92 <span
class="math inline">±</span> 400.67</td>
<td style="text-align: center;">33.43 <span class="math inline">±</span>
20.08</td>
<td style="text-align: center;">2996.21 <span
class="math inline">±</span> 74.40</td>
<td style="text-align: center;">22.50 <span class="math inline">±</span>
16.97</td>
<td style="text-align: center;">2676.47 <span
class="math inline">±</span> 300.43</td>
<td style="text-align: center;">30.67 <span class="math inline">±</span>
32.30</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyCarCircle1-v0</span></td>
<td style="text-align: center;">33.29 <span class="math inline">±</span>
6.55</td>
<td style="text-align: center;">20.67 <span class="math inline">±</span>
28.48</td>
<td style="text-align: center;">34.38 <span class="math inline">±</span>
1.55</td>
<td style="text-align: center;">2.25 <span class="math inline">±</span>
3.90</td>
<td style="text-align: center;">31.42 <span class="math inline">±</span>
11.67</td>
<td style="text-align: center;">22.33 <span class="math inline">±</span>
26.16</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyCarGoal1-v0</span></td>
<td style="text-align: center;">22.80 <span class="math inline">±</span>
8.75</td>
<td style="text-align: center;">17.33 <span class="math inline">±</span>
21.40</td>
<td style="text-align: center;">7.31 <span class="math inline">±</span>
5.34</td>
<td style="text-align: center;">33.83 <span class="math inline">±</span>
31.03</td>
<td style="text-align: center;">10.83 <span class="math inline">±</span>
11.29</td>
<td style="text-align: center;">22.67 <span class="math inline">±</span>
28.91</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyPointCircle1-v0</span></td>
<td style="text-align: center;">70.71 <span class="math inline">±</span>
13.61</td>
<td style="text-align: center;">22.00 <span class="math inline">±</span>
32.80</td>
<td style="text-align: center;">83.07 <span class="math inline">±</span>
3.49</td>
<td style="text-align: center;">7.83 <span class="math inline">±</span>
15.79</td>
<td style="text-align: center;">83.68 <span class="math inline">±</span>
3.32</td>
<td style="text-align: center;">12.83 <span class="math inline">±</span>
19.53</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyPointGoal1-v0</span></td>
<td style="text-align: center;">17.17 <span class="math inline">±</span>
10.03</td>
<td style="text-align: center;">20.33 <span class="math inline">±</span>
31.59</td>
<td style="text-align: center;">25.27 <span class="math inline">±</span>
2.74</td>
<td style="text-align: center;">28.00 <span class="math inline">±</span>
15.75</td>
<td style="text-align: center;">21.45 <span class="math inline">±</span>
6.97</td>
<td style="text-align: center;">19.17 <span class="math inline">±</span>
9.72</td>
</tr>
</tbody>
<thead>
<tr class="header">
<th style="text-align: left;"></th>
<th colspan="2"
style="text-align: center;"><strong>DDPGPID</strong></th>
<th colspan="2" style="text-align: center;"><strong>TD3PID</strong></th>
<th colspan="2" style="text-align: center;"><strong>SACPID</strong></th>
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
class="smallcaps">SafetyAntVelocity-v1</span></td>
<td style="text-align: center;">2078.27 <span
class="math inline">±</span> 704.77</td>
<td style="text-align: center;">18.20 <span class="math inline">±</span>
7.21</td>
<td style="text-align: center;">2410.46 <span
class="math inline">±</span> 217.00</td>
<td style="text-align: center;">44.50 <span class="math inline">±</span>
38.39</td>
<td style="text-align: center;">1940.55 <span
class="math inline">±</span> 482.41</td>
<td style="text-align: center;">13.73 <span class="math inline">±</span>
7.24</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyHalfCheetahVelocity-v1</span></td>
<td style="text-align: center;">2737.61 <span
class="math inline">±</span> 45.93</td>
<td style="text-align: center;">36.10 <span class="math inline">±</span>
11.03</td>
<td style="text-align: center;">2695.64 <span
class="math inline">±</span> 29.42</td>
<td style="text-align: center;">35.93 <span class="math inline">±</span>
14.03</td>
<td style="text-align: center;">2689.01 <span
class="math inline">±</span> 15.46</td>
<td style="text-align: center;">21.43 <span class="math inline">±</span>
5.49</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyHopperVelocity-v1</span></td>
<td style="text-align: center;">1034.42 <span
class="math inline">±</span> 350.59</td>
<td style="text-align: center;">29.53 <span class="math inline">±</span>
34.54</td>
<td style="text-align: center;">1225.97 <span
class="math inline">±</span> 224.71</td>
<td style="text-align: center;">46.87 <span class="math inline">±</span>
65.28</td>
<td style="text-align: center;">812.80 <span
class="math inline">±</span> 381.86</td>
<td style="text-align: center;">92.23 <span class="math inline">±</span>
77.64</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyHumanoidVelocity-v1</span></td>
<td style="text-align: center;">1082.36 <span
class="math inline">±</span> 486.48</td>
<td style="text-align: center;">15.00 <span class="math inline">±</span>
19.51</td>
<td style="text-align: center;">6179.38 <span
class="math inline">±</span> 105.70</td>
<td style="text-align: center;">5.60 <span class="math inline">±</span>
6.23</td>
<td style="text-align: center;">6107.36 <span
class="math inline">±</span> 113.24</td>
<td style="text-align: center;">6.20 <span class="math inline">±</span>
10.14</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetySwimmerVelocity-v1</span></td>
<td style="text-align: center;">23.99 <span class="math inline">±</span>
7.76</td>
<td style="text-align: center;">30.70 <span class="math inline">±</span>
21.81</td>
<td style="text-align: center;">28.62 <span class="math inline">±</span>
8.48</td>
<td style="text-align: center;">22.47 <span class="math inline">±</span>
7.69</td>
<td style="text-align: center;">7.50 <span class="math inline">±</span>
10.42</td>
<td style="text-align: center;">7.77 <span class="math inline">±</span>
8.48</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyWalker2dVelocity-v1</span></td>
<td style="text-align: center;">1378.75 <span
class="math inline">±</span> 896.73</td>
<td style="text-align: center;">14.77 <span class="math inline">±</span>
13.02</td>
<td style="text-align: center;">2769.64 <span
class="math inline">±</span> 67.23</td>
<td style="text-align: center;">6.53 <span class="math inline">±</span>
8.86</td>
<td style="text-align: center;">1251.87 <span
class="math inline">±</span> 721.54</td>
<td style="text-align: center;">41.23 <span class="math inline">±</span>
73.33</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyCarCircle1-v0</span></td>
<td style="text-align: center;">26.89 <span class="math inline">±</span>
11.18</td>
<td style="text-align: center;">31.83 <span class="math inline">±</span>
33.59</td>
<td style="text-align: center;">34.77 <span class="math inline">±</span>
3.24</td>
<td style="text-align: center;">47.00 <span class="math inline">±</span>
39.53</td>
<td style="text-align: center;">34.41 <span class="math inline">±</span>
7.19</td>
<td style="text-align: center;">5.00 <span class="math inline">±</span>
11.18</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyCarGoal1-v0</span></td>
<td style="text-align: center;">19.35 <span class="math inline">±</span>
14.63</td>
<td style="text-align: center;">17.50 <span class="math inline">±</span>
21.31</td>
<td style="text-align: center;">27.28 <span class="math inline">±</span>
4.50</td>
<td style="text-align: center;">9.50 <span class="math inline">±</span>
12.15</td>
<td style="text-align: center;">16.21 <span class="math inline">±</span>
12.65</td>
<td style="text-align: center;">6.67 <span class="math inline">±</span>
14.91</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetyPointCircle1-v0</span></td>
<td style="text-align: center;">71.63 <span class="math inline">±</span>
8.39</td>
<td style="text-align: center;">0.00 <span class="math inline">±</span>
0.00</td>
<td style="text-align: center;">70.95 <span class="math inline">±</span>
6.00</td>
<td style="text-align: center;">0.00 <span class="math inline">±</span>
0.00</td>
<td style="text-align: center;">75.15 <span class="math inline">±</span>
6.65</td>
<td style="text-align: center;">4.50 <span class="math inline">±</span>
4.65</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span
class="smallcaps">SafetyPointGoal1-v0</span></td>
<td style="text-align: center;">19.85 <span class="math inline">±</span>
5.32</td>
<td style="text-align: center;">22.67 <span class="math inline">±</span>
13.73</td>
<td style="text-align: center;">18.76 <span class="math inline">±</span>
7.87</td>
<td style="text-align: center;">12.17 <span class="math inline">±</span>
9.39</td>
<td style="text-align: center;">15.87 <span class="math inline">±</span>
6.73</td>
<td style="text-align: center;">27.50 <span class="math inline">±</span>
15.25</td>
</tr>
</tbody>
  <caption><p style="font-family: 'Times New Roman', Times, serif;"><b>Table 2:</b> The performance of OmniSafe off-policy algorithms, which underwent evaluation under the experimental setting of cost_limit=25.00. During experimentation, it was observed that off-policy algorithms did not violate safety constraints in SafetyHumanoidVeloicty-v1. This observation suggests that the agent may not have fully learned to run within 1e6 steps; consequently, the 3e6 results were utilized in off-policy SafetyHumanoidVeloicty-v1. Meanwhile, in environments with strong stochasticity such as SafetyCarCircle1-v0, SafetyCarGoal1-v0, SafetyPointCircle1-v0, and SafetyPointGoal1-v0, off-policy methods require more training steps to estimate a more accurate Q-function. Therefore, we also conducted evaluations on these four environments using a training duration of 3e6 steps. For other environments, we use the evaluation results after 1e6 training steps.</p></caption>
</table>


#### Performance Curves

<table id="curve_off_base">
  <tr>
    <td style="text-align:center">
      <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/off-policy/benchmarks/base_ant_1e6.png">
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
        SafetyAntVelocity-v1
      </div>
    </td>
  </tr>
  <tr>
    <td style="text-align:center">
      <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/off-policy/benchmarks/base_halfcheetah_1e6.png">
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
        SafetyHalfCheetahVelocity-v1
      </div>
    </td>
  </tr>
  <tr>
    <td style="text-align:center">
      <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/off-policy/benchmarks/base_hopper_1e6.png">
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
        SafetyHopperVelocity-v1
      </div>
    </td>
  </tr>
  <tr>
    <td style="text-align:center">
      <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/off-policy/benchmarks/base_humanoid_3e6.png">
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
        SafetyHumanoidVelocity-v1
      </div>
    </td>
  </tr>
  <tr>
    <td style="text-align:center">
      <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/off-policy/benchmarks/base_swimmer_1e6.png">
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
        SafetySwimmerVelocity-v1
      </div>
    </td>
  </tr>
  <tr>
    <td style="text-align:center">
      <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/off-policy/benchmarks/base_walker2d_1e6.png">
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
        SafetyWalker2dVelocity-v1
      </div>
    </td>
  </tr>
  <tr>
    <td style="text-align:center">
      <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/off-policy/benchmarks/base_carcircle1_3e6.png">
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
        SafetyCarCircle1-v0
      </div>
    </td>
  </tr>
  <tr>
    <td style="text-align:center">
      <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/off-policy/benchmarks/base_cargoal1_3e6.png">
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
        SafetyCarGoal1-v0
      </div>
    </td>
  </tr>
  <tr>
    <td style="text-align:center">
      <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/off-policy/benchmarks/base_pointcircle1_3e6.png">
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
        SafetyPointCircle1-v0
      </div>
    </td>
  </tr>
  <tr>
    <td style="text-align:center">
      <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/off-policy/benchmarks/base_pointgoal1_3e6.png">
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
        SafetyPointGoal1-v0
      </div>
    </td>
  </tr>
  <caption><p style="font-family: 'Times New Roman', Times, serif;"><b>Figure 1:</b> Training curves in Safety-Gymnasium environments, covering classical reinforcement learning algorithms mentioned in <a href="#compare_off_policy">Table 1</a> and safe reinforcement learning algorithms.</p></caption>
</table>

<table id="curve_off_lag">
  <tr>
    <td style="text-align:center">
      <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/off-policy/benchmarks/lag_ant_1e6.png">
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
        SafetyAntVelocity-v1
      </div>
    </td>
  </tr>
  <tr>
    <td style="text-align:center">
      <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/off-policy/benchmarks/lag_halfcheetah_1e6.png">
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
        SafetyHalfCheetahVelocity-v1
      </div>
    </td>
  </tr>
  <tr>
    <td style="text-align:center">
      <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/off-policy/benchmarks/lag_hopper_1e6.png">
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
        SafetyHopperVelocity-v1
      </div>
    </td>
  </tr>
  <tr>
    <td style="text-align:center">
      <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/off-policy/benchmarks/lag_humanoid_3e6.png">
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
        SafetyHumanoidVelocity-v1
      </div>
    </td>
  </tr>
  <tr>
    <td style="text-align:center">
      <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/off-policy/benchmarks/lag_swimmer_1e6.png">
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
        SafetySwimmerVelocity-v1
      </div>
    </td>
  </tr>
  <tr>
    <td style="text-align:center">
      <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/off-policy/benchmarks/lag_walker2d_1e6.png">
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
        SafetyWalker2dVelocity-v1
      </div>
    </td>
  </tr>
  <tr>
    <td style="text-align:center">
      <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/off-policy/benchmarks/lag_carcircle1_3e6.png">
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
        SafetyCarCircle1-v0
      </div>
    </td>
  </tr>
  <tr>
    <td style="text-align:center">
      <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/off-policy/benchmarks/lag_cargoal1_3e6.png">
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
        SafetyCarGoal1-v0
      </div>
    </td>
  </tr>
  <tr>
    <td style="text-align:center">
      <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/off-policy/benchmarks/lag_pointcircle1_3e6.png">
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
        SafetyPointCircle1-v0
      </div>
    </td>
  </tr>
  <tr>
    <td style="text-align:center">
      <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/off-policy/benchmarks/lag_pointgoal1_3e6.png">
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
        SafetyPointGoal1-v0
      </div>
    </td>
  </tr>
  <caption><p style="font-family: 'Times New Roman', Times, serif;"><b>Figure 2:</b> Training curves in Safety-Gymnasium environments, covering lagrangian reinforcement learning algorithms mentioned in <a href="#compare_off_policy">Table 1</a> and safe reinforcement learning algorithms.</p></caption>
</table>

<table id="curve_off_pid">
  <tr>
    <td style="text-align:center">
      <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/off-policy/benchmarks/pid_ant_1e6.png">
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
        SafetyAntVelocity-v1
      </div>
    </td>
  </tr>
  <tr>
    <td style="text-align:center">
      <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/off-policy/benchmarks/pid_halfcheetah_1e6.png">
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
        SafetyHalfCheetahVelocity-v1
      </div>
    </td>
  </tr>
  <tr>
    <td style="text-align:center">
      <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/off-policy/benchmarks/pid_hopper_1e6.png">
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
        SafetyHopperVelocity-v1
      </div>
    </td>
  </tr>
  <tr>
    <td style="text-align:center">
      <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/off-policy/benchmarks/pid_humanoid_3e6.png">
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
        SafetyHumanoidVelocity-v1
      </div>
    </td>
  </tr>
  <tr>
    <td style="text-align:center">
      <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/off-policy/benchmarks/pid_swimmer_1e6.png">
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
        SafetySwimmerVelocity-v1
      </div>
    </td>
  </tr>
  <tr>
    <td style="text-align:center">
      <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/off-policy/benchmarks/pid_walker2d_1e6.png">
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
        SafetyWalker2dVelocity-v1
      </div>
    </td>
  </tr>
  <tr>
    <td style="text-align:center">
      <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/off-policy/benchmarks/pid_carcircle1_3e6.png">
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
        SafetyCarCircle1-v0
      </div>
    </td>
  </tr>
  <tr>
    <td style="text-align:center">
      <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/off-policy/benchmarks/pid_cargoal1_3e6.png">
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
        SafetyCarGoal1-v0
      </div>
    </td>
  </tr>
  <tr>
    <td style="text-align:center">
      <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/off-policy/benchmarks/pid_pointcircle1_3e6.png">
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
        SafetyPointCircle1-v0
      </div>
    </td>
  </tr>
  <tr>
    <td style="text-align:center">
      <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/off-policy/benchmarks/pid_pointgoal1_3e6.png">
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
        SafetyPointGoal1-v0
      </div>
    </td>
  </tr>
  <caption><p style="font-family: 'Times New Roman', Times, serif;"><b>Figure 3:</b> Training curves in Safety-Gymnasium environments, covering pid-lagrangian reinforcement learning algorithms mentioned in <a href="#compare_off_policy">Table 1</a> and safe reinforcement learning algorithms.</p></caption>
</table>

## Experiment Analysis

### Hyperparameters

**We are continuously improving the performance of off-policy algorithms and finding better hyperparameters. So we are happy to receive any advice from users, feel free for opening an [issue](https://github.com/PKU-Alignment/omnisafe/issues/new/choose) or [pull request](https://github.com/PKU-Alignment/omnisafe/pulls).**

Off-policy algorithms almost share the same hyperparameters, which are listed below:

|   Hyperparameter    |   Value    |
| :-----------------: | :--------: |
|   `batch_size`    |    256     |
|      `gamma`      |    0.99    |
|      `size`       |  1000000   |
|  `update_iters`   |     1      |
|  `update_cycle`   |     1      |
|  `hidden_sizes`   | [256, 256] |

However, there are some differences between the three algorithms. We list the differences below:

#### TD3

|      Hyperparameter      | Value |
| :----------------------: | :---: |
|     `policy_noise`     |  0.2  |
|      `noise_clip`      |  0.5  |
| `start_learning_steps` | 25000 |

#### SAC

|      Hyperparameter      | Value |
| :----------------------: | :---: |
|        `alpha`         |  0.2  |
| `start_learning_steps` | 5000  |

### Lagrangian

The lagrangian versions of off-policy algorithms share the same set of
lagrangian hyperparameters. We recommend:

|      Hyperparameter      | Value |
| :----------------------: | :---: |
| `cost_limit` | 25.0  |
| `lagrangian_multiplier_init` | 0.001  |
| `lambda_lr` | 0.00001  |
| `lambda_optimizer` | Adam |

for Velocity tasks such as:

- `SafetyAntVelocity-v1`
- `SafetyHalfCheetahVelocity-v1`
- `SafetyHopperVelocity-v1`
- `SafetyHumanoidVelocity-v1`
- `SafetyWalker2dVelocity-v1`
- `SafetySwimmerVelocity-v1`

Then:

|      Hyperparameter      | Value |
| :----------------------: | :---: |
| `cost_limit` | 25.0  |
| `lagrangian_multiplier_init` | 0.000  |
| `lambda_lr` | 0.0000005  |
| `lambda_optimizer` | Adam |

for Navigation tasks such as:

- `SafetyCarCircle1-v0`
- `SafetyCarGoal1-v0`
- `SafetyPointCircle1-v0`
- `SafetyPointGoal1-v0`

### Learning Rate

In our experiments, we found that in the Navigation task, when the actor learning rate (`actor:lr`) is too large, the value of the `cost_critic` decreases rapidly, even becoming negative. We attribute this phenomenon to the fact that compared to estimating rewards, cost is relatively sparse and more difficult to estimate accurately. When the `actor:lr` is too large, the cost_critic becomes highly susceptible to the actor's influence, leading to an inaccurate estimation of the cost and subsequently affecting the actor's updates. Therefore, we attempted to decrease the `actor:lr` and achieved a promising performance as a result.

### PID-Lagrangian

PID-Lagrangian methods use a PID controller to control the lagrangian multiplier, The `pid_kp`, `pid_kd` and `pid_ki` count for the proportional gain, derivative gain and integral gain of the PID controller respectively. As PID-Lagrangian methods use a PID controller to control the lagrangian multiplier, the hyperparameters of the PID controller are important for the performance of the algorithm.

- `pid_kp`: The proportional gain of the PID controller, determines how much the output responds to changes in the `ep_costs` signal. If the `pid_kp` is too large, the lagrangian multiplier will oscillate and the performance will be bad. If the `pid_kp` is too small, the lagrangian multiplier will update slowly and the performance will also be bad.
- `pid_kd`: The derivative gain of the PID controller, determines how much the output responds to changes in the `ep_costs` signal. If the `pid_kd` is too large, the lagrangian multiplier may be too sensitive to noise or changes in the `ep_costs` signal, leading to instability or oscillations. If the `pid_kd` is too small, the lagrangian multiplier may not respond quickly or accurately enough to changes in the `ep_costs`.
- `pid_ki`: The integral gain of the PID controller, determines the controller's ability to eliminate the steady-state error, by integrating the `ep_costs` signal over time. If the `pid_ki` is too large, the lagrangian multiplier may become too responsive to errors before.

We have done some experiments to find relatively good `pid_kp`, `pid_ki`, and `pid_kd` for all environments, and we found that the following value is a good value for this hyperparameter.

| Parameters | Descriptions| Values |
| -----------| ------------| ------ |
|`pid_kp`|The proportional gain of the PID controller|0.000001|
|`pid_ki`|The derivative gain of the PID controller|0.0000001|
|`pid_kd`|The integral gain of the PID controller|0.0000001|

### Some Hints

In our experiments, we found that some hyperparameters are important for the performance of the algorithm:

- `obs_normalize`: Whether to normalize the observation.
- `reward_normalize`: Whether to normalize the reward.
- `cost_normalize`: Whether to normalize the cost.

We have done some experiments to show the effect of these hyperparameters, and we log the best configuration for each algorithm to conquer all environments. You can check out the `omnisafe/configs/off_policy`.

Generally, we recommend:

|      Hyperparameter      | Value |
| :----------------------: | :---: |
| `obs_normalize`   |  `False`  |
| `reward_normalize`|  `False`  |
| `cost_normalize`  | `True` |

for Velocity tasks such as:

- `SafetyAntVelocity-v1`
- `SafetyHalfCheetahVelocity-v1`
- `SafetyHopperVelocity-v1`
- `SafetyHumanoidVelocity-v1`
- `SafetyWalker2dVelocity-v1`
- `SafetySwimmerVelocity-v1`

Then:

|      Hyperparameter      | Value |
| :----------------------: | :---: |
| `obs_normalize`   |  `False`  |
| `reward_normalize`|  `False`  |
| `cost_normalize`  | `False` |

for Navigation tasks such as:

- `SafetyCarCircle1-v0`
- `SafetyCarGoal1-v0`
- `SafetyPointCircle1-v0`
- `SafetyPointGoal1-v0`

Besides, the hyperparameter `torch_num_threads` in `train_cfgs` is also important. In a single training session, a larger value for `torch_num_threads` often means faster training speed. However, we found in experiments that setting `torch_num_threads` too high can cause resource contention between parallel training sessions, resulting in slower overall experiment speed. In the configs file, we set the default value for `torch_num_threads` to 16, which ensures faster training speed for a single session. If you need to launch multiple training sessions in parallel, please consider your device configuration. For example, suppose your CPU has 2 physical threads per core and has a total of 32 cores, if you are launching 16 training scripts in parallel, you had better not set `torch_num_threads` to a value greater than 4.

If you find that other hyperparameters perform better, please feel free to open an [issue](https://github.com/PKU-Alignment/omnisafe/issues/new/choose) or [pull request](https://github.com/PKU-Alignment/omnisafe/pulls).

<a name="footnote1">[1]</a>  This paper is [safety-gym](https://openai.com/research/safety-gym) original paper. Its public code base [safety-starter-agents](https://github.com/openai/safety-starter-agents) implemented `SACLag` but does not report it in the paper.  We can not find the source of `DDPGLag` and `TD3Lag`. However, this paper introduced lagrangian methods and it implemented `SACLag`, so we also use it as a source of `DDPGLag` and `TD3Lag`.
