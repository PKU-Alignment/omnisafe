# OmniSafe's Mujoco Velocity Benchmark on Off-Policy Algorithms

OmniSafe's Mujoco Velocity Benchmark evaluated the performance of OmniSafe algorithm implementations in 6 environments from the Safety-Gymnasium task suite. For each algorithm and environment supported, we provide:
- Default hyperparameters used for the benchmark and scripts to reproduce the results
- A comparison of performance or code-level details with other open-source implementations or classic papers.
- Graphs and raw data that can be used for research purposes, - Log details obtained during training
- Some hints on how to fine-tune the algorithm for optimal results.

Supported algorithms are listed below:
- [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/pdf/1509.02971.pdf)
- [Twin Delayed DDPG (TD3)](https://arxiv.org/pdf/1802.09477.pdf)
- [Soft Actor-Critic (SAC)](https://arxiv.org/pdf/1812.05905.pdf)
- [DDPGLag](https://cdn.openai.com/safexp-short.pdf)
- [TD3Lag](https://cdn.openai.com/safexp-short.pdf)
- [SACLag](https://cdn.openai.com/safexp-short.pdf)

## Safety-Gymnasium

We highly recommend using ``safety-gymnasium`` to run the following experiments. To install, in a linux machine, type:

```bash
pip install safety_gymnasium
```

## Run the Benchmark
You can set the main function of ``examples/benchmarks/experimrnt_grid.py`` as:

```python
    eg = ExperimentGrid(exp_name='Off-Policy-Velocity')

    # set up the algorithms.
    off_policy = ['DDPG', 'SAC', 'TD3']
    eg.add('algo', off_policy)

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

For example, if I train ``DDPG`` in ``SafetyHumanoidVelocity-v1``

```python
    LOG_DIR = '~/omnisafe/examples/runs/DDPG-<SSafetyHumanoidVelocity-v1>/seed-000-2023-03-07-20-25-48'
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

In an effort to ascertain the credibility of OmniSafe ’s algorithmic implementation, a com-
parative assessment was conducted, juxtaposing the performance of classical reinforcement
learning algorithms. Such as DDPG, TD3 and SAC. The performance table is provided in <a href="#compare_off_policy">Table 1</a>. with
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
class="math inline">±</span> 318.6</td>
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
<td style="text-align: center;"><strong>6012.3 <span
class="math inline">±</span> 102.64</strong></td>
<td style="text-align: center;">2404.5 <span
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
class="math inline">±</span> 1045.2</td>
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
class="math inline">±</span> 1832.9</td>
<td style="text-align: center;">3511.06 <span
class="math inline">±</span> 2214.12</td>
<td style="text-align: center;"><strong>6039.77 <span
class="math inline">±</span> 167.82</strong></td>
<td style="text-align: center;">5424.55 <span
class="math inline">±</span> 118.52</td>
<td style="text-align: center;">2713.6 <span
class="math inline">±</span> 2256.89</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span
class="smallcaps">SafetySwimmerVelocity-v1</span></td>
<td style="text-align: center;">139.39 <span
class="math inline">±</span> 11.74</td>
<td style="text-align: center;">138.98 <span
class="math inline">±</span> 8.6</td>
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
<td style="text-align: center;">247.33 <span
class="math inline">±</span> 122.02</td>
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
  <caption><p style="font-family: 'Times New Roman', Times, serif;"><b>Table 1:</b> The performance of OmniSafe off-policy algorithms, which was evaluated about published baselines within the Safety-Gymnasium MuJoCo Velocity environments. Experimental outcomes, comprising mean and standard deviation, were derived from 10 assessment iterations encompassing multiple random seeds. A noteworthy distinction lies in the fact that Stable-Baselines3 employs distinct parameters tailored to each environment, while OmniSafe maintains a consistent parameter set across all environments.</p></caption>
</table>

### Safe Reinforcement Learning Algorithms

Serving as a reliable SafeRL baseline, OmniSafe offers performance insights for SafeRL
algorithms within the Safety-Gymnasium environment. The results of OmniSafe are
presented in <a href="#performance_off_policy">Table 2</a> and <a href="#curve_off_policy">Figure 1</a>.

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
<td style="text-align: center;">3435.70 <span
class="math inline">±</span> 52.48</td>
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
<td style="text-align: center;">90.10 <span class="math inline">±</span>
29.01</td>
<td style="text-align: center;">115.27 <span
class="math inline">±</span> 44.90</td>
<td style="text-align: center;">44.30 <span class="math inline">±</span>
0.49</td>
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
<thead>
<tr class="header">
<th style="text-align: left;"></th>
<th colspan="2" style="text-align: center;"><strong>DDPGLag</strong></th>
<th colspan="2" style="text-align: center;"><strong>TD3Lag</strong></th>
<th colspan="2" style="text-align: center;"><strong>SACLag</strong></th>
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
</tbody>
  <caption><p style="font-family: 'Times New Roman', Times, serif;"><b>Table 2:</b> The performance of OmniSafe off-policy algorithms, encompassing both reward and cost, was assessed within the Safety-Gymnasium environments. During experimentation, it was observed that off-policy algorithms did not violate safety constraints in SafetyHumanoidVeloicty-v1. This observation suggests that the agent may not have fully learned to run within 1e6 steps; consequently, the 3e6 results were utilized in off-policy SafetyHumanoidVeloicty-v1. With this exception in consideration, all off-policy algorithms were evaluated after 1e6 training steps.</p></caption>
</table>


#### Performance Curves

<table id="curve_off_policy">
  <tr>
    <td style="text-align:center">
      <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="./benchmarks/Ant.png">
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
        SafetyHumanoidVelocity-v1
      </div>
    </td>
  </tr>
  <tr>
    <td style="text-align:center">
      <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="./benchmarks/Halfcheetah.png">
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
        SafetyHalfcheetahVelocity-v1
      </div>
    </td>
  </tr>
  <tr>
    <td style="text-align:center">
      <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="./benchmarks/Hopper.png">
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
        SafetyHopperVelocity-v1
      </div>
    </td>
  </tr>
  <tr>
    <td style="text-align:center">
      <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="./benchmarks/Humanoid.png">
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
        SafetyHumanoidVelocity-v1
      </div>
    </td>
  </tr>
  <tr>
    <td style="text-align:center">
      <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="./benchmarks/Walker2d.png">
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
        SafetyWalker2dVelocity-v1
      </div>
    </td>
  </tr>
  <tr>
    <td style="text-align:center">
      <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="./benchmarks/Swimmer.png">
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
        SafetySwimmerVelocity-v1
      </div>
    </td>
  </tr>
  <caption><p style="font-family: 'Times New Roman', Times, serif;"><b>Figure 1:</b> Training curves in Safety-Gymnasium MuJoCo Velocity environments, covering classical reinforcement learning algorithms mentioned in <a href="#compare_off_policy">Table 1</a> and safe reinforcement learning algorithms. The rewards are obtained from the 1e6 steps interaction.</p></caption>
</table>

## Experiment Analysis

### Hyperparameters

Off-Policy algorithms almost share the same hyperparameters, the share hyperparameters are listed below:

|   Hyperparameter    |   Value    |
| :-----------------: | :--------: |
|   ``batch_size``    |    256     |
|      ``gamma``      |    0.99    |
|      ``size``       |  1000000   |
|  ``update_iters``   |     1      |
| ``step_per_sample`` |     1      |
|  ``hidden_sizes``   | [256, 256] |

However, there are some differences between the three algorithms. We list the differences below:


### TD3

|      Hyperparameter      | Value |
| :----------------------: | :---: |
|     ``policy_noise``     |  0.2  |
|      ``noise_clip``      |  0.5  |
| ``start_learning_steps`` | 25000 |

### SAC

|      Hyperparameter      | Value |
| :----------------------: | :---: |
|        ``alpha``         |  0.2  |
| ``start_learning_steps`` | 5000  |

### Lagragian

The lagrangian version of off-policy algorithms share the same set of
lagrangian hyperparameters. The hyperparameters are listed below:

|      Hyperparameter      | Value |
| :----------------------: | :---: |
| ``cost_limit`` | 25.0  |
| ``lagrangian_multiplier_init`` | 0.001  |
| ``lambda_lr`` | 0.00001  |
| ``lambda_optimizer`` | Adam |

### Some Hints

In our experiments, we found that some hyperparameters are important for the performance of the algorithm:

- ``obs_normlize``: Whether to normalize the observation.
- ``rew_normlize``: Whether to normalize the reward.
- ``cost_normlize``: Whether to normalize the cost.

We have done some experiments to show the effect of these hyperparameters, and we log the best configuration for each algorithm in each environment. You can check it in the ``omnisafe/configs/off_policy``.

Generally, we recommend not to normalize the observation and reward, but not the cost.

Besides, the hyperparameter ``train_cfgs:torch_num_threads`` is also important. Off-policy algorithms always use more time to update policy than to sample data. So we use ``torch_num_threads`` to speed up the update process.

This hyperparamter depens on the number of CPU cores. We set it to 8 in our experiments. You can set it to some other porper value according to your CPU cores.

If you find that other hyperparameters perform better, please feel free to open an issue or pull request.
