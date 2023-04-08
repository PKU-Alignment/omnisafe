<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->

<div align="center">
  <img src="https://github.com/OmniSafeAI/omnisafe/raw/HEAD/images/logo.png" width="75%"/>
</div>

<div align="center">

  [![Organization](https://img.shields.io/badge/Organization-PKU_MARL-blue.svg)](https://github.com/OmniSafeAI)
  [![PyPI](https://img.shields.io/pypi/v/omnisafe?logo=pypi)](https://pypi.org/project/omnisafe)
  [![tests](https://img.shields.io/github/actions/workflow/status/OmniSafeAI/omnisafe/test.yml?label=tests&logo=github)](https://github.com/OmniSafeAI/omnisafe/tree/HEAD/tests)
  [![Documentation Status](https://img.shields.io/readthedocs/omnisafe?logo=readthedocs)](https://omnisafe.readthedocs.io)
  [![Downloads](https://static.pepy.tech/personalized-badge/omnisafe?period=total&left_color=grey&right_color=blue&left_text=downloads)](https://pepy.tech/project/omnisafe)
  [![GitHub Repo Stars](https://img.shields.io/github/stars/OmniSafeAI/omnisafe?color=brightgreen&logo=github)](https://github.com/OmniSafeAI/OmniSafe/stargazers)
  [![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
  [![License](https://img.shields.io/github/license/OmniSafeAI/OmniSafe?label=license)](#license)
  [![CodeCov](https://img.shields.io/codecov/c/github/OmniSafeAI/omnisafe/main?logo=codecov)](https://app.codecov.io/gh/OmniSafeAI/omnisafe)
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OmniSafeAI/omnisafe/)

</div>

<p align="center">
  <a href="https://omnisafe.readthedocs.io">Documentation</a> |
  <a href="https://github.com/OmniSafeAI/omnisafe#implemented-algorithms">Implemented Algorithms</a> |
  <a href="https://github.com/OmniSafeAI/omnisafe#installation">Installation</a> |
  <a href="https://github.com/OmniSafeAI/omnisafe#getting-started">Getting Started</a> |
  <a href="https://github.com/OmniSafeAI/omnisafe#license">License</a>
</p>

--------------------------------------------------------------------------------

**This library is currently under heavy development - if you have suggestions on the API or use-cases you'd like to be covered, please open a GitHub issue or reach out. We'd love to hear about how you're using the library.**

OmniSafe is an infrastructural framework designed to accelerate safe reinforcement learning (RL) research by providing a comprehensive and reliable benchmark for safe RL algorithms. The field of RL has great potential to benefit society, but safety concerns are a significant issue, and RL algorithms have raised concerns about unintended harm or unsafe behavior. Safe RL intends to develop algorithms that minimize the risk of unintended harm or unsafe behavior, but there is currently a lack of commonly recognized safe RL algorithm benchmarks.

OmniSafe addresses these issues by providing more than 40 experimentally validated algorithms and a sound and efficient simulation environment. Researchers can use OmniSafe to conduct experiments and verify their ideas, ensuring consistency and enabling more efficient development of safe RL algorithms. By using OmniSafe as a benchmark, researchers can evaluate the performance of their own safe RL algorithms and contribute to the advancement of safe RL research.

--------------------------------------------------------------------------------

### Table of Contents  <!-- omit in toc --> <!-- markdownlint-disable heading-increment -->

- [Implemented Algorithms](#implemented-algorithms)
  - [Latest SafeRL Papers](#latest-saferl-papers)
  - [List of Algorithms](#list-of-algorithms)
    - [On-Policy Safe](#on-policy-safe)
    - [Off-Policy Safe](#off-policy-safe)
    - [Model-Based Safe](#model-based-safe)
    - [Offline Safe](#offline-safe)
    - [Others](#others)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
    - [Install from source](#install-from-source)
    - [Install from PyPI](#install-from-pypi)
  - [Examples](#examples)
    - [Try with CLI](#try-with-cli)
- [Getting Started](#getting-started)
  - [Important Hints](#important-hints)
  - [Quickstart: Colab in the Cloud](#quickstart-colab-in-the-cloud)
- [Changelog](#changelog)
- [The OmniSafe Team](#the-omnisafe-team)
- [License](#license)

--------------------------------------------------------------------------------

## Implemented Algorithms

The supported interface algorithms currently include:

### Latest SafeRL Papers

- **[AAAI 2023]** Augmented Proximal Policy Optimization for Safe Reinforcement Learning (APPO)
- **[NeurIPS 2022]** [Constrained Update Projection Approach to Safe Policy Optimization (CUP)](https://arxiv.org/abs/2209.07089)
- **[NeurIPS 2022]** [Effects of Safety State Augmentation on Safe Exploration (Simmer)](https://arxiv.org/abs/2206.02675)
- **[NeurIPS 2022]** [Model-based Safe Deep Reinforcement Learning via a Constrained Proximal Policy Optimization Algorithm](https://arxiv.org/abs/2210.07573)
- **[ICML 2022]** [Sauté RL: Almost Surely Safe Reinforcement Learning Using State Augmentation (SauteRL)](https://arxiv.org/abs/2202.06558)
- **[ICML 2022]** [Constrained Variational Policy Optimization for Safe Reinforcement Learning (CVPO)](https://arxiv.org/abs/2201.11927)
- **[IJCAI 2022]** [Penalized Proximal Policy Optimization for Safe Reinforcement Learning](https://arxiv.org/abs/2205.11814)
- **[ICLR 2022]** [Constrained Policy Optimization via Bayesian World Models (LA-MBDA)](https://arxiv.org/abs/2201.09802)
- **[AAAI 2022]** [Conservative and Adaptive Penalty for Model-Based Safe Reinforcement Learning (CAP)](https://arxiv.org/abs/2112.07701)

### List of Algorithms

#### On-Policy Safe

- [X] [The Lagrange version of PPO (PPO-Lag)](https://cdn.openai.com/safexp-short.pdf)
- [X] [The Lagrange version of TRPO (TRPO-Lag)](https://cdn.openai.com/safexp-short.pdf)
- [X] **[ICML 2017]** [Constrained Policy Optimization (CPO)](https://proceedings.mlr.press/v70/achiam17a)
- [X] **[ICLR 2019]** [Reward Constrained Policy Optimization (RCPO)](https://openreview.net/forum?id=SkfrvsA9FX)
- [X] **[ICML 2020]** [Responsive Safety in Reinforcement Learning by PID Lagrangian Methods (PID-Lag)](https://arxiv.org/abs/2007.03964)
- [X] **[NeurIPS 2020]** [First Order Constrained Optimization in Policy Space (FOCOPS)](https://arxiv.org/abs/2002.06506)
- [X] **[AAAI 2020]** [IPO: Interior-point Policy Optimization under Constraints (IPO)](https://arxiv.org/abs/1910.09615)
- [X] **[ICLR 2020]** [Projection-Based Constrained Policy Optimization (PCPO)](https://openreview.net/forum?id=rke3TJrtPS)
- [X] **[ICML 2021]** [CRPO: A New Approach for Safe Reinforcement Learning with Convergence Guarantee](https://arxiv.org/abs/2011.05869)
- [x] **[IJCAI 2022]** [Penalized Proximal Policy Optimization for Safe Reinforcement Learning(P3O)](https://arxiv.org/pdf/2205.11814.pdf)

#### Off-Policy Safe

- [X] The Lagrange version of TD3 (TD3-Lag)
- [X] The Lagrange version of DDPG (DDPG-Lag)
- [X] The Lagrange version of SAC (SAC-Lag)
- [X] **[ICML 2019]** [Lyapunov-based Safe Policy Optimization for Continuous Control (SDDPG)](https://arxiv.org/abs/1901.10031)
- [X] **[ICML 2019]** [Lyapunov-based Safe Policy Optimization for Continuous Control (SDDPG-modular)](https://arxiv.org/abs/1901.10031)
- [ ] **[ICML 2022]** [Constrained Variational Policy Optimization for Safe Reinforcement Learning (CVPO)](https://arxiv.org/abs/2201.11927)

#### Model-Based Safe

- [ ] **[NeurIPS 2021]** [Safe Reinforcement Learning by Imagining the Near Future (SMBPO)](https://arxiv.org/abs/2202.07789)
- [X] **[CoRL 2021 (Oral)]** [Learning Off-Policy with Online Planning (SafeLOOP)](https://arxiv.org/abs/2008.10066)
- [X] **[AAAI 2022]** [Conservative and Adaptive Penalty for Model-Based Safe Reinforcement Learning (CAP)](https://arxiv.org/abs/2112.07701)
- [X] **[NeurIPS 2022]** [Model-based Safe Deep Reinforcement Learning via a Constrained Proximal Policy Optimization Algorithm](https://arxiv.org/abs/2210.07573)
- [ ] **[ICLR 2022]** [Constrained Policy Optimization via Bayesian World Models (LA-MBDA)](https://arxiv.org/abs/2201.09802)

#### Offline Safe

- [X] [The Lagrange version of BCQ (BCQ-Lag)](https://arxiv.org/abs/1812.02900)
- [X] [The Constrained version of CRR (C-CRR)](https://proceedings.neurips.cc/paper/2020/hash/588cb956d6bbe67078f29f8de420a13d-Abstract.html)
- [ ] **[AAAI 2022]** [Constraints Penalized Q-learning for Safe Offline Reinforcement Learning CPQ](https://arxiv.org/abs/2107.09003)
- [ ] **[ICLR 2022 (Spotlight)]** [COptiDICE: Offline Constrained Reinforcement Learning via Stationary Distribution Correction Estimation](https://arxiv.org/abs/2204.08957?context=cs.AI)
- [ ] **[ICML 2022]** [Constrained Offline Policy Optimization (COPO)](https://proceedings.mlr.press/v162/polosky22a.html)

#### Others

- [X] [Safe Exploration in Continuous Action Spaces (Safety Layer)](https://arxiv.org/abs/1801.08757)
- [ ] **[RA-L 2021]** [Recovery RL: Safe Reinforcement Learning with Learned Recovery Zones](https://arxiv.org/abs/2010.15920)
- [X] **[ICML 2022]** [Sauté RL: Almost Surely Safe Reinforcement Learning Using State Augmentation (SauteRL)](https://arxiv.org/abs/2202.06558)
- [X] **[NeurIPS 2022]** [Effects of Safety State Augmentation on Safe Exploration](https://arxiv.org/abs/2206.02675)

--------------------------------------------------------------------------------

## Installation

### Prerequisites

OmniSafe requires Python 3.8+ and PyTorch 1.10+.

> We support and test for Python 3.8, 3.9, 3.10 on Linux. The support of M1 and M2 versions of macOS is undergoing internal verification. We will accept PRs related to Windows, but do not officially support it.

#### Install from source

```bash
# Clone the repo
git clone https://github.com/OmniSafeAI/omnisafe
cd omnisafe

# Create a conda environment
conda create -n omnisafe python=3.8
conda activate omnisafe

# Install omnisafe
pip install -e .
```

#### Install from PyPI
OmniSafe is hosted in [![PyPI](https://img.shields.io/pypi/v/omnisafe?label=pypi&logo=pypi)](https://pypi.org/project/omnisafe) / ![Status](https://img.shields.io/pypi/status/omnisafe?label=status).
```bash
pip install omnisafe
```


### Examples

```bash
cd examples
python train_policy.py --algo PPOLag --env-id SafetyPointGoal1-v0 --parallel 1 --total-steps 1024000 --device cpu --vector-env-nums 1 --torch-threads 1
```


**algo:**
| Type              | Name                                                             |
| ----------------- | ---------------------------------------------------------------- |
| `Base-On-Policy`  | `PolicyGradient, PPO`<br> `NaturalPG, TRPO`                      |
| `Base-Off-Policy` | `DDPG, TD3, SAC`                                                 |
| `Naive Lagrange`  | `RCPO, PPOLag, TRPOLag`<br> `DDPGLag, TD3Lag, SACLag`            |
| `PID Lagrange`    | `CPPOPid, TRPOPid`                                               |
| `First Order`     | `FOCOPS, CUP`                                                    |
| `Second Order`    | `SDDPG, CPO, PCPO`                                               |
| `Saute RL`        | `PPOSaute, PPOLagSaute`                                          |
| `Simmer RL`       | `PPOSimmerQ, PPOSimmerPid` <br> `PPOLagSimmerQ, PPOLagSimmerPid` |
| `EarlyTerminated` | `PPOEarlyTerminated` <br> `PPOLagEarlyTerminated`                |
| `Model-Based`     | `CAP, MBPPOLag, SafeLOOP`                                        |


**env-id:** Environment id in [Safety Gymnasium](https://www.safety-gymnasium.com/), here a list of envs that safety-gymnasium supports.

<table border="1">
<thead>
  <tr>
    <th>Category</th>
    <th>Task</th>
    <th>Agent</th>
    <th>Example</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="4">Safe Navigation</td>
    <td>Goal[012]</td>
    <td rowspan="4">Point, Car, Racecar, Ant</td>
    <td rowspan="4">SafetyPointGoal1-v0</td>
  </tr>
  <tr>
    <td>Button[012]</td>
  </tr>
  <tr>
    <td>Push[012]</td>
  </tr>
  <tr>
    <td>Circle[012]</td>
  </tr>
  <tr>
    <td>Safe Velocity</td>
    <td>Velocity</td>
    <td>HalfCheetah, Hopper, Swimmer, Walker2d, Ant, Humanoid</td>
    <td>SafetyHumanoidVelocity-v1</td>
  </tr>
</tbody>
</table>

More information about environments, please refer to [Safety Gymnasium](https://www.safety-gymnasium.com/)

**parallel:** `Number of parallels`


#### Try with CLI

**A video example**

![Segmentfault](https://github.com/OmniSafeAI/omnisafe/blob/main/images/CLI_example.svg)

```bash
pip install omnisafe

omnisafe --help # Ask for help

omnisafe benchmark --help # The benchmark also can be replaced with 'eval', 'train', 'train-config'

# Quick benchmarking for your research, just specify: 1.exp_name, 2.num_pool(how much processes are concurrent), 3.path of the config file(refer to omnisafe/examples/benchmarks for format)
omnisafe benchmark test_benchmark 2 ./saved_source/benchmark_config.yaml

# Quick evaluating and rendering your trained policy, just specify: 1.path of algorithm which you trained
omnisafe eval ./saved_source/PPO-{SafetyPointGoal1-v0} --num-episode 1

# Quick training some algorithms to validate your thoughts
# Note: use `key1:key2`, your can select key of hyperparameters which are recursively contained, and use `--custom-cfgs`, you can add custom cfgs via CLI
omnisafe train --algo PPO --total-steps 2048 --vector-env-nums 1 --custom-cfgs algo_cfgs:update_cycle --custom-cfgs 1024

# Quick training some algorithms via a saved config file, the format is as same as default format
omnisafe train-config ./saved_source/train_config.yaml
```


--------------------------------------------------------------------------------

## Getting Started
### Important Hints
- `train_cfgs:torch_threads` is especially important for training speed, and is varying with users' machine, this value shouldn't be too small or too large.

### Quickstart: Colab in the Cloud
Explore OmniSafe easily and quickly through a series of colab notebooks:
- [Getting Started](https://colab.research.google.com/github/OmniSafeAI/omnisafe/blob/main/tutorials/English/1.Getting_Started.ipynb) Introduce the basic usage of OmniSafe so that users can quickly hand on it.
- [CLI Command](https://colab.research.google.com/github/OmniSafeAI/omnisafe/blob/main/tutorials/English/2.CLI_Command.ipynb) Introduce how to use the CLI tool of OmniSafe.

We take great pleasure in collaborating with our users to create tutorials in various languages. Please refer to our list of currently supported languages. If you are interested in translating the tutorial into a new language or improving an existing version, kindly submit a PR to us."

--------------------------------------------------------------------------------

## Changelog
See [CHANGELOG.md](https://github.com/OmniSafeAI/omnisafe/blob/main/CHANGELOG.md).

## The OmniSafe Team

OmniSafe is mainly developed by the SafeRL research team directed by Prof. [Yaodong Yang](https://github.com/orgs/OmniSafeAI/people/PKU-YYang). Our SafeRL research team members include [Borong Zhang](https://github.com/muchvo), [Jiayi Zhou](https://github.com/Gaiejj), [JTao Dai](https://github.com/calico-1226), [Weidong Huang](https://github.com/hdadong), [Ruiyang Sun](https://github.com/rockmagma02), [Xuehai Pan](https://github.com/XuehaiPan) and [Jiaming Ji](https://github.com/zmsn-2077). If you have any questions in the process of using omnisafe, don't hesitate to ask your questions on [the GitHub issue page](https://github.com/OmniSafeAI/omnisafe/issues/new/choose), we will reply to you in 2-3 working days.

## License

OmniSafe is released under Apache License 2.0.
