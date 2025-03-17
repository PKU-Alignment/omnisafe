<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->

<div align="center">
  <img src="https://github.com/PKU-Alignment/omnisafe/raw/HEAD/images/logo.png" width="75%"/>
</div>

<div align="center">

  [![Organization](https://img.shields.io/badge/Organization-PKU--Alignment-blue)](https://github.com/PKU-Alignment)
  [![PyPI](https://img.shields.io/pypi/v/omnisafe?logo=pypi)](https://pypi.org/project/omnisafe)
  [![tests](https://img.shields.io/github/actions/workflow/status/PKU-Alignment/omnisafe/test.yml?label=tests&logo=github)](https://github.com/PKU-Alignment/omnisafe/tree/HEAD/tests)
  [![Documentation Status](https://img.shields.io/readthedocs/omnisafe?logo=readthedocs)](https://omnisafe.readthedocs.io)
  [![Downloads](https://static.pepy.tech/personalized-badge/omnisafe?period=total&left_color=grey&right_color=blue&left_text=downloads)](https://pepy.tech/project/omnisafe)
  [![GitHub Repo Stars](https://img.shields.io/github/stars/PKU-Alignment/omnisafe?color=brightgreen&logo=github)](https://github.com/PKU-Alignment/OmniSafe/stargazers)
  [![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
  [![License](https://img.shields.io/github/license/PKU-Alignment/OmniSafe?label=license)](#license)
  [![CodeCov](https://img.shields.io/codecov/c/github/PKU-Alignment/omnisafe/main?logo=codecov)](https://app.codecov.io/gh/PKU-Alignment/omnisafe)
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PKU-Alignment/omnisafe/)

</div>

<p align="center">
  <a href="https://omnisafe.readthedocs.io">Documentation</a> |
  <a href="https://github.com/PKU-Alignment/omnisafe#implemented-algorithms">Implemented Algorithms</a> |
  <a href="https://github.com/PKU-Alignment/omnisafe#installation">Installation</a> |
  <a href="https://github.com/PKU-Alignment/omnisafe#getting-started">Getting Started</a> |
  <a href="https://github.com/PKU-Alignment/omnisafe#license">License</a>
</p>

--------------------------------------------------------------------------------

OmniSafe is an infrastructural framework designed to accelerate safe reinforcement learning (RL) research.
It provides a comprehensive and reliable benchmark for safe RL algorithms, and also an out-of-box modular toolkit for researchers.
SafeRL intends to develop algorithms that minimize the risk of unintended harm or unsafe behavior.

OmniSafe stands as the inaugural unified learning framework in the realm of safe reinforcement learning, aiming to foster the Growth of SafeRL Learning Community.
The key features of OmniSafe:

- **Highly Modular Framework.** OmniSafe presents a highly modular framework, incorporating an extensive collection of tens of algorithms tailored for safe reinforcement learning across diverse domains. This framework is versatile due to its abstraction of various algorithm types and well-designed API, using the Adapter and Wrapper design components to bridge gaps and enable seamless interactions between different components. This design allows for easy extension and customization, making it a powerful tool for developers working with different types of algorithms.

- **High-performance parallel computing acceleration.** By harnessing the capabilities of `torch.distributed`, OmniSafe accelerates the learning process of algorithms
with process parallelism. This enables OmniSafe not only to support environment-level asynchronous parallelism but also incorporates agent asynchronous learning. This methodology bolsters training stability and expedites the training process via the deployment of a parallel exploration mechanism. The integration of agent asynchronous learning in OmniSafe underscores its commitment to providing a versatile and robust platform for advancing SafeRL research.

- **Out-of-box toolkits.** OmniSafe offers customizable toolkits for tasks like training, benchmarking, analyzing, and rendering. [Tutorials](https://github.com/PKU-Alignment/omnisafe#getting-started) and user-friendly [APIs](https://omnisafe.readthedocs.io/en/latest/baserlapi/on_policy.html) make it easy for beginners and average users, while advanced researchers can enhance their efficiency without complex code.

![Train video](https://github-production-user-asset-6210df.s3.amazonaws.com/73586554/237139607-d1e6f413-aa2c-4f68-b8ee-08a4361493a0.gif)

If you find OmniSafe useful or use OmniSafe in your research, please cite it in your publications.

```bibtex
@article{JMLR:v25:23-0681,
  author  = {Jiaming Ji and Jiayi Zhou and Borong Zhang and Juntao Dai and Xuehai Pan and Ruiyang Sun and Weidong Huang and Yiran Geng and Mickel Liu and Yaodong Yang},
  title   = {OmniSafe: An Infrastructure for Accelerating Safe Reinforcement Learning Research},
  journal = {Journal of Machine Learning Research},
  year    = {2024},
  volume  = {25},
  number  = {285},
  pages   = {1--6},
  url     = {http://jmlr.org/papers/v25/23-0681.html}
}
```

--------------------------------------------------------------------------------

### Table of Contents  <!-- omit in toc --> <!-- markdownlint-disable heading-increment -->

- [Quick Start](#quick-start)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Install from source](#install-from-source)
    - [Install from PyPI](#install-from-pypi)
- [Implemented Algorithms](#implemented-algorithms)
  - [Examples](#examples)
    - [Algorithms Registry](#algorithms-registry)
    - [Supported Environments](#supported-environments)
    - [Customizing your environment](#customizing-your-environment)
    - [Try with CLI](#try-with-cli)
- [Getting Started](#getting-started)
  - [Important Hints](#important-hints)
  - [Quickstart: Colab on the Cloud](#quickstart-colab-on-the-cloud)
- [Changelog](#changelog)
- [Citing OmniSafe](#citing-omnisafe)
- [Publications using OmniSafe](#publications-using-omnisafe)
- [The OmniSafe Team](#the-omnisafe-team)
- [License](#license)

--------------------------------------------------------------------------------

## Quick Start

### Installation

#### Prerequisites

OmniSafe requires Python 3.8+ and PyTorch 1.10+.

> We support and test for Python 3.8, 3.9, 3.10 on Linux. Meanwhile, we also support M1 and M2 versions of macOS. We will accept PRs related to Windows, but do not officially support it.

#### Install from source

```bash
# Clone the repo
git clone https://github.com/PKU-Alignment/omnisafe.git
cd omnisafe

# Create a conda environment
conda env create --file conda-recipe.yaml
conda activate omnisafe

# Install omnisafe
pip install -e .
```

#### Install from PyPI

OmniSafe is hosted in [![PyPI](https://img.shields.io/pypi/v/omnisafe?label=pypi&logo=pypi)](https://pypi.org/project/omnisafe) / ![Status](https://img.shields.io/pypi/status/omnisafe?label=status).

```bash
pip install omnisafe
```

## Implemented Algorithms

<details>
<summary><b><big>Latest SafeRL Papers</big></b></summary>

- **[AAAI 2023]** Augmented Proximal Policy Optimization for Safe Reinforcement Learning (APPO)
- **[NeurIPS 2022]** [Constrained Update Projection Approach to Safe Policy Optimization (CUP)](https://arxiv.org/abs/2209.07089)
- **[NeurIPS 2022]** [Effects of Safety State Augmentation on Safe Exploration (Simmer)](https://arxiv.org/abs/2206.02675)
- **[NeurIPS 2022]** [Model-based Safe Deep Reinforcement Learning via a Constrained Proximal Policy Optimization Algorithm](https://arxiv.org/abs/2210.07573)
- **[ICML 2022]** [Sauté RL: Almost Surely Safe Reinforcement Learning Using State Augmentation (SauteRL)](https://arxiv.org/abs/2202.06558)
- **[IJCAI 2022]** [Penalized Proximal Policy Optimization for Safe Reinforcement Learning](https://arxiv.org/abs/2205.11814)
- **[AAAI 2022]** [Conservative and Adaptive Penalty for Model-Based Safe Reinforcement Learning (CAP)](https://arxiv.org/abs/2112.07701)

</details>

<details>
<summary><b><big>List of Algorithms</big></b></summary>

<summary><b><big>On Policy SafeRL</big></b></summary>

- [x] [The Lagrange version of PPO (PPO-Lag)](https://cdn.openai.com/safexp-short.pdf)
- [x] [The Lagrange version of TRPO (TRPO-Lag)](https://cdn.openai.com/safexp-short.pdf)
- [x] **[ICML 2017]** [Constrained Policy Optimization (CPO)](https://proceedings.mlr.press/v70/achiam17a)
- [x] **[ICLR 2019]** [Reward Constrained Policy Optimization (RCPO)](https://openreview.net/forum?id=SkfrvsA9FX)
- [x] **[ICML 2020]** [Responsive Safety in Reinforcement Learning by PID Lagrangian Methods (PID-Lag)](https://arxiv.org/abs/2007.03964)
- [x] **[NeurIPS 2020]** [First Order Constrained Optimization in Policy Space (FOCOPS)](https://arxiv.org/abs/2002.06506)
- [x] **[AAAI 2020]** [IPO: Interior-point Policy Optimization under Constraints (IPO)](https://arxiv.org/abs/1910.09615)
- [x] **[ICLR 2020]** [Projection-Based Constrained Policy Optimization (PCPO)](https://openreview.net/forum?id=rke3TJrtPS)
- [x] **[ICML 2021]** [CRPO: A New Approach for Safe Reinforcement Learning with Convergence Guarantee](https://arxiv.org/abs/2011.05869)
- [x] **[IJCAI 2022]** [Penalized Proximal Policy Optimization for Safe Reinforcement Learning(P3O)](https://arxiv.org/pdf/2205.11814.pdf)

<summary><b><big>Off Policy SafeRL</big></b></summary>

- **[Preprint 2019]** [The Lagrangian version of DDPG (DDPGLag)](https://cdn.openai.com/safexp-short.pdf)
- **[Preprint 2019]** [The Lagrangian version of TD3 (TD3Lag)](https://cdn.openai.com/safexp-short.pdf)
- **[Preprint 2019]** [The Lagrangian version of SAC (SACLag)](https://cdn.openai.com/safexp-short.pdf)
- **[ICML 2020]** [Responsive Safety in Reinforcement Learning by PID Lagrangian Methods (DDPGPID)](https://arxiv.org/abs/2007.03964)
- **[ICML 2020]** [Responsive Safety in Reinforcement Learning by PID Lagrangian Methods (TD3PID)](https://arxiv.org/abs/2007.03964)
- **[ICML 2020]** [Responsive Safety in Reinforcement Learning by PID Lagrangian Methods (SACPID)](https://arxiv.org/abs/2007.03964)

<summary><b><big>Model-Based SafeRL</big></b></summary>

- [ ] **[NeurIPS 2021]** [Safe Reinforcement Learning by Imagining the Near Future (SMBPO)](https://arxiv.org/abs/2202.07789)
- [x] **[CoRL 2021 (Oral)]** [Learning Off-Policy with Online Planning (SafeLOOP)](https://arxiv.org/abs/2008.10066)
- [x] **[AAAI 2022]** [Conservative and Adaptive Penalty for Model-Based Safe Reinforcement Learning (CAP)](https://arxiv.org/abs/2112.07701)
- [x] **[NeurIPS 2022]** [Model-based Safe Deep Reinforcement Learning via a Constrained Proximal Policy Optimization Algorithm](https://arxiv.org/abs/2210.07573)
- [ ] **[ICLR 2022]** [Constrained Policy Optimization via Bayesian World Models (LA-MBDA)](https://arxiv.org/abs/2201.09802)
- [x] **[ICML 2022 Workshop]** [Constrained Model-based Reinforcement Learning with Robust Cross-Entropy Method (RCE)](https://arxiv.org/abs/2010.07968)
- [x] **[NeurIPS 2018]** [Constrained Cross-Entropy Method for Safe Reinforcement Learning (CCE)](https://proceedings.neurips.cc/paper/2018/hash/34ffeb359a192eb8174b6854643cc046-Abstract.html)

<summary><b><big>Offline SafeRL</big></b></summary>

- [x] [The Lagrange version of BCQ (BCQ-Lag)](https://arxiv.org/abs/1812.02900)
- [x] [The Constrained version of CRR (C-CRR)](https://proceedings.neurips.cc/paper/2020/hash/588cb956d6bbe67078f29f8de420a13d-Abstract.html)
- [ ] **[AAAI 2022]** [Constraints Penalized Q-learning for Safe Offline Reinforcement Learning CPQ](https://arxiv.org/abs/2107.09003)
- [x] **[ICLR 2022 (Spotlight)]** [COptiDICE: Offline Constrained Reinforcement Learning via Stationary Distribution Correction Estimation](https://arxiv.org/abs/2204.08957?context=cs.AI)
- [ ] **[ICML 2022]** [Constrained Offline Policy Optimization (COPO)](https://proceedings.mlr.press/v162/polosky22a.html)

<summary><b><big>Others</big></b></summary>

- [ ] **[RA-L 2021]** [Recovery RL: Safe Reinforcement Learning with Learned Recovery Zones](https://arxiv.org/abs/2010.15920)
- [x] **[ICML 2022]** [Sauté RL: Almost Surely Safe Reinforcement Learning Using State Augmentation (SauteRL)](https://arxiv.org/abs/2202.06558)
- [x] **[NeurIPS 2022]** [Effects of Safety State Augmentation on Safe Exploration](https://arxiv.org/abs/2206.02675)

</details>

--------------------------------------------------------------------------------

### Examples

```bash
cd examples
python train_policy.py --algo PPOLag --env-id SafetyPointGoal1-v0 --parallel 1 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 1
```

#### Algorithms Registry

<table>
<thead>
  <tr>
    <th>Domains</th>
    <th>Types</th>
    <th>Algorithms Registry</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="5">On Policy</td>
    <td rowspan="2">Primal Dual</td>
    <td>TRPOLag; PPOLag; PDO; RCPO</td>
  </tr>
  <tr>
    <td>TRPOPID; CPPOPID</td>
  </tr>
  <tr>
    <td>Convex Optimization</td>
    <td><span style="font-weight:400;font-style:normal">CPO; PCPO; </span>FOCOPS; CUP</td>
  </tr>
  <tr>
    <td>Penalty Function</td>
    <td>IPO; P3O</td>
  </tr>
  <tr>
    <td>Primal</td>
    <td>OnCRPO</td>
  </tr>
  <tr>
    <td rowspan="2">Off Policy</td>
    <td rowspan="2">Primal-Dual</td>
    <td>DDPGLag; TD3Lag; SACLag</td>
  </tr>
  <tr>
    <td><span style="font-weight:400;font-style:normal">DDPGPID; TD3PID; SACPID</span></td>
  </tr>
  <tr>
    <td rowspan="2">Model-based</td>
    <td>Online Plan</td>
    <td>SafeLOOP; CCEPETS; RCEPETS</td>
  </tr>
  <tr>
    <td><span style="font-weight:400;font-style:normal">Pessimistic Estimate</span></td>
    <td>CAPPETS</td>
  </tr>
    <td rowspan="2">Offline</td>
    <td>Q-Learning Based</td>
    <td>BCQLag; C-CRR</td>
  </tr>
  <tr>
    <td>DICE Based</td>
    <td>COptDICE</td>
  </tr>
  <tr>
    <td rowspan="3">Other Formulation MDP</td>
    <td>ET-MDP</td>
    <td><span style="font-weight:400;font-style:normal">PPO</span>EarlyTerminated; TRPOEarlyTerminated</td>
  </tr>
  <tr>
    <td>SauteRL</td>
    <td>PPOSaute; TRPOSaute</td>
  </tr>
  <tr>
    <td>SimmerRL</td>
    <td><span style="font-weight:400;font-style:normal">PPOSimmerPID; TRPOSimmerPID</span></td>
  </tr>
</tbody>
</table>

#### Supported Environments

Here is a list of environments that [Safety-Gymnasium](https://www.safety-gymnasium.com) supports:

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
  <tr>
    <td rowspan="4">Safe Isaac Gym</td>
    <td>OverSafeFinger</td>
    <td rowspan="4">ShadowHand</td>
    <td rowspan="4">ShadowHandOverSafeFinger</td>
  </tr>
  <tr>
    <td>OverSafeJoint</td>
  </tr>
  <tr>
    <td>CatchOver2UnderarmSafeFinger</td>
  </tr>
  <tr>
    <td>CatchOver2UnderarmSafeJoint</td>
  </tr>
</tbody>
</table>

For more information about environments, please refer to [Safety-Gymnasium](https://www.safety-gymnasium.com).

#### Customizing your environment

We offer a flexible customized environment interface that allows users to achieve the following **without modifying the OmniSafe source code**:

- Use OmniSafe to train algorithms on customized environments.
- Create the the environment with specified personalized parameters.
- Complete the recording of environment-specific information in Logger.

We provide **step-by-step tutorials** on [Environment Customization From Scratch](https://colab.research.google.com/github/PKU-Alignment/omnisafe/blob/main/tutorials/English/3.Environment%20Customization%20from%20Scratch.ipynb) and [Environment Customization From Community](https://colab.research.google.com/github/PKU-Alignment/omnisafe/blob/main/tutorials/English/4.Environment%20Customization%20from%20Community.ipynb) to give you a detailed introduction on how to use this extraordinary feature of OmniSafe.

*Note: If you find trouble customizing your environment, please feel free to open an [issue](https://github.com/PKU-Alignment/omnisafe/issues) or [discussion](https://github.com/PKU-Alignment/omnisafe/discussions). [Pull requests](https://github.com/PKU-Alignment/omnisafe/pulls) are also welcomed if you're willing to contribute the implementation of your environments interface.*

#### Try with CLI

```bash
pip install omnisafe

omnisafe --help  # Ask for help

omnisafe benchmark --help  # The benchmark also can be replaced with 'eval', 'train', 'train-config'

# Quick benchmarking for your research, just specify:
# 1. exp_name
# 2. num_pool(how much processes are concurrent)
# 3. path of the config file (refer to omnisafe/examples/benchmarks for format)

# Here we provide an exampe in ./tests/saved_source.
# And you can set your benchmark_config.yaml by following it
omnisafe benchmark test_benchmark 2 ./tests/saved_source/benchmark_config.yaml

# Quick evaluating and rendering your trained policy, just specify:
# 1. path of algorithm which you trained
omnisafe eval ./tests/saved_source/PPO-{SafetyPointGoal1-v0} --num-episode 1

# Quick training some algorithms to validate your thoughts
# Note: use `key1:key2`, your can select key of hyperparameters which are recursively contained, and use `--custom-cfgs`, you can add custom cfgs via CLI
omnisafe train --algo PPO --total-steps 2048 --vector-env-nums 1 --custom-cfgs algo_cfgs:steps_per_epoch --custom-cfgs 1024

# Quick training some algorithms via a saved config file, the format is as same as default format
omnisafe train-config ./tests/saved_source/train_config.yaml
```

--------------------------------------------------------------------------------

## Getting Started

### Important Hints

We have provided benchmark results for various algorithms, including on-policy, off-policy, model-based, and offline approaches, along with parameter tuning analysis. Please refer to the following:

- [On-Policy](./benchmarks/on-policy/)
- [Off-Policy](./benchmarks/off-policy/)
- [Model-based](./benchmarks/model-based/)
- [Offline](./benchmarks/offline/)

### Quickstart: Colab on the Cloud

Explore OmniSafe easily and quickly through a series of Google Colab notebooks:

- [Getting Started](https://colab.research.google.com/github/PKU-Alignment/omnisafe/blob/main/tutorials/English/1.Getting_Started.ipynb) Introduce the basic usage of OmniSafe so that users can quickly hand it.
- [CLI Command](https://colab.research.google.com/github/PKU-Alignment/omnisafe/blob/main/tutorials/English/2.CLI_Command.ipynb) Introduce how to use the CLI tool of OmniSafe.

We take great pleasure in collaborating with our users to create tutorials in various languages.
Please refer to our list of currently supported languages.
If you are interested in translating the tutorial into a new language or improving an existing version, kindly submit a PR to us.

--------------------------------------------------------------------------------

## Changelog

See [CHANGELOG.md](https://github.com/PKU-Alignment/omnisafe/blob/main/CHANGELOG.md).


## Publications using OmniSafe

We have compiled a list of papers that use OmniSafe for algorithm implementation or experimentation. If you are willing to include your work in this list, or if you wish to have your implementation officially integrated into OmniSafe, please feel free to [contact us](https://github.com/PKU-Alignment/omnisafe/issues).

| Papers | Publisher|
|:---:|:---:|
| [Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation](https://arxiv.org/pdf/2405.20860) | NeurIPS 2024 |
| [Off-Policy Primal-Dual Safe Reinforcement Learning](https://openreview.net/pdf?id=vy42bYs1Wo) | ICLR 2024 |
| [Safe Offline Reinforcement Learning with Feasibility-Guided Diffusion Model](https://openreview.net/pdf?id=j5JvZCaDM0) | ICLR 2024 |
| [Iterative Reachability Estimation for Safe Reinforcement Learning](https://proceedings.neurips.cc/paper_files/paper/2023/file/dca63f2650fe9e88956c1b68440b8ee9-Paper-Conference.pdf) | NeurIPS 2023 |
| [Balance Reward and Safety Optimization for Safe Reinforcement Learning: A Perspective of Gradient Manipulation](https://ojs.aaai.org/index.php/AAAI/article/view/30102/31944) | AAAI 2024 |
| [Learning Safety Constraints From Demonstration Using One-Class Decision Trees](https://openreview.net/pdf?id=dxUi16pvub) | AAAI 2024 WorkShops |

## The OmniSafe Team

OmniSafe is mainly developed by the SafeRL research team directed by Prof. [Yaodong Yang](https://www.yangyaodong.com/).
Our SafeRL research team members include [Borong Zhang](https://github.com/muchvo), [Jiayi Zhou](https://github.com/Gaiejj), [JTao Dai](https://github.com/calico-1226), [Weidong Huang](https://github.com/hdadong), [Ruiyang Sun](https://github.com/rockmagma02), [Xuehai Pan](https://github.com/XuehaiPan) and [Jiaming Ji](https://github.com/zmsn-2077).
If you have any questions in the process of using OmniSafe, don't hesitate to ask your questions on [the GitHub issue page](https://github.com/PKU-Alignment/omnisafe/issues/new/choose), we will reply to you in 2-3 working days.

## License

OmniSafe is released under Apache License 2.0.
