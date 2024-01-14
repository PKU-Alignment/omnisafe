# OmniSafe's Safety-Gymnasium Benchmark for SafeRL algorithms

The OmniSafe Safety-Gymnasium Benchmark evaluates the effectiveness of OmniSafe's SafeRL algorithms across multiple environments from the [Safety-Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium) task suite. We provide default hyperparameters and scripts for replicating the results for each supported algorithm and environment. Furthermore, we compare the performance and code-level details with other open-source implementations. Our package includes graphs, raw data, and training log details that can be used for research purposes. Additionally, we provide tips on fine-tuning the algorithm for optimal results. Overall, our benchmark offers a comprehensive evaluation of OmniSafe's SafeRL algorithms with detailed information and resources for further research.

## On-Policy
### Supported Algorithms

**First-Order**

- **[NIPS 1999]** [Policy Gradient (PG)](https://papers.nips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)
- **[Preprint 2017]** [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf)
- **[Preprint 2019]** [The Lagrange version of PPO (PPOLag)](https://cdn.openai.com/safexp-short.pdf)
- **[IJCAI 2022]** [Penalized Proximal Policy Optimization for Safe Reinforcement Learning (P3O)]( https://arxiv.org/pdf/2205.11814.pdf)
- **[NeurIPS 2020]** [First Order Constrained Optimization in Policy Space (FOCOPS)](https://arxiv.org/abs/2002.06506)
- **[NeurIPS 2022]**  [Constrained Update Projection Approach to Safe Policy Optimization (CUP)](https://arxiv.org/abs/2209.07089)

**Second-Order**

- **[NeurIPS 2001]** [A Natural Policy Gradient (NaturalPG))](https://proceedings.neurips.cc/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf)
- **[PMLR 2015]** [Trust Region Policy Optimization (TRPO)](https://arxiv.org/abs/1502.05477)
- **[Preprint 2019]**  [The Lagrange version of TRPO (TRPOLag)](https://cdn.openai.com/safexp-short.pdf)
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

> **More details can be refer to [On Policy Experiment](https://github.com/PKU-Alignment/omnisafe/tree/main/benchmarks/on-policy/README.md).**

## Off-Policy

### Supported Algorithms

- **[ICLR 2016]** [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/pdf/1509.02971.pdf)
- **[ICML 2018]** [Twin Delayed DDPG (TD3)](https://arxiv.org/pdf/1802.09477.pdf)
- **[ICML 2018]** [Soft Actor-Critic (SAC)](https://arxiv.org/pdf/1812.05905.pdf)
- **[Preprint 2019]<sup>[[1]](#footnote1)</sup>** [The Lagrangian version of DDPG (DDPGLag)](https://cdn.openai.com/safexp-short.pdf)
- **[Preprint 2019]<sup>[[1]](#footnote1)</sup>** [The Lagrangian version of TD3 (TD3Lag)](https://cdn.openai.com/safexp-short.pdf)
- **[Preprint 2019]<sup>[[1]](#footnote1)</sup>** [The Lagrangian version of SAC (SACLag)](https://cdn.openai.com/safexp-short.pdf)
- **[ICML 2020]** [Responsive Safety in Reinforcement Learning by PID Lagrangian Methods (DDPGPID)](https://arxiv.org/abs/2007.03964)
- **[ICML 2020]** [Responsive Safety in Reinforcement Learning by PID Lagrangian Methods (TD3PID)](https://arxiv.org/abs/2007.03964)
- **[ICML 2020]** [Responsive Safety in Reinforcement Learning by PID Lagrangian Methods (SACPID)](https://arxiv.org/abs/2007.03964)

> **More details can be refer to [Off Policy Experiment](https://github.com/PKU-Alignment/omnisafe/tree/main/benchmarks/off-policy/README.md).**

## Model-based

- **[NeurIPS 2001]** [Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models (PETS))](https://arxiv.org/abs/1805.12114)
- **[CoRL 2021]** [Learning Off-Policy with Online Planning (LOOP and SafeLOOP)](https://arxiv.org/abs/2008.10066)
- **[AAAI 2022]** [Conservative and Adaptive Penalty for Model-Based Safe Reinforcement Learning (CAP)](https://arxiv.org/abs/2112.07701)
- **[ICML 2022 Workshop]** [Constrained Model-based Reinforcement Learning with Robust Cross-Entropy Method (RCE)](https://arxiv.org/abs/2010.07968)
- **[NeurIPS 2018]** [Constrained Cross-Entropy Method for Safe Reinforcement Learning (CCE)](https://proceedings.neurips.cc/paper/2018/hash/34ffeb359a192eb8174b6854643cc046-Abstract.html)

> **More details can be refer to [Model-based Experiment](https://github.com/PKU-Alignment/omnisafe/tree/main/benchmarks/model-based/README.md).**

## Offline

- **[ICML 2019]** [Batch-Constrained deep Q-learning(BCQ)](https://arxiv.org/pdf/1812.02900.pdf)
- [The Lagrange version of BCQ (BCQ-Lag)](https://arxiv.org/pdf/1812.02900.pdf)
- **[NeurIPS 2020]** [Critic Regularized Regression](https://proceedings.neurips.cc//paper/2020/file/588cb956d6bbe67078f29f8de420a13d-Paper.pdf)
- [The Constrained version of CRR (C-CRR)](https://proceedings.neurips.cc/paper/2020/hash/588cb956d6bbe67078f29f8de420a13d-Abstract.html)
- **[ICLR 2022 (Spotlight)]** [COptiDICE: Offline Constrained Reinforcement Learning via Stationary Distribution Correction Estimation](https://arxiv.org/abs/2204.08957?context=cs.AI)

> **More details can be refer to [Offline Experiment](https://github.com/PKU-Alignment/omnisafe/tree/main/benchmarks/offline/README.md).**

<a name="footnote1">[1]</a>  This paper is [safety-gym](https://openai.com/research/safety-gym) original paper. Its public code base [safety-starter-agents](https://github.com/openai/safety-starter-agents) implemented `SACLag` but does not report it in the paper.  We can not find the source of `DDPGLag` and `TD3Lag`. However, this paper introduced lagrangian methods and it implemented `SACLag`, so we also use it as a source of `DDPGLag` and `TD3Lag`.
