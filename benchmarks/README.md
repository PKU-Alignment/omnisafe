# OmniSafe's Mujoco Velocity Benchmark on SafeRL algorithms

The OmniSafe Mujoco Velocity Benchmark assesses the efficacy of OmniSafe's SafeRL algorithms in six environments from the Safety-Gymnasium task suite. For each supported algorithm and environment, we offer default hyperparameters utilized during the benchmark, as well as scripts to replicate the results. Additionally, we provide performance comparisons and code-level details with other open-source implementations or classic papers. Our package includes graphs and raw data that can be used for research purposes, along with log details from training. Finally, we offer hints on fine-tuning the algorithm for optimal results.

## On-Policy
### Supported Algorithms

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

> **More details can be refer to [On Policy Experiment](https://github.com/OmniSafeAI/omnisafe/tree/main/benchmarks/on-policy/README.md).**

## Off-Policy
### Supported Algorithms
- [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/pdf/1509.02971.pdf)
- [Twin Delayed DDPG (TD3)](https://arxiv.org/pdf/1802.09477.pdf)
- [Soft Actor-Critic (SAC)](https://arxiv.org/pdf/1812.05905.pdf)

> **More details can be refer to [Off Policy Experiment](https://github.com/OmniSafeAI/omnisafe/tree/main/benchmarks/off-policy/README.md).**
