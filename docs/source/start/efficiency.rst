Efficiency
==========

To demonstrate the effectiveness and resource utilization of OmniSafe as a SafeRL infrastructure, we have added a comparison of the runtime efficiency between OmniSafe and other SafeRL libraries, *i.e.*, `SafePO <https://proceedings.neurips.cc/paper_files/paper/2023/file/3c557a3d6a48cc99444f85e924c66753-Paper-Datasets_and_Benchmarks.pdf>`_, `RL-Safety-Algorithms <https://github.com/SvenGronauer/RL-Safety-Algorithms>`_, and `Safety-starter-agents <https://github.com/openai/safety-starter-agents>`_. The test results are shown in Table 1:



.. table:: **Table 1**: Comparison of computational time consumption between OmniSafe and other libraries in one thread (unit: seconds). We selected classic algorithms PPOLag and CPO for analysis and tested the average single epoch time consumption over 10 epochs with different sizes of neural networks on SG's SafetyPointGoal1-v0.
   :name: appendix_f
   :width: 100 %

   +------------------------------------+----------------------------+------------------+------------------+------------------+
   |                                    | **PPOLag**                 |                  | **CPO**          |                  |
   +------------------------------------+----------------------------+------------------+------------------+------------------+
   |**Hidden Layers Size**              | 64 x 64                    | 1024 x 1024      | 64 x 64          | 1024 x 1024      |
   +------------------------------------+----------------------------+------------------+------------------+------------------+
   |**Safety-starter-agents**           | 51.64 ± 1.56               | 63.99 ± 1.75     | 50.70 ± 1.17     | 83.09 ± 0.92     |
   +------------------------------------+----------------------------+------------------+------------------+------------------+
   | **RL-Safety-Algorithms**           | 46.25 ± 0.43               | 107.50 ± 2.18    | 47.24 ± 0.43     | 134.12 ± 0.71    |
   +------------------------------------+----------------------------+------------------+------------------+------------------+
   | **SafePO**                         | 15.91 ± 0.46               | 20.84 ± 0.26     | 16.50 ± 0.50     | 19.72 ± 0.16     |
   +------------------------------------+----------------------------+------------------+------------------+------------------+
   | **OmniSafe**                       | **10.59 ± 0.15**           | **14.02 ± 0.16** | **10.06 ± 0.09** | **12.28 ± 0.81** |
   +------------------------------------+----------------------------+------------------+------------------+------------------+


In our comparative experiments, we rigorously ensure uniformity across all experimental settings. More specifically, PPOLag and CPO implement early stopping techniques, which vary the number of
updates based on the KL divergence between the current and reference policies. This introduces
randomness into the time measurements. To control for consistent variables, we fixed the number of
``update_iters`` at 1, ``steps_per_epoch`` at 20,000, and ``batch_size`` at 64, conducting the tests on the same machine with no other processes running. The specific device parameters are:

- **CPU**: AMD Ryzen Threadripper PRO 3975WX 32-Cores
- **GPU**: NVIDIA GeForce RTX 3090, Driver Version: 535.154.05

Under these consistent conditions, **OmniSafe achieved the lowest computational time consumption on
the same baseline algorithms**, which we attribute to 3 factors: *vectorized environment
parallelism* for accelerated data collections, `asynchronous agent parallelism <https://arxiv.org/abs/1602.01783>`_ for parallelized learning, and *GPU resource utilization* for immense network
support. We will elaborate on how these features contribute to OmniSafe's computational efficiency.

**Vectorized Environment Parallelism**: OmniSafe and SafePO support vectorized environment
interfaces and buffers. In this experiment, we set the parallelism number of vectorized
environments to 10, meaning that a single agent can simultaneously generate 10 actions based on 10
vectorized observations and perform batch updates through a vectorized buffer. This feature
enhances the efficiency of agents' data sampling from environments.

**Asynchronous Agent Parallelism**: OmniSafe supports *Asynchronous Advantage Actor-Critic (A3C)*
parallelism based on the distributed framework ``torch.distributed``. In this experiment, we set
the parallelism number of asynchronous agents to 2, meaning two agents were instantiated to sample
and learn simultaneously, synchronizing their neural network parameters at the end of each epoch.
This feature further enhances the efficiency of agent sampling and updating.

**GPU Resource Utilization**: Since only OmniSafe and SafePO utilize GPU computing resources, in
this experiment, we used the NVIDIA GeForce RTX 3090 as the computing device. As shown in
:ref:`Table 1 <appendix_f>`., when the hidden layer parameters increased from 64 x 64 to 1024 x
1024, the runtime of RL-Safety-Algorithms and Safety-starter-agents significantly increased,
whereas the runtime increase for OmniSafe and SafePO was relatively smaller. This trend is
particularly notable with the CPO algorithm, which requires computing a second-order Hessian matrix
during updates. If computed using a CPU, the computational overhead would increase with the size of
the neural network parameters. However, OmniSafe and SafePO, which support GPU acceleration, are
almost unaffected.
