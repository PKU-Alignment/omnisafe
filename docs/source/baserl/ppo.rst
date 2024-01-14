Proximal Policy Optimization
============================

Quick Facts
-----------

.. card::
    :class-card: sd-outline-info  sd-rounded-1
    :class-body: sd-font-weight-bold

    #. PPO is an :bdg-info-line:`on-policy` algorithm.
    #. PPO can be thought of as being a simple implementation of :bdg-ref-info-line:`TRPO <trpo>`  .
    #. The OmniSafe implementation of PPO support :bdg-info-line:`parallelization`.
    #. An :bdg-ref-info-line:`API Documentation <ppoapi>` is available for PPO.

PPO Theorem
-----------

Background
~~~~~~~~~~

**Proximal Policy Optimization(PPO)** is a reinforcement learning algorithm
inheriting some of the
benefits of :doc:`TRPO<trpo>`.
However, it is much simpler to implement.
PPO shares the same goal as TRPO:

.. note::
    Take the largest possible improvement step on a policy update
    using the available data, without stepping too far and causing performance
    collapse.

However, instead of using a complex second-order method like TRPO, PPO uses a
few tricks to keep the new policies close to the old ones. There are two
primary variants of PPO:
:bdg-ref-info-line:`PPO-Penalty<PPO-Penalty>` and
:bdg-ref-info-line:`PPO-Clip<PPO-Clip>`.

.. grid:: 2

    .. grid-item::
        :columns: 12 6 6 5

        .. card::
            :class-header: sd-bg-warning sd-text-white sd-font-weight-bold
            :class-card: sd-outline-warning  sd-rounded-1 sd-font-weight-bold

            Problems of TRPO
            ^^^
            - The calculation of KL divergence in TRPO is too complicated.

            - Only the raw data sampled by the Monte Carlo method is used.

            - Using second-order optimization methods.

    .. grid-item::
        :columns: 12 6 6 6

        .. card::
            :class-header: sd-bg-primary sd-text-white sd-font-weight-bold
            :class-card: sd-outline-primary  sd-rounded-1 sd-font-weight-bold

            Advantage of PPO
            ^^^
            - Using ``clip`` method to make the difference between the two strategies less significant.

            - Using the :math:`\text{GAE}` method to process advantage function.

            - Simple to implement.

            - Using first-order optimization methods.

------

Optimization Objective
~~~~~~~~~~~~~~~~~~~~~~

In the previous chapters, we introduced that TRPO solves the following
optimization problems:

.. _ppo-eq-1:

.. math::
    :label: ppo-eq-1

    & \pi_{k+1}=\arg\max_{\pi \in \Pi_{\boldsymbol{\theta}}}J^R(\pi)\\
    \text{s.t.}\quad & D(\pi,\pi_k)\le\delta


where :math:`\Pi_{\boldsymbol{\theta}} \subseteq \Pi` denotes the set of
parameterized policies with parameters :math:`\boldsymbol{\theta}`, and
:math:`D` is some distance measure.

TRPO tackles the challenge of determining the appropriate direction and step
size for policy updates, aiming to improve performance while minimizing
deviations from the original policy. To achieve this, TRPO reformulates
Problem :eq:`ppo-eq-1` as:

.. _ppo-eq-2:

.. math::
    :label: ppo-eq-2

    \underset{\theta}{\max} \quad & L_{\theta_{old}}(\theta)  \\
    \text{s.t. } \quad & \bar{D}_{\mathrm{KL}}(\theta_{old}, \theta) \le \delta


where
:math:`L_{\theta_{old}}(\theta)= \frac{\pi_\theta(a \mid s)}{\pi_{\theta_{old}}(a \mid s)} \hat{A}^{R}_\pi(s, a)`,
moreover, :math:`\hat{A}^{R}_{\pi}(s, a)` is an estimator of the advantage
function
given :math:`s` and  :math:`a`.

You may still have a question: Why are we using :math:`\hat{A}` instead of
:math:`A`.
This is a trick named **generalized advantage estimator** (:math:`\text{GAE}`).
Almost all advanced reinforcement learning algorithms use :math:`\text{GAE}`
technique to estimate more efficient advantage :math:`A`.
:math:`\hat{A}` is the :math:`\text{GAE}` version of :math:`A`.

------

.. _PPO-Penalty:

PPO-Penalty
~~~~~~~~~~~

TRPO has advocated using a penalty method to transform constrained problems
into unconstrained ones for solving:

.. _ppo-eq-3:

.. math::
    :label: ppo-eq-3

    \max _\theta \mathbb{E}[\frac{\pi_\theta(a \mid s)}{\pi_{\theta_{old}}(a \mid s)} \hat{A}_\pi(s, a)-\beta D_{K L}[\pi_{\theta_{old}}(* \mid s), \pi_\theta(* \mid s)]]

However, experiments have shown that simply choosing a fixed penalty
coefficient :math:`\beta` and optimizing the penalized objective :eq:`ppo-eq-3`
with SGD (stochastic gradient descent) is not sufficient. Therefore, TRPO
abandoned this method.

PPO-Penalty uses an approach called ``Adaptive KL Penalty Coefficient`` to
address this problem and improve the performance of :eq:`ppo-eq-3` in
experiments. In the simplest implementation of this algorithm, PPO-Penalty
performs the following steps in each policy update iteration:

.. grid:: 2

    .. grid-item::
        :columns: 12 6 6 7

        .. card::
            :class-header: sd-bg-info sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-1 sd-font-weight-bold

            Step I
            ^^^
            Using several epochs of mini-batch SGD, optimize the KL-penalized objective shown as :eq:`ppo-eq-3`,

            .. math::
                :label: ppo-eq-4

                L^{\mathrm{KLPEN}}(\theta)&=\hat{\mathbb{E}}[\frac{\pi_\theta(a \mid s)}{\pi_{\theta_{old}}(a \mid s)} \hat{A}_\pi(s, a)\\
                &-\beta D_{K L}[\pi_{\theta_{old}}(* \mid s), \pi_\theta(* \mid s)]]



    .. grid-item::
        :columns: 12 6 6 5

        .. card::
            :class-header: sd-bg-info sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-1 sd-font-weight-bold

            Step II
            ^^^
            Compute :math:`d=\hat{\mathbb{E}}[\mathrm{KL}[\pi_{\theta_{\text {old }}}(\cdot \mid s), \pi_\theta(\cdot \mid s)]]`

            If :math:`d<d_{\text {targ }} / 1.5, \beta \leftarrow \beta / 2`

            If :math:`d>d_{\text {targ }} \times 1.5, \beta \leftarrow \beta * 2`

            The updated :math:`\beta` is used for the next policy update.

------

.. _PPO-Clip:

PPO-Clip
~~~~~~~~

Let :math:`r(\theta)` denote the probability ratio
:math:`r(\theta)=\frac{\pi_\theta(a \mid s)}{\pi_{\theta_{old}}(a \mid s)}`,
PPO-Clip rewrites the surrogate objective as:

.. _ppo-eq-5:

.. math::
    :label: ppo-eq-5

    L^{\mathrm{CLIP}}(\pi)=\mathbb{E}[\text{min} (r(\theta) \hat{A}_{\pi}(s, a), \text{clip}(r(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_{\pi}(s, a))]

The hyperparameter :math:`\varepsilon` represents a small value that
approximately indicates the allowable distance between the new and the
old policy. The formula involved in this context is quite intricate, making it
challenging to comprehend its purpose or how it contributes to maintaining the
proximity between the new and old policies. To facilitate a clearer
understanding of the aforementioned expression,

let :math:`L(s, a, \theta)` denote
:math:`\max [r(\theta) \hat{A}_{\pi}(s, a), \text{clip}(r(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_{\pi}(s, a)]`,
we'll simplify the formula in two cases:

.. card::
    :class-header: sd-bg-info sd-text-white sd-font-weight-bold
    :class-card: sd-outline-info  sd-rounded-1 sd-font-weight-bold
    :class-footer: sd-font-weight-bold

    PPO Clip
    ^^^

    #. When Advantage is positive, we can rewrite :math:`L(s, a, \theta)` as:

       .. math::
        :label: ppo-eq-6

        L(s, a, \theta)=\max (r(\theta),(1-\varepsilon)) \hat{A}_{\pi}(s, a)

    #. When Advantage is negative, we can rewrite :math:`L(s, a, \theta)` as:

       .. math::
        :label: ppo-eq-7

        L(s, a, \theta)=\max (r(\theta),(1+\varepsilon)) \hat{A}_{\pi}(s, a)

With the above clipped surrogate function and :eq:`ppo-eq-5`,
PPO-Clip can guarantee the new policy
would not update so far away from the old.
In the experiment, PPO-Clip performs better than PPO-Penalty.

------

Practical Implementation
------------------------

Generalized Advantage Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One style of policy gradient implementation, popularized in and well-suited for
use with recurrent neural networks, runs the policy for :math:`T`
timesteps (where :math:`T` is much less than the episode length), and uses the
collected samples for an update. This style requires an advantage estimator
that does not look beyond timestep :math:`T`. This section will focus on
producing an accurate estimate of the advantage function
:math:`\hat{A}_{\pi}(s,a)` (Equals to :math:`\hat{A}^{R}_{\pi}(s,a)` since only reward is considered here, same as the following.) using only information
from the current trajectory up to timestep :math:`T`.

Define :math:`\delta^V=r_t+\gamma V(s_{t+1})-V(s)` as the TD residual of
:math:`V` with discount :math:`\gamma`.
Next, let us consider taking the sum of :math:`k` of these :math:`\delta`
terms, which we will denote by :math:`\hat{A}_{\pi}^{(k)}`.

.. math::
    :label: ppo-eq-8

    \begin{array}{ll}
    \hat{A}_{\pi}^{(1)}:=\delta_t^V =-V(s_t)+r_t+\gamma V(s_{t+1}) \\
    \hat{A}_{\pi}^{(2)}:=\delta_t^V+\gamma \delta_{t+1}^V =-V(s_t)+r_t+\gamma r_{t+1}+\gamma^2 V(s_{t+2}) \\
    \hat{A}_{\pi}^{(3)}:=\delta_t^V+\gamma \delta_{t+1}^V+\gamma^2 \delta_{t+2}^V =-V(s_t)+r_t+\gamma r_{t+1}+\gamma^2 r_{t+2}+\gamma^3 V(s_{t+3}) \\
    \hat{A}_{\pi}^{(k)}:=\sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V =-V(s_t)+r_t+\gamma r_{t+1}+\cdots+\gamma^{k-1} r_{t+k-1}+\gamma^k V(s_{t+k})
    \end{array}

We can consider :math:`\hat{A}_{\pi}^{(k)}` to be an estimator of the advantage
function.

.. hint::
    The bias generally becomes smaller as :math:`k \rightarrow +\infty`,
    since the term :math:`\gamma^k V(s_{t+k})` becomes more heavily discounted.
    Taking :math:`k \rightarrow +\infty`, we get:

    .. math::
        :label: ppo-eq-9

        \hat{A}_{\pi}^{(\infty)}=\sum_{l=0}^{\infty} \gamma^l \delta_{t+l}^V=-V(s_t)+\sum_{l=0}^{\infty} \gamma^l r_{t+l}


    which is simply the empirical returns minus the value function baseline.

The generalized advantage estimator :math:`\text{GAE}(\gamma,\lambda)` is
defined as the exponentially-weighted average of these :math:`k`-step
estimators:

.. _ppo-eq-6:

.. math::
    :label: ppo-eq-10

    \hat{A}_{\pi}:&= (1-\lambda)(\hat{A}_{\pi}^{(1)}+\lambda \hat{A}_{\pi}^{(2)}+\lambda^2 \hat{A}_{\pi}^{(3)}+\ldots) \\
    &= (1-\lambda)(\delta_t^V+\lambda(\delta_t^V+\gamma \delta_{t+1}^V)+\lambda^2(\delta_t^V+\gamma \delta_{t+1}^V+\gamma^2 \delta_{t+2}^V)+\ldots) \\
    &= (1-\lambda)(\delta_t^V(1+\lambda+\lambda^2+\ldots)+\gamma \delta_{t+1}^V(\lambda+\lambda^2+\lambda^3+\ldots) .+\gamma^2 \delta_{t+2}^V(\lambda^2+\lambda^3+\lambda^4+\ldots)+\ldots) \\
    &= (1-\lambda)(\delta_t^V(\frac{1}{1-\lambda})+\gamma \delta_{t+1}^V(\frac{\lambda}{1-\lambda})+\gamma^2 \delta_{t+2}^V(\frac{\lambda^2}{1-\lambda})+\ldots) \\
    &= \sum_{l=0}^{\infty}(\gamma \lambda)^l \delta_{t+l}^V


There are two notable special cases of this formula, obtained by setting
:math:`\lambda =0` and :math:`\lambda =1`.

.. math::
    :label: ppo-eq-11

    \text{GAE}(\gamma, 0):\quad & \hat{A}_{\pi}:=\delta_t  =r_t+\gamma V(s_{t+1})-V(s_t) \\
    \text{GAE}(\gamma, 1):\quad & \hat{A}_{\pi}:=\sum_{l=0}^{\infty} \gamma^l \delta_{t+l}  =\sum_{l=0}^{\infty} \gamma^l r_{t+l}-V(s_t)


.. hint::
    :math:`\text{GAE}(\gamma,1)` is the traditional MC-based method to estimate the advantage function,
    but it has a high variance due to the sum of terms.
    :math:`\text{GAE}(\gamma,0)` is TD-based method with low variance,
    but it suffers from bias.

The generalized advantage estimator for :math:`0\le\lambda\le1` makes a
compromise between bias and variance,
controlled by parameter :math:`\lambda`.

Code with OmniSafe
~~~~~~~~~~~~~~~~~~

Quick start
"""""""""""

.. card::
    :class-header: sd-bg-success sd-text-white sd-font-weight-bold
    :class-card: sd-outline-success  sd-rounded-1 sd-font-weight-bold
    :class-footer: sd-font-weight-bold

    Run PPO in OmniSafe
    ^^^^^^^^^^^^^^^^^^^
    Here are 3 ways to run PPO in OmniSafe:

    * Run Agent from preset yaml file
    * Run Agent from custom config dict
    * Run Agent from custom terminal config

    .. tab-set::

        .. tab-item:: Yaml file style

            .. code-block:: python
                :linenos:

                import omnisafe


                env_id = 'SafetyPointGoal1-v0'

                agent = omnisafe.Agent('PPO', env_id)
                agent.learn()

        .. tab-item:: Config dict style

            .. code-block:: python
                :linenos:

                import omnisafe


                env_id = 'SafetyPointGoal1-v0'
                custom_cfgs = {
                    'train_cfgs': {
                        'total_steps': 10000000,
                        'vector_env_nums': 1,
                        'parallel': 1,
                    },
                    'algo_cfgs': {
                        'steps_per_epoch': 20000,
                    },
                    'logger_cfgs': {
                        'use_wandb': False,
                        'use_tensorboard': True,
                    },
                }

                agent = omnisafe.Agent('PPO', env_id, custom_cfgs=custom_cfgs)
                agent.learn()


        .. tab-item:: Terminal config style

            We use ``train_policy.py`` as the entrance file. You can train the agent with PPO simply using ``train_policy.py``, with arguments about PPO and environments does the training.
            For example, to run PPO in SafetyPointGoal1-v0 , with 1 torch thread, seed 0 and single environment, you can use the following command:

            .. code-block:: bash
                :linenos:

                cd examples
                python train_policy.py --algo PPO --env-id SafetyPointGoal1-v0 --parallel 1 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 1

------

Here is the documentation of PPO in PyTorch version.


Architecture of functions
"""""""""""""""""""""""""

-  ``PPO.learn()``

   - ``PPO._env.rollout()``
   - ``PPO._update()``

     - ``PPO._buf.get()``
     - ``PPO.update_lagrange_multiplier(ep_costs)``
     - ``PPO._update_actor()``
     - ``PPO._update_reward_critic()``

------

Documentation of algorithm specific functions
"""""""""""""""""""""""""""""""""""""""""""""

.. tab-set::

    .. tab-item:: ppo._loss_pi()

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success  sd-rounded-1 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            ppo._loss_pi()
            ^^^
            Compute the loss of ``actor``, flowing the next steps:

            (1) Get the policy importance sampling ratio.

            .. code-block:: python
                :linenos:

                distribution = self._actor_critic.actor(obs)
                logp_ = self._actor_critic.actor.log_prob(act)
                std = self._actor_critic.actor.std
                ratio = torch.exp(logp_ - logp)


            (2) Get the clipped surrogate function.

            .. code-block:: python
                :linenos:

                ratio_cliped = torch.clamp(
                    ratio, 1 - self._cfgs.algo_cfgs.clip, 1 + self._cfgs.algo_cfgs.clip
                )
                loss = -torch.min(ratio * adv, ratio_cliped * adv).mean()
                loss -= self._cfgs.algo_cfgs.entropy_coef * distribution.entropy().mean()

            (3) Return the loss of ``actor``.

------

Configs
""""""""""

.. tab-set::

    .. tab-item:: Train

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success  sd-rounded-1 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            Train Configs
            ^^^

            - device (str): Device to use for training, options: ``cpu``, ``cuda``, ``cuda:0``, etc.
            - torch_threads (int): Number of threads to use for PyTorch.
            - total_steps (int): Total number of steps to train the agent.
            - parallel (int): Number of parallel agents, similar to A3C.
            - vector_env_nums (int): Number of the vector environments.

    .. tab-item:: Algorithm

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success  sd-rounded-1 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            Algorithms Configs
            ^^^

            .. note::

                The following configs are specific to PPO algorithm.

                - clip (float): Clipping parameter for PPO.

            - steps_per_epoch (int): Number of steps to update the policy network.
            - update_iters (int): Number of iterations to update the policy network.
            - batch_size (int): Batch size for each iteration.
            - target_kl (float): Target KL divergence.
            - entropy_coef (float): Coefficient of entropy.
            - reward_normalize (bool): Whether to normalize the reward.
            - cost_normalize (bool): Whether to normalize the cost.
            - obs_normalize (bool): Whether to normalize the observation.
            - kl_early_stop (bool): Whether to stop the training when KL divergence is too large.
            - max_grad_norm (float): Maximum gradient norm.
            - use_max_grad_norm (bool): Whether to use maximum gradient norm.
            - use_critic_norm (bool): Whether to use critic norm.
            - critic_norm_coef (float): Coefficient of critic norm.
            - gamma (float): Discount factor.
            - cost_gamma (float): Cost discount factor.
            - lam (float): Lambda for GAE-Lambda.
            - lam_c (float): Lambda for cost GAE-Lambda.
            - adv_estimation_method (str): The method to estimate the advantage.
            - standardized_rew_adv (bool): Whether to use standardized reward advantage.
            - standardized_cost_adv (bool): Whether to use standardized cost advantage.
            - penalty_coef (float): Penalty coefficient for cost.
            - use_cost (bool): Whether to use cost.


    .. tab-item:: Model

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success  sd-rounded-1 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            Model Configs
            ^^^

            - weight_initialization_mode (str): The type of weight initialization method.
            - actor_type (str): The type of actor, default to ``gaussian_learning``.
            - linear_lr_decay (bool): Whether to use linear learning rate decay.
            - exploration_noise_anneal (bool): Whether to use exploration noise anneal.
            - std_range (list): The range of standard deviation.

            .. hint::

                actor (dictionary): parameters for actor network ``actor``

                - activations: tanh
                - hidden_sizes:
                - 64
                - 64

            .. hint::

                critic (dictionary): parameters for critic network ``critic``

                - activations: tanh
                - hidden_sizes:
                - 64
                - 64

    .. tab-item:: Logger

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success  sd-rounded-1 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            Logger Configs
            ^^^

            - use_wandb (bool): Whether to use wandb to log the training process.
            - wandb_project (str): The name of wandb project.
            - use_tensorboard (bool): Whether to use tensorboard to log the training process.
            - log_dir (str): The directory to save the log files.
            - window_lens (int): The length of the window to calculate the average reward.
            - save_model_freq (int): The frequency to save the model.

------

References
----------

- `Trust Region Policy Optimization <https://arxiv.org/abs/1502.05477>`__
- `Proximal Policy Optimization Algorithms <https://arxiv.org/pdf/1707.06347.pdf>`__
