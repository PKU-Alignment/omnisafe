Proximal Policy Optimization Algorithms
=======================================

Quick Facts
-----------

.. card::
    :class-card: sd-outline-info  sd-rounded-1
    :class-body: sd-font-weight-bold

    #. PPO is an :bdg-info-line:`on-policy` algorithm.
    #. PPO can be used for environments with both :bdg-info-line:`discrete` and :bdg-info-line:`continuous` action spaces.
    #. PPO can be thought of as being a simple implementation of :bdg-info-line:`TRPO` .
    #. The OmniSafe implementation of PPO support :bdg-info-line:`parallelization`.

PPO Theorem
-----------

Background
~~~~~~~~~~

**Proximal Policy Optimization(PPO)** is a RL algorithm inheriting some of the benefits of :doc:`trpo<trpo>`,
but are much simpler to implement.
PPO share the same target with TRPO:
how can we take a as big as improvement step on a policy update using the data we already have,
without stepping so far that we accidentally cause performance collapse?
Instead of solving this problem with a complex second-order method as TRPO do,
PPO use a few other tricks to keep new policies close to old.
There are two primary variants of PPO: :bdg-ref-info-line:`PPO-Penalty<PPO-Penalty>` and :bdg-ref-info-line:`PPO-Clip<PPO-Clip>`.

.. grid:: 2

    .. grid-item::
        :columns: 12 6 6 6

        .. card::
            :class-header: sd-bg-warning sd-text-white sd-font-weight-bold
            :class-card: sd-outline-warning  sd-rounded-1 sd-font-weight-bold

            Problems of TRPO
            ^^^
            -  The calculation of KL divergence in TRPO is too complicated.

            -  Only the raw data sampled by the Monte Carlo method is used.

            -  Using second-order optimization methods.

    .. grid-item::
        :columns: 12 6 6 6

        .. card::
            :class-header: sd-bg-primary sd-text-white sd-font-weight-bold
            :class-card: sd-outline-primary  sd-rounded-1 sd-font-weight-bold

            Advantage of PPO
            ^^^
            -  Using ``clip`` method to makes the difference between the two strategies less significant.

            -  Using the :math:`\text{GAE}` method to process data.

            -  Simple to implement.

            -  Using first-order optimization methods.

------

Optimization Objective
~~~~~~~~~~~~~~~~~~~~~~

In the previous chapters, we introduced that TRPO solves the following optimization problems:

.. _ppo-eq-1:

.. math::
    :label: ppo-eq-1

    &\pi_{k+1}=\arg\max_{\pi \in \Pi_{\boldsymbol{\theta}}}J^R(\pi)\\
    \text{s.t.}\quad&D(\pi,\pi_k)\le\delta


where :math:`\Pi_{\boldsymbol{\theta}} \subseteq \Pi` denotes the set of parameterized policies with parameters :math:`\boldsymbol{\theta}`, and :math:`D` is some distance measure.
The problem that TRPO needs to solve is how to find a suitable update direction and update step,
so that updating the actor can improve the performance without being too different from the original actor.
Finally, TRPO rewrites Problem :eq:`ppo-eq-1` as:

.. _ppo-eq-2:

.. math::
    :label: ppo-eq-2

    &\underset{\theta}{\max} L_{\theta_{old}}(\theta)  \\
    &\text{s.t. } \quad \bar{D}_{\mathrm{KL}}(\theta_{old}, \theta) \le \delta


where :math:`L_{\theta_{old}}(\theta)= \frac{\pi_\theta(a \mid s)}{\pi_{\theta_{old}}(a \mid s)} \hat{A}_\pi(s, a)`,
and :math:`\hat{A}_{\pi}(s,a)` is an estimator of the advantage function given :math:`s` and  :math:`a`.

You may still have a question: Why are we using :math:`\hat{A}` instead of :math:`A`.
Actually this is a trick named **generalized advantage estimator** (:math:`\text{GAE}`).
Almost all advanced reinforcement learning algorithms use :math:`\text{GAE}` technique to make more efficient estimates of :math:`A`.
:math:`\hat{A}` is the :math:`\text{GAE}` version of :math:`A`.

------

.. _PPO-Penalty:

PPO-Penalty
~~~~~~~~~~~

TRPO actually suggests using a penalty instead of a constraint to solve the unconstrained optimization problem:

.. _ppo-eq-3:

.. math::
    :label: ppo-eq-3

    \max _\theta \mathbb{E}[\frac{\pi_\theta(a \mid s)}{\pi_{\theta_{old}}(a \mid s)} \hat{A}_\pi(s, a)-\beta D_{K L}[\pi_{\theta_{old}}(* \mid s), \pi_\theta(* \mid s)]]


However, experiments show that it is not sufficient to simply choose a fixed penalty coefficient :math:`\beta` and optimize the penalized objective Equation :eq:`ppo-eq-3` with SGD(stochastic gradient descent),
so finally TRPO abandoned this method.

PPO-Penalty use an approach named Adaptive KL Penalty Coefficient to solve above problem,
thus making :eq:`ppo-eq-3` perform well in experiment.
In the simplest implementation of this algorithm,
PPO-Penalty perform the following steps in each policy update:


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

                L^{\mathrm{KLPEN}}(\theta)&=&\hat{\mathbb{E}}[\frac{\pi_\theta(a \mid s)}{\pi_{\theta_{old}}(a \mid s)} \hat{A}_\pi(s, a)\\
                &-&\beta D_{K L}[\pi_{\theta_{old}}(* \mid s), \pi_\theta(* \mid s)]]



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

Let :math:`r(\theta)` denote the probability ratio :math:`r(\theta)=\frac{\pi_\theta(a \mid s)}{\pi \theta_{d d}(a \mid s)}`,
PPO-Clip rewrite the surrogate objective as:

.. _ppo-eq-5:

.. math::
    :label: ppo-eq-5

    L^{\mathrm{CLIP}}(\pi)=\mathbb{E}[\text{min} (r(\theta) \hat{A}_{\pi}(s, a), \text{clip}(r(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_{\pi}(s, a))]


in which :math:`\varepsilon` is a (small) hyperparameter which roughly says how far away the new policy is allowed to go from the old.
This is a very complex formula,
and it's difficult to tell at first glance what it's doing,
or how it helps keep the new policy close to the old policy.
To help you better understand the above expression,
let :math:`L(s, a, \theta)` denote :math:`\max [r(\theta) \hat{A}_{\pi}(s, a), \text{clip}(r(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_{\pi}(s, a)]`,
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

With above clipped surrogate function and :eq:`ppo-eq-5`,
PPO-Clip can guarantee the new policy would not update so far away from the old.
In experiment, PPO-Clip perform better that PPO-Penalty.

------

Practical Implementation
------------------------

Generalized Advantage Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One style of policy gradient implementation, popularized in and well-suited for use with recurrent neural networks,
runs the policy for :math:`T` timesteps (where :math:`T` is much less than the episode length), and uses the collected samples for an update.
This style requires an advantage estimator that does not look beyond timestep :math:`T`.
This section will be concerned with producing an accurate estimate :math:`\hat{A}_{\pi}(s,a)`.

Define :math:`\delta^V=r_t+\gamma V(s_{t+1})-V(s)` as the TD residual of :math:`V` with discount :math:`\gamma`.
Next, let us consider taking the sum of :math:`k` of these :math:`\delta` terms, which we will denote by :math:`\hat{A}_{\pi}^{(k)}`.

.. math::
    :label: ppo-eq-8

    \begin{array}{ll}
    \hat{A}_{\pi}^{(1)}:=\delta_t^V =-V(s_t)+r_t+\gamma V(s_{t+1}) \\
    \hat{A}_{\pi}^{(2)}:=\delta_t^V+\gamma \delta_{t+1}^V =-V(s_t)+r_t+\gamma r_{t+1}+\gamma^2 V(s_{t+2}) \\
    \hat{A}_{\pi}^{(3)}:=\delta_t^V+\gamma \delta_{t+1}^V+\gamma^2 \delta_{t+2}^V =-V(s_t)+r_t+\gamma r_{t+1}+\gamma^2 r_{t+2}+\gamma^3 V(s_{t+3}) \\
    \hat{A}_{\pi}^{(k)}:=\sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V =-V(s_t)+r_t+\gamma r_{t+1}+\cdots+\gamma^{k-1} r_{t+k-1}+\gamma^k V(s_{t+k})
    \end{array}

We can consider :math:`\hat{A}_{\pi}^{(k)}` to be an estimator of the advantage function.

.. hint::
    The bias generally becomes smaller as :math:`k arrow +\infty`,
    since the term :math:`\gamma^k V(s_{t+k})` becomes more heavily discounted.
    Taking :math:`k \rightarrow +\infty`, we get:

    .. math::
        :label: ppo-eq-9

        \hat{A}_{\pi}^{(\infty)}=\sum_{l=0}^{\infty} \gamma^l \delta_{t+l}^V=-V(s_t)+\sum_{l=0}^{\infty} \gamma^l r_{t+l}


    which is simply the empirical returns minus the value function baseline.

The generalized advantage estimator :math:`\text{GAE}(\gamma,\lambda)` is defined as the exponentially-weighted average of these :math:`k`-step estimators:

.. _ppo-eq-6:

.. math::
    :label: ppo-eq-10

    \hat{A}_{\pi}:&= (1-\lambda)(\hat{A}_{\pi}^{(1)}+\lambda \hat{A}_{\pi}^{(2)}+\lambda^2 \hat{A}_{\pi}^{(3)}+\ldots) \\
    &= (1-\lambda)(\delta_t^V+\lambda(\delta_t^V+\gamma \delta_{t+1}^V)+\lambda^2(\delta_t^V+\gamma \delta_{t+1}^V+\gamma^2 \delta_{t+2}^V)+\ldots) \\
    &= (1-\lambda)(\delta_t^V(1+\lambda+\lambda^2+\ldots)+\gamma \delta_{t+1}^V(\lambda+\lambda^2+\lambda^3+\ldots) .+\gamma^2 \delta_{t+2}^V(\lambda^2+\lambda^3+\lambda^4+\ldots)+\ldots) \\
    &= (1-\lambda)(\delta_t^V(\frac{1}{1-\lambda})+\gamma \delta_{t+1}^V(\frac{\lambda}{1-\lambda})+\gamma^2 \delta_{t+2}^V(\frac{\lambda^2}{1-\lambda})+\ldots) \\
    &= \sum_{l=0}^{\infty}(\gamma \lambda)^l \delta_{t+l}^V


There are two notable special cases of this formula, obtained by setting :math:`\lambda =0` and :math:`\lambda =1`.

.. math::
    :label: ppo-eq-11

    \text{GAE}(\gamma, 0):\quad & \hat{A}_{\pi}:=\delta_t  =r_t+\gamma V(s_{t+1})-V(s_t) \\
    \text{GAE}(\gamma, 1):\quad & \hat{A}_{\pi}:=\sum_{l=0}^{\infty} \gamma^l \delta_{t+l}  =\sum_{l=0}^{\infty} \gamma^l r_{t+l}-V(s_t)


.. hint::
    :math:`\text{GAE}(\gamma,1)` is the traditional MC-based method to estimate the advantage function,
    but it has high variance due to the sum of terms.
    :math:`\text{GAE}(\gamma,0)` is TD-based method with low variance,
    but is suffers from bias.

The generalized advantage estimator for :math:`0\le\lambda\le1` makes a compromise between bias and variance,
controlled by parameter :math:`\lambda`.

Code with OmniSafe
~~~~~~~~~~~~~~~~~~

Quick start
"""""""""""

.. card::
    :class-header: sd-bg-success sd-text-white sd-font-weight-bold
    :class-card: sd-outline-success  sd-rounded-1 sd-font-weight-bold
    :class-footer: sd-font-weight-bold

    Run PPO in Omnisafe
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
                        'total_steps': 1024000,
                        'vector_env_nums': 1,
                        'parallel': 1,
                    },
                    'algo_cfgs': {
                        'update_cycle': 2048,
                        'update_iters': 1,
                    },
                    'logger_cfgs': {
                        'use_wandb': False,
                    },
                }

                agent = omnisafe.Agent('PPO', env_id, custom_cfgs=custom_cfgs)
                agent.learn()


        .. tab-item:: Terminal config style

            We use ``train_on_policy.py`` as the entrance file. You can train the agent with PPO simply using ``train_on_policy.py``, with arguments about PPO and environments does the training.
            For example, to run PPO in SafetyPointGoal1-v0 , with 4 cpu cores and seed 0, you can use the following command:

            .. code-block:: bash
                :linenos:

                cd examples
                python train_policy.py --algo PPO --env-id SafetyPointGoal1-v0 --parallel 1 --total-steps 1024000 --device cpu --vector-env-nums 1 --torch-threads 1

------

Here are the documentation of PPO in PyTorch version.


Architecture of functions
"""""""""""""""""""""""""

- ``ppo.learn()``

  - ``env.roll_out()``
  - ``ppo.update()``

    - ``ppo.buf.get()``
    - ``ppo.update_policy_net()``
    - ``ppo.update_value_net()``

- ``ppo.log()``

------

Documentation of new functions
""""""""""""""""""""""""""""""

.. tab-set::

    .. tab-item:: ppo.compute_loss_pi()

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success  sd-rounded-1 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            ppo.compute_loss_pi()
            ^^^
            Compute the loss of Actor ``pi``, flowing the next steps:

            (1) Get the policy importance sampling ratio.

            .. code-block:: python
                :linenos:

                dist, _log_p = self.ac.pi(data['obs'], data['act'])
                # Importance ratio
                ratio = torch.exp(_log_p - data['log_p'])


            (2) Get the clipped surrogate function.

            .. code-block:: python
                :linenos:

                ratio_clip = torch.clamp(ratio, 1 - self.clip, 1 + self.clip)
                loss_pi = -(torch.min(ratio * data['adv'], ratio_clip * data['adv'])).mean()
                loss_pi -= self.entropy_coef * dist.entropy().mean()


            (3) Log useful information.

            .. code-block:: python
                :linenos:

                approx_kl = (0.5 * (dist.mean - data['act']) ** 2 / dist.stddev**2).mean().item()
                ent = dist.entropy().mean().item()
                pi_info = dict(kl=approx_kl, ent=ent, ratio=ratio_clip.mean().item())

            (4) Return the loss of Actor ``pi`` and useful information.

------

Parameters
""""""""""

.. tab-set::

    .. tab-item:: Specific Parameters

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success  sd-rounded-1 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            Specific Parameters
            ^^^
            -  target_kl(float): Constraint for KL-distance to avoid too far gap
            -  cg_damping(float): parameter plays a role in building Hessian-vector
            -  cg_iters(int): Number of iterations of conjugate gradient to perform.
            -  cost_limit(float): Constraint for agent to avoid too much cost

    .. tab-item:: Basic parameters

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success  sd-rounded-1 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            Basic parameters
            ^^^
            -  algo (string): The name of algorithm corresponding to current class,
               it does not actually affect any things which happen in the following.
            -  actor (string): The type of network in actor, discrete or continuous.
            -  model_cfgs (dictionary) : Actor and critic's net work configuration,
               it originates from ``algo.yaml`` file to describe ``hidden layers`` , ``activation function``, ``shared_weights`` and ``weight_initialization_mode``.

               -  shared_weights (bool) : Use shared weights between actor and critic network or not.

               -  weight_initialization_mode (string) : The type of weight initialization method.

                  -  pi (dictionary) : parameters for actor network ``pi``

                     -  hidden_sizes:

                        -  64
                        -  64

                     -  activations: tanh

                  -  val (dictionary) parameters for critic network ``v``

                     -  hidden_sizes:

                        -  64
                        -  64

                        .. hint::

                            ======== ================  ========================================================================
                            Name        Type              Description
                            ======== ================  ========================================================================
                            ``v``    ``nn.Module``     Gives the current estimate of **V** for states in ``s``.
                            ``pi``   ``nn.Module``     Deterministically or continuously computes an action from the agent,
                                                       conditioned on states in ``s``.
                            ======== ================  ========================================================================

                  -  activations: tanh
                  -  env_id (string): The name of environment we want to roll out.
                  -  seed (int): Define the seed of experiments.
                  -  parallel (int): Define the seed of experiments.
                  -  epochs (int): The number of epochs we want to roll out.
                  -  steps_per_epoch (int):The number of time steps per epoch.
                  -  pi_iters (int): The number of iteration when we update actor network per mini batch.
                  -  critic_iters (int): The number of iteration when we update critic network per mini batch.

    .. tab-item:: Optional parameters

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success  sd-rounded-1 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            Optional parameters
            ^^^
            -  use_cost_critic (bool): Use cost value function or not.
            -  linear_lr_decay (bool): Use linear learning rate decay or not.
            -  exploration_noise_anneal (bool): Use exploration noise anneal or not.
            -  reward_penalty (bool): Use cost to penalize reward or not.
            -  kl_early_stopping (bool): Use KL early stopping or not.
            -  max_grad_norm (float): Use maximum gradient normalization or not.
            -  scale_rewards (bool): Use reward scaling or not.

    .. tab-item:: Buffer parameters

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success  sd-rounded-1 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            Buffer parameters
            ^^^
            .. hint::
                  ============= =============================================================================
                     Name                    Description
                  ============= =============================================================================
                  ``Buffer``      A buffer for storing trajectories experienced by an agent interacting
                                  with the environment, and using **Generalized Advantage Estimation (GAE)**
                                  for calculating the advantages of state-action pairs.
                  ============= =============================================================================

            .. warning::
                Buffer collects only raw data received from environment.

            -  gamma (float): The gamma for GAE.
            -  lam (float): The lambda for reward GAE.
            -  adv_estimation_method (float):Roughly what KL divergence we think is
               appropriate between new and old policies after an update. This will
               get used for early stopping. (Usually small, 0.01 or 0.05.)
            -  standardized_reward (int):  Use standardized reward or not.
            -  standardized_cost (bool): Use standardized cost or not.

------

References
----------

-  `Trust Region Policy Optimization <https://arxiv.org/abs/1502.05477>`__
-  `Proximal Policy Optimization Algorithms <https://arxiv.org/pdf/1707.06347.pdf>`__
