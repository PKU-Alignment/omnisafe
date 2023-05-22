First Order Constrained Optimization in Policy Space
====================================================

Quick Facts
-----------

.. card::
    :class-card: sd-outline-info  sd-rounded-1
    :class-body: sd-font-weight-bold

    #. FOCOPS is an :bdg-info-line:`on-policy` algorithm.
    #. FOCOPS can be used for environments with both :bdg-info-line:`discrete` and :bdg-info-line:`continuous` action spaces.
    #. FOCOPS is an algorithm using :bdg-info-line:`first-order method`.
    #. An :bdg-ref-info-line:`API Documentation <focopsapi>` is available for FOCOPS.

FOCOPS Theorem
--------------

Background
~~~~~~~~~~

**First Order Constrained Optimization in Policy Space (FOCOPS)** is a
first-order method that maximizes an agent's overall reward while ensuring the
agent satisfies a set of cost constraints. FOCOPS purposes that CPO has
disadvantages below:

.. grid:: 2

    .. grid-item::
        :columns: 12 6 6 6

        .. card::
            :class-header: sd-bg-warning sd-text-white sd-font-weight-bold
            :class-card: sd-outline-warning  sd-rounded-1 sd-font-weight-bold

            Problems of CPO
            ^^^
            -  Error resulting from taking sample trajectories from the current policy.

            -  Approximation errors resulting from Taylor approximations.

            -  Approximation errors result from using the conjugate method to calculate the inverse of the Fisher information matrix.

    .. grid-item::
        :columns: 12 6 6 6

        .. card::
            :class-header: sd-bg-primary sd-text-white sd-font-weight-bold
            :class-card: sd-outline-primary  sd-rounded-1 sd-font-weight-bold

            Advantage of FOCOPS
            ^^^
            -  Extremely simple to implement since it only utilizes first-order approximations.

            -  Simple first-order method avoids error caused by Taylor method and the conjugate method.

            -  Outperform CPO in the experiment.

            -  No recovery steps are required.


FOCOPS mainly includes the following contributions:

- Provides a **two-stage policy update** to optimize the current policy.
- Gives the practical implementation for solving the two-stage policy update.
- Offers rigorous derivative proofs for the above theories, as detailed in the :bdg-ref-info:`Appendix<focops-appendix>` to this tutorial.

One suggested reading order is CPO(:doc:`../saferl/cpo`),
PCPO(:doc:`../saferl/pcpo`), then FOCOPS. If you have yet to read the PCPO, it
does not matter.
Nevertheless, be sure to read this article after reading the CPO tutorial we
have written so that you can fully understand the following passage.


------

Optimization Objective
~~~~~~~~~~~~~~~~~~~~~~

In the previous chapters, you learned that CPO solves the following
optimization problems:

.. _`focops-eq-1`:

.. math::
    :label: focops-eq-1

    \pi_{k+1}&=\arg \max _{\pi \in \Pi_{\boldsymbol{{\boldsymbol{\theta}}}}} \mathbb{E}_{\substack{s \sim d_{\pi_k}\\a \sim \pi}}[A^R_{\pi_k}(s, a)]\\
    \text{s.t.} \quad J^{C_i}\left(\pi_k\right) &\leq d_i-\frac{1}{1-\gamma} \mathbb{E}_{\substack{s \sim d_{\pi_k} \\ a \sim \pi}}\left[A^{C_i}_{\pi_k}(s, a)\right] \quad \forall i  \\
    \bar{D}_{K L}\left(\pi \| \pi_k\right) &\leq \delta


where :math:`\prod_{{\boldsymbol{\theta}}}\subseteq\prod` denotes the parametrized policies
with parameters :math:`{\boldsymbol{\theta}}`, and :math:`\bar{D}_{K L}` is the :math:`KL`
divergence of two policies. In local policy search for CMDPs, we require policy
iterates to be feasible. Instead of optimizing over
:math:`\prod_{{\boldsymbol{\theta}}}`, PCPO optimizes over
:math:`\prod_{{\boldsymbol{\theta}}}\cap\prod_{C}`. Next, we
will introduce you to how FOCOPS solves the above optimization problems. For
you to have a clearer understanding, we hope that you will read the next
section with the following questions:

.. card::
    :class-header: sd-bg-primary sd-text-white sd-font-weight-bold
    :class-card: sd-outline-primary  sd-rounded-1 sd-font-weight-bold

    Questions
    ^^^
    -  What is a two-stage policy update, and how?

    -  How to practically implement FOCOPS?

    -  How do parameters impact the performance of the algorithm?

------

Two-stage Policy Update
~~~~~~~~~~~~~~~~~~~~~~~

Instead of solving the :eq:`focops-eq-1`  directly, FOCOPS uses a **two-stage**
approach summarized below:

.. card::
    :class-header: sd-bg-primary sd-text-white sd-font-weight-bold
    :class-card: sd-outline-primary  sd-rounded-1 sd-font-weight-bold

    Two-stage Policy Update
    ^^^
    -  Given policy :math:`\pi_{{\boldsymbol{\theta}}_k}`, find an optimal update policy :math:`\pi^*` by solving the optimization problem from :eq:`focops-eq-1` in the non-parameterized policy space.

    -  Project the policy found in the previous step back into the parameterized policy space :math:`\Pi_{{\boldsymbol{\theta}}}` by searching for the closest policy :math:`\pi_{{\boldsymbol{\theta}}}\in\Pi_{{\boldsymbol{\theta}}}` to :math:`\pi^*`, to obtain :math:`\pi_{{\boldsymbol{\theta}}_{k+1}}`.

------

Finding the Optimal Update Policy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the first stage, FOCOPS rewrites :eq:`focops-eq-1`  as below:

.. _`focops-eq-4`:

.. math::
    :label: focops-eq-2

    \pi^* &=\arg \max _{\pi \in \Pi} \mathbb{E}_{\substack{s \sim d_{\pi_k}\\a \sim \pi}}[A^R_{\pi_k}(s, a)]\\
    \text{s.t.} \quad  J^{C}\left(\pi_k\right) &\leq d-\frac{1}{1-\gamma} \mathbb{E}{\substack{s \sim d_{\pi_k} \\ a \sim \pi}}\left[A^{C}_{\pi_k}(s, a)\right] \quad  \\
    \bar{D}_{K L}\left(\pi \| \pi_k\right) & \leq \delta


These problems are only slightly different from :eq:`focops-eq-1` , that is,
what we focus on now is the non-parameterized policy :math:`\pi` but
not the policy parameter :math:`{\boldsymbol{\theta}}`.
Then FOCOPS provides a solution as follows:

.. _focops-theorem-1:

.. card::
    :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
    :class-card: sd-outline-info  sd-rounded-1
    :class-footer: sd-font-weight-bold
    :link: focops-appendix
    :link-type: ref

    Theorem 1
    ^^^
    Let :math:`\tilde{b}=(1-\gamma)\left(b-\tilde{J}^C\left(\pi_{{\boldsymbol{\theta}}_k}\right)\right)`.
    If :math:`\pi_{{\boldsymbol{\theta}}_k}` is a feasible solution, the optimal policy for :eq:`focops-eq-2` takes the form

    .. _`focops-eq-7`:

    .. math::
        :label: focops-eq-3

        \pi^*(a \mid s)=\frac{\pi_{{\boldsymbol{\theta}}_k}(a \mid s)}{Z_{\lambda, \nu}(s)} \exp \left(\frac{1}{\lambda}\left(A^R_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)-\nu A^C_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)\right)\right)

    where :math:`Z_{\lambda,\nu}(s)` is the partition function which ensures :eq:`focops-eq-3` is a valid probability distribution, :math:`\lambda` and :math:`\nu` are solutions to the optimization problem:

    .. _`focops-eq-8`:

    .. math::
        :label: focops-eq-4

        \min _{\lambda, \nu \geq 0} \lambda \delta+\nu \tilde{b}+\lambda \underset{\substack{s \sim d_{\pi_{{\boldsymbol{\theta}}_k}} \\ a \sim \pi^*}}{\mathbb{E}}[\log Z_{\lambda, \nu}(s)]

    +++
    The proof of the :bdg-info-line:`Theorem 1` can be seen in the :bdg-info:`Appendix`, click on this :bdg-info-line:`card` to jump to view.

The form of the optimal policy is intuitive.
It gives high probability mass to areas of the state-action space with high
return, offset by a penalty term times the cost advantage.
We will refer to the optimal solution to :eq:`focops-eq-2`  as the *optimal
update policy*.
Suppose you need help understanding the meaning of the above Equation.
In that case, you can first think that FOCOPS finally solves :eq:`focops-eq-2`
by solving :eq:`focops-eq-3` and :eq:`focops-eq-4`.
:bdg-info-line:`Theorem 1` is a viable solution.


.. tab-set::

    .. tab-item:: Question I
        :sync: key1

        .. card::
            :class-header: sd-bg-success  sd-text-white sd-font-weight-bold
            :class-card:  sd-outline-success  sd-rounded-1 sd-font-weight-bold

            Question
            ^^^
            What is the bound for FOCOPS worst-case guarantee for cost constraint?

    .. tab-item:: Question II
        :sync: key2

        .. card::
            :class-header: sd-bg-success  sd-text-white sd-font-weight-bold
            :class-card:  sd-outline-success  sd-rounded-1 sd-font-weight-bold

            Question
            ^^^
            Can FOCOPS solve the multi-constraint problem and how?


.. tab-set::

    .. tab-item:: Answer I
        :sync: key1

        .. card::
            :class-header: sd-bg-primary  sd-text-white sd-font-weight-bold
            :class-card:  sd-outline-primary  sd-rounded-1 sd-font-weight-bold

            Answer
            ^^^
            FOCOPS purposes that the optimal update policy :math:`\pi^*` satisfies the following bound for the worst-case guarantee for cost constraint in CPO:

            .. math::
                :label: focops-eq-5

                J^C\left(\pi^*\right) \leq d+\frac{\sqrt{2 \delta} \gamma \epsilon_{\pi^*}^C}{(1-\gamma)^2}

            where :math:`\epsilon^C_{\pi^*}=\max _s\left|\underset{a \sim \pi}{\mathbb{E}}\left[A^C_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)\right]\right|`.


    .. tab-item:: Answer II
        :sync: key2

        .. card::
            :class-header: sd-bg-primary  sd-text-white sd-font-weight-bold
            :class-card:  sd-outline-primary  sd-rounded-1 sd-font-weight-bold

            Answer
            ^^^
            By introducing Lagrange multipliers :math:`\nu_1,\nu_2,...,\nu_m\ge0`, one for each cost constraint and applying a similar duality argument, FOCOPS extends its results to accommodate for multiple constraints.

------

Approximating the Optimal Update Policy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The optimal update policy :math:`\pi^*` is obtained in the previous section.
However, it is not a parameterized policy.
In this section, we will show you how FOCOPS projects the optimal update policy
back into the parameterized policy space by minimizing the loss function:

.. math::
    :label: focops-eq-6

    \mathcal{L}({\boldsymbol{\theta}})=\underset{s \sim d_{\pi_{{\boldsymbol{\theta}}_k}}}{\mathbb{E}}\left[D_{\mathrm{KL}}\left(\pi_{\boldsymbol{\theta}} \| \pi^*\right)[s]\right]

Here :math:`\pi_{{\boldsymbol{\theta}}}\in \Pi_{{\boldsymbol{\theta}}}` is some projected policy that FOCOPS
will use to approximate the optimal update policy.
The first-order methods are also used to minimize this loss function:

.. card::
    :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
    :class-card: sd-outline-info  sd-rounded-1
    :class-footer: sd-font-weight-bold
    :link: focops-appendix
    :link-type: ref

    Corollary 1
    ^^^
    The gradient of :math:`\mathcal{L}({\boldsymbol{\theta}})` takes the form

    .. _`focops-eq-10`:

    .. math::
        :label: focops-eq-7

        \nabla_{\boldsymbol{\theta}} \mathcal{L}({\boldsymbol{\theta}})=\underset{s \sim d_{\pi_{\boldsymbol{\theta}}}}{\mathbb{E}}\left[\nabla_{\boldsymbol{\theta}} D_{K L}\left(\pi_{\boldsymbol{\theta}} \| \pi^*\right)[s]\right]

    where

    .. math::
        :label: focops-eq-8

        \nabla_{\boldsymbol{\theta}} D_{K L}\left(\pi_{\boldsymbol{\theta}} \| \pi^*\right)[s] &=\nabla_{\boldsymbol{\theta}} D_{K L}\left(\pi_{\boldsymbol{\theta}} \| \pi_{{\boldsymbol{\theta}}_k}\right)[s] \\
        & -\frac{1}{\lambda} \underset{a \sim \pi_{{\boldsymbol{\theta}}_k}}{\mathbb{E}}\left[\frac{\nabla_{\boldsymbol{\theta}} \pi_{\boldsymbol{\theta}}(a \mid s)}{\pi_{{\boldsymbol{\theta}}_k}(a \mid s)}\left(A^R_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)-\nu A^C_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)\right)\right]

    +++
    The proof of the :bdg-info-line:`Corollary 1` can be seen in the :bdg-info:`Appendix`, click on this :bdg-info-line:`card` to jump to view.

Note that :eq:`focops-eq-7` can be estimated by sampling from the trajectories
generated by policy :math:`\pi_{{\boldsymbol{\theta}}_k}` so it can be trained using
stochastic gradients.

:bdg-info-line:`Corollary 1` outlines the FOCOPS algorithm:

.. note::

    At every iteration, we begin with a policy :math:`\pi_{{\boldsymbol{\theta}}_k}`, which we use
    to run trajectories and gather data.
    We use that data and :eq:`focops-eq-4` first to estimate :math:`\lambda` and
    :math:`\nu`.
    We then draw a mini-batch from the data to estimate
    :math:`\nabla_{\boldsymbol{\theta}} \mathcal{L}({\boldsymbol{\theta}})`
    given in :bdg-info-line:`Corollary 1`.
    After taking a gradient step using Equation :eq:`focops-eq-7`,
    we draw another mini-batch then repeat the process.

------

Practical Implementation
------------------------

.. hint::

    Solving :eq:`focops-eq-4` is computationally impractical for large state or action spaces as it requires calculating the partition function :math:`Z_{\lambda,\nu}(s)`, which often involves evaluating a high-dimensional integral or sum.
    Furthermore, :math:`\lambda` and :math:`\nu` are depend on :math:`k` and should be adapted at every iteration.

This section will introduce you to how FOCOPS practically implements its
algorithm. In practice, though hyperparameter sweeps, FOCOPS found that
a fixed :math:`\lambda` provides good results, which means the value
:math:`\lambda` does not have to be updated. However, :math:`\nu` needs to be
continuously adapted during training to ensure cost-constraint satisfaction.
FOCOPS appeals to an intuitive heuristic for determining :math:`\nu` based on
primal-dual gradient methods. With strong duality, the optimal
:math:`\lambda^*` and :math:`\nu` minimizes the dual function
:eq:`focops-eq-4`, which is then denoted as :math:`L(\pi^*,\lambda,\nu)`. By
applying gradient descent w.r.t :math:`\nu` to minimize
:math:`L(\pi^*,\lambda,\nu)`, we obtain:

.. card::
    :class-header: sd-bg-success  sd-text-white sd-font-weight-bold
    :class-card: sd-outline-info  sd-rounded-1
    :class-footer: sd-font-weight-bold
    :link: focops-appendix
    :link-type: ref

    Corollary 2
    ^^^
    The derivative of :math:`L(\pi^*,\lambda,\nu)` w.r.t :math:`\nu` is

    .. _`focops-eq-12`:

    .. math::
        :label: focops-eq-9

        \frac{\partial L\left(\pi^*, \lambda, \nu\right)}{\partial \nu}=\tilde{b}-\underset{\substack{s \sim d_{\pi^*} \\ a \sim \pi^*}}{\mathbb{E}}\left[A^R_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)\right]

    +++
    The proof of the :bdg-success-line:`Corollary 2` can be seen in the :bdg-success:`Appendix`, click on this :bdg-success-line:`card` to jump to view.

The last term in the gradient expression in :eq:`focops-eq-9` cannot be
evaluated since we do not have access to :math:`\pi^*`.
Since :math:`\pi_{{\boldsymbol{\theta}}_k}` and :math:`\pi^*` are close, it is reasonable to
assume that :math:`\underset{\substack{s \sim d_{\pi_k}\\ a \sim \pi^*}}{\mathbb{E}}\left[A^R_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)\right] \approx \underset{\substack{s \sim d_{\pi_k}\\ a \sim \pi_{{\boldsymbol{\theta}}_k}}}{\mathbb{E}}\left[A^R_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)\right]=0`.
In practice, this term can be set to zero, which gives the updated term:

.. _`focops-eq-13`:

.. math::
    :label: focops-eq-10

    \nu \leftarrow \underset{\nu}{\operatorname{proj}}\left[\nu-\alpha\left(d-J^C\left(\pi_{{\boldsymbol{\theta}}_k}\right)\right)\right]


where :math:`\alpha` is the step size.
Note that we have incorporated the discount term :math:`(1-\gamma)` into
:math:`\tilde{b}` into the step size.
The projection operator :math:`proj_{\nu}` projects :math:`\nu` back into the
interval :math:`[0,\nu_{max}]`, where :math:`\nu_{max}` is chosen so that
:math:`\nu` does not become too large.
In fact. FOCOPS purposed that even setting :math:`\nu_{max}=+\infty` does not
appear to reduce performance greatly.
Practically, :math:`J^C(\pi_{{\boldsymbol{\theta}}_k})` can be estimated via Monte Carlo
methods using trajectories collected from :math:`\pi_{{\boldsymbol{\theta}}_k}`.
Using the update rule :eq:`focops-eq-10`, FOCOPS performs one update step on
:math:`\nu` before updating the policy parameters :math:`{\boldsymbol{\theta}}`.
A per-state acceptance indicator function :math:`I\left(s_j\right)^n:=\mathbf{1}_{D_{\mathrm{KL}}\left(\pi_{\boldsymbol{\theta}} \| \pi_{{\boldsymbol{\theta}}_k}\right)\left[s_j\right] \leq \delta}` is added to :eq:`focops-eq-7`,
in order better to enforce the accuracy for the first-order purposed method.

.. hint::

    Here :math:`N` is the number of samples collected by policy :math:`\pi_{{\boldsymbol{\theta}}_k}`. :math:`\hat A^R` and :math:`\hat A^C` are estimates of the advantage functions (for the return and cost) obtained from critic networks.
    The advantage functions are obtained using the Generalized Advantage Estimator (GAE).
    Note that FOCOPS only requires first-order methods (gradient descent) and is thus extremely simple to implement.

------

Variables Analysis
~~~~~~~~~~~~~~~~~~

In this section, we will explain the meaning of parameters :math:`\lambda` and
:math:`\mu` of FOCOPS and their impact on the algorithm's performance in the
experiment.

.. tab-set::

    .. tab-item:: Analysis of :math:`\lambda`

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Analysis of :math:`\lambda`
            ^^^
            In :eq:`focops-eq-3`, note that as :math:`\lambda \rightarrow 0`, :math:`\pi^*` approaches a greedy policy;
            as :math:`\lambda` increases, the policy becomes more exploratory.
            Therefore :math:`\lambda` is similar to the temperature term used in maximum entropy reinforcement learning,
            which has been shown to produce good results when fixed during training.
            In practice, FOCOPS finds that its algorithm reaches the best performance when the :math:`\lambda` is fixed.

    .. tab-item:: Analysis of :math:`\nu`

        .. card::
            :class-header: sd-bg-success  sd-text-white sd-font-weight-bold
            :class-card:  sd-outline-success  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Analysis of :math:`\nu`
            ^^^
            We recall that in :eq:`focops-eq-3`,
            :math:`\nu` acts as a cost penalty term. Increasing :math:`\nu` makes it less likely for state-action pairs with higher costs to be sampled by :math:`\pi^*`.
            Hence in this regard, the update rule in :eq:`focops-eq-10` is intuitive,
            because it increases :math:`\nu` if :math:`J^C(\pi_{{\boldsymbol{\theta}}_k})>d`
            (which means the agent violates the cost constraints) and decreases :math:`\nu` otherwise.

------

.. _focops_code_with_omniSafe:

Code with OmniSafe
~~~~~~~~~~~~~~~~~~

Quick start
"""""""""""

.. card::
    :class-header: sd-bg-success sd-text-white sd-font-weight-bold
    :class-card: sd-outline-success  sd-rounded-1 sd-font-weight-bold
    :class-footer: sd-font-weight-bold

    Run FOCOPS in OmniSafe
    ^^^
    Here are 3 ways to run FOCOPS in OmniSafe:

    * Run Agent from preset yaml file
    * Run Agent from custom config dict
    * Run Agent from custom terminal config

    .. tab-set::

        .. tab-item:: Yaml file style

            .. code-block:: python
                :linenos:

                import omnisafe


                env_id = 'SafetyPointGoal1-v0'

                agent = omnisafe.Agent('FOCOPS', env_id)
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

                agent = omnisafe.Agent('FOCOPS', env_id, custom_cfgs=custom_cfgs)
                agent.learn()


        .. tab-item:: Terminal config style

            We use ``train_policy.py`` as the entrance file. You can train the agent with FOCOPS simply using ``train_policy.py``, with arguments about FOCOPS and environments does the training.
            For example, to run FOCOPS in SafetyPointGoal1-v0 , with 1 torch thread, seed 0 and single environment, you can use the following command:

            .. code-block:: bash
                :linenos:

                cd examples
                python train_policy.py --algo FOCOPS --env-id SafetyPointGoal1-v0 --parallel 1 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 1

------

Architecture of functions
"""""""""""""""""""""""""

-  ``FOCOPS.learn()``

   - ``FOCOPS._env.rollout()``
   - ``FOCOPS._update()``

     - ``FOCOPS._buf.get()``
     - ``FOCOPS._update_lagrange()``
     - ``FOCOPS._update_actor()``
     - ``FOCOPS._update_cost_critic()``
     - ``FOCOPS._update_reward_critic()``

------


Documentation of algorithm specific functions
"""""""""""""""""""""""""""""""""""""""""""""

.. tab-set::

    .. tab-item:: _compute_adv_surrogate()

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success  sd-rounded-1 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            FOCOPS._compute_adv_surrogate()
            ^^^
            Compute the surrogate advantage function.

            .. code-block:: python
                :linenos:

                return (adv_r - self._lagrange.lagrangian_multiplier * adv_c) / (
                    1 + self._lagrange.lagrangian_multiplier
                )

    .. tab-item:: FOCOPS._loss_pi()

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success  sd-rounded-1 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            FOCOPS._loss_pi()
            ^^^
            Compute the loss of policy network.

            In FOCOPS, the loss is defined as:

            .. math::

                L = \nabla_{\boldsymbol{\theta}} D_{K L}\left(\pi_{\boldsymbol{\theta}}^{'} \| \pi_{{\boldsymbol{\theta}}}\right)[s]
                -\frac{1}{\eta} \underset{a \sim \pi_{{\boldsymbol{\theta}}}}
                {\mathbb{E}}\left[\frac{\nabla_{\boldsymbol{\theta}} \pi_{\boldsymbol{\theta}}(a \mid s)}
                {\pi_{{\boldsymbol{\theta}}}(a \mid s)}\left(A^{R}_{\pi_{{\boldsymbol{\theta}}}}(s, a)
                -\lambda A^C_{\pi_{{\boldsymbol{\theta}}}}(s, a)\right)\right]

            In code implementation, we use the following code to compute the loss:

            .. code-block:: python
                :linenos:

                distribution = self._actor_critic.actor(obs)
                logp_ = self._actor_critic.actor.log_prob(act)
                std = self._actor_critic.actor.std
                ratio = torch.exp(logp_ - logp)

                kl = torch.distributions.kl_divergence(distribution, self._p_dist).sum(-1, keepdim=True)
                loss = (kl - (1 / self._cfgs.algo_cfgs.focops_lam) * ratio * adv) * (
                    kl.detach() <= self._cfgs.algo_cfgs.focops_eta
                ).type(torch.float32)
                loss = loss.mean()
                loss -= self._cfgs.algo_cfgs.entropy_coef * distribution.entropy().mean()

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

                The following configs are specific to FOCOPS algorithm.

                - clip (float): Clipping parameter for FOCOPS.

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

    .. tab-item:: Lagrange

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success  sd-rounded-1 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            Lagrange Configs
            ^^^

            .. note::

                The following configs are specific to FOCOPS algorithm.

                - lagrangian_upper_bound (float): Upper bound of Lagrange multiplier.

            - cost_limit (float): Tolerance of constraint violation.
            - lagrangian_multiplier_init (float): Initial value of Lagrange multiplier.
            - lambda_lr (float): Learning rate of Lagrange multiplier.
            - lambda_optimizer (str): Optimizer for Lagrange multiplier.


------

References
----------

-  `Constrained Policy Optimization <https://arxiv.org/abs/1705.10528>`__
-  `Projection-Based Constrained Policy Optimization <https://arxiv.org/pdf/2010.03152.pdf>`__
-  `Trust Region Policy Optimization <https://arxiv.org/abs/1502.05477>`__
-  `First Order Constrained Optimization in Policy Space <https://arxiv.org/pdf/2002.06506.pdf>`__

.. _focops-appendix:

Appendix
--------

Proof for Theorem 1
~~~~~~~~~~~~~~~~~~~

.. card::
   :class-header: sd-bg-info sd-text-white sd-font-weight-bold
   :class-card: sd-outline-info  sd-rounded-1

   Lemma 1
   ^^^
   Problem
   :eq:`focops-eq-2`
   is convex w.r.t
   :math:`\pi={\pi(a|s):s\in \mathcal{S},a\in\mathcal{A}}`.

.. card::
    :class-header: sd-bg-info sd-text-white sd-font-weight-bold
    :class-card: sd-outline-info  sd-rounded-1

    Proof of Lemma 1
    ^^^
    First, note that the objective function is linear w.r.t :math:`\pi`.
    Since :math:`J^{C}(\pi_{{\boldsymbol{\theta}}_k})` is a constant w.r.t :math:`\pi`, constraint :eq:`focops-eq-2` is linear.
    Constraint :eq:`focops-eq-2` can be rewritten as :math:`\sum_s d_{\pi_{{\boldsymbol{\theta}}_k}}(s) D_{\mathrm{KL}}\left(\pi \| \pi_{{\boldsymbol{\theta}}_k}\right)[s] \leq \delta`.
    The :math:`KL` divergence is convex w.r.t its first argument.
    Hence constraint :eq:`focops-eq-2`, a linear combination of convex functions, is also convex.
    Since :math:`\pi_{{\boldsymbol{\theta}}_k}` satisfies constraint :eq:`focops-eq-2` also satisfies constraint :eq:`focops-eq-2`, therefore Slater's constraint qualification holds, and strong duality holds.

.. dropdown:: Proof of Theorem 1 (Click here)
    :color: info
    :class-body: sd-outline-info

    Based on :bdg-info-line:`Lemma 1` the optimal value of the :eq:`focops-eq-2`  :math:`p^*` can be solved by solving the corresponding dual problem.
    Let

    .. math::
        :label: focops-eq-11

        L(\pi, \lambda, \nu)=\lambda \delta+\nu \tilde{b}+\underset{s \sim d_{\pi_{{\boldsymbol{\theta}}_k}}}{\mathbb{E}}\left[A^{lag}-\lambda D_{\mathrm{KL}}\left(\pi \| \pi_{{\boldsymbol{\theta}}_k}\right)[s]\right]\nonumber

    where :math:`A^{lag}=\underset{a \sim \pi(\cdot \mid s)}{\mathbb{E}}\left[A^R_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)-\nu A^C_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)\right]`.
    Therefore.

    .. _`focops-eq-15`:

    .. math::
        :label: focops-eq-12

        p^*=\max _{\pi \in \Pi} \min _{\lambda, \nu \geq 0} L(\pi, \lambda, \nu)=\min _{\lambda, \nu \geq 0} \max _{\pi \in \Pi} L(\pi, \lambda, \nu)

    Note that if :math:`\pi^*`, :math:`\lambda^*`, :math:`\nu^*` are optimal for :eq:`focops-eq-12`, :math:`\pi^*` is also optimal for :eq:`focops-eq-2`  because of the strong duality.

    Consider the inner maximization problem in :eq:`focops-eq-12`.
    We separate it from the original problem and try to solve it first:

    .. _`focops-eq-16`:

    .. math::
        :label: focops-eq-13

        &\underset{\pi}{\operatorname{max}}  A^{lag}-\underset{a \sim \pi(\cdot \mid s)}{\mathbb{E}}\left[\lambda\left(\log \pi(a \mid s)+\log \pi_{{\boldsymbol{\theta}}_k}(a \mid s)\right)\right] \\
        \text { s.t. } & \sum_a \pi(a \mid s)=1 \\
        & \pi(a \mid s) \geq 0 \quad \forall a \in \mathcal{A}


    Which is equivalent to the inner maximization problem in :eq:`focops-eq-12`.
    We can solve this convex optimization problem using a simple Lagrangian argument.
    We can write the Lagrangian of it as:

    .. math::
        :label: focops-eq-14

        G(\pi)=\sum_a \pi(a \mid s)[A^R_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)-\nu A^C_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)
        -\lambda(\log \pi(a \mid s)-\log \pi_{{\boldsymbol{\theta}}_k}(a \mid s))+\zeta]-1


    where :math:`\zeta > 0` is the Lagrange multiplier associated with the constraint :math:`\sum_a \pi(a \mid s)=1`.
    Different :math:`G(\pi)` w.r.t. :math:`\pi(a \mid s)` for some :math:`a`:

    .. _`focops-eq-18`:

    .. math::
        :label: focops-eq-15

        \frac{\partial G}{\partial \pi(a \mid s)}=A^R_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)-\nu A^C_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)-\lambda\left(\log \pi(a \mid s)+1-\log \pi_{{\boldsymbol{\theta}}_k}(a \mid s)\right)+\zeta


    Setting :eq:`focops-eq-15` to zero and rearranging the term, we obtain:

    .. math::
        :label: focops-eq-16

        \pi(a \mid s)=\pi_{{\boldsymbol{\theta}}_k}(a \mid s) \exp \left(\frac{1}{\lambda}\left(A^R_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)-\nu A^C_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)\right)+\frac{\zeta}{\lambda}+1\right)

    We chose :math:`\zeta` so that :math:`\sum_a \pi(a \mid s)=1` and rewrite :math:`\zeta / \lambda+1` as :math:`Z_{\lambda, \nu}(s)`.
    We find that the optimal solution :math:`\pi^*` to :eq:`focops-eq-13` takes the form

    .. math::
        :label: focops-eq-17

        \pi^*(a \mid s)=\frac{\pi_{{\boldsymbol{\theta}}_k}(a \mid s)}{Z_{\lambda, \nu}(s)} \exp \left(\frac{1}{\lambda}\left(A^R_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)-\nu A^C_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)\right)\right)

    Then we obtain:

    .. math::
        :label: focops-eq-18

        &\underset{\substack{s \sim d_{{\boldsymbol{\theta}}_{{\boldsymbol{\theta}}_k}} \\
        a \sim \pi^*}}{\mathbb{E}}\left[A^R_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)-\nu A^C_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)-\lambda\left(\log \pi^*(a \mid s)-\log \pi_{{\boldsymbol{\theta}}_k}(a \mid s)\right)\right] \\
        = &\underset{\substack{s \sim d_{\pi_{{\boldsymbol{\theta}}_k}} \\
        a \sim \pi^*}}{\mathbb{E}}\left[A^R_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)-\nu A^C_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)-\lambda\left(\log \pi_{{\boldsymbol{\theta}}_k}(a \mid s)-\log Z_{\lambda, \nu}(s)\right.\right. \\
        &\left.\left. + \frac{1}{\lambda}\left(A^R_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)-\nu A^C_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)\right)-\log \pi_{{\boldsymbol{\theta}}_k}(a \mid s)\right)\right]\\
        = &\lambda\underset{\substack{s \sim d_{{\boldsymbol{\theta}}_{{\boldsymbol{\theta}}_k}} \\
        a \sim \pi^*}}{\mathbb{E}}[logZ_{\lambda,\nu}(s)]\nonumber


    Plugging the result back to :eq:`focops-eq-12`, we obtain:

    .. math::
        :label: focops-eq-19

        p^*=\underset{\lambda,\nu\ge0}{\min}\lambda\delta+\nu\tilde{b}+\lambda\underset{\substack{s \sim d_{{\boldsymbol{\theta}}_{{\boldsymbol{\theta}}_k}} \\
        a \sim \pi^*}}{\mathbb{E}}[logZ_{\lambda,\nu}(s)]

------

.. _focops-proof-corollary:

Proof of Corollary
~~~~~~~~~~~~~~~~~~

.. tab-set::

   .. tab-item:: Proof of Corollary 1

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card:  sd-outline-info  sd-rounded-1

            Proof of Corollary 1
            ^^^
            We only need to calculate the gradient of the loss function for a single sampled s. We first note that,

            .. math::
                :label: focops-eq-20

                &D_{\mathrm{KL}}\left(\pi_{\boldsymbol{\theta}} \| \pi^*\right)[s]\\
                =&-\sum_a \pi_{\boldsymbol{\theta}}(a \mid s) \log \pi^*(a \mid s)+\sum_a \pi_{\boldsymbol{\theta}}(a \mid s) \log \pi_{\boldsymbol{\theta}}(a \mid s) \\
                =&H\left(\pi_{\boldsymbol{\theta}}, \pi^*\right)[s]-H\left(\pi_{\boldsymbol{\theta}}\right)[s]


            where :math:`H\left(\pi_{\boldsymbol{\theta}}\right)[s]` is the entropy and :math:`H\left(\pi_{\boldsymbol{\theta}}, \pi^*\right)[s]` is the cross-entropy under state :math:`s`.
            The above is the basic mathematical knowledge in information theory, which you can get in any information theory textbook.
            We expand the cross entropy term, which gives us the following:

            .. math::
                :label: focops-eq-21

                &H\left(\pi_{\boldsymbol{\theta}}, \pi^*\right)[s]\\
                &=-\sum_a \pi_{\boldsymbol{\theta}}(a \mid s) \log \pi^*(a \mid s) \\
                &=-\sum_a \pi_{\boldsymbol{\theta}}(a \mid s) \log \left(\frac{\pi_{{\boldsymbol{\theta}}_k}(a \mid s)}{Z_{\lambda, \nu}(s)} \exp \left[\frac{1}{\lambda}\left(A^R_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)-\nu A^C_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)\right)\right]\right) \\
                &=-\sum_a \pi_{\boldsymbol{\theta}}(a \mid s) \log \pi_{{\boldsymbol{\theta}}_k}(a \mid s)+\log Z_{\lambda, \nu}(s)-\frac{1}{\lambda} \sum_a \pi_{\boldsymbol{\theta}}(a \mid s)\left(A^R_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)-\nu A^C_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)\right)


            We then subtract the entropy term to recover the :math:`KL` divergence:

            .. math::
                :label: focops-eq-22

                &D_{\mathrm{KL}}\left(\pi_{\boldsymbol{\theta}} \| \pi^*\right)[s]=D_{\mathrm{KL}}\left(\pi_{\boldsymbol{\theta}} \| \pi_{{\boldsymbol{\theta}}_k}\right)[s]+\log Z_{\lambda, \nu}(s)-\\&\frac{1}{\lambda} \underset{a \sim \pi_{{\boldsymbol{\theta}}_k}(\cdot \mid s)}{\mathbb{E}}\left[\frac{\pi_{\boldsymbol{\theta}}(a \mid s)}{\pi_{{\boldsymbol{\theta}}_k}(a \mid s)}\left(A^R_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)-\nu A^C_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)\right)\right]\nonumber


            In the last equality, we applied importance sampling to rewrite the expectation w.r.t. :math:`\pi_{{\boldsymbol{\theta}}_k}`.
            Finally, taking the gradient on both sides gives us the following:

            .. math::
                :label: focops-eq-23

                &\nabla_{\boldsymbol{\theta}} D_{\mathrm{KL}}\left(\pi_{\boldsymbol{\theta}} \| \pi^*\right)[s]=\nabla_{\boldsymbol{\theta}} D_{\mathrm{KL}}\left(\pi_{\boldsymbol{\theta}} \| \pi_{{\boldsymbol{\theta}}_k}\right)[s]\\&-\frac{1}{\lambda} \underset{a \sim \pi_{{\boldsymbol{\theta}}_k}(\cdot \mid s)}{\mathbb{E}}\left[\frac{\nabla_{\boldsymbol{\theta}} \pi_{\boldsymbol{\theta}}(a \mid s)}{\pi_{{\boldsymbol{\theta}}_k}(a \mid s)}\left(A^R_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)-\nu A^C_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)\right)\right]\nonumber


   .. tab-item:: Proof of Corollary 2

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card:  sd-outline-info  sd-rounded-1

            Proof of Corollary 2
            ^^^
            From :bdg-ref-info-line:`Theorem 1<focops-theorem-1>`, we have:

            .. math::
                :label: focops-eq-24

                L\left(\pi^*, \lambda, \nu\right)=\lambda \delta+\nu \tilde{b}+\lambda \underset{\substack{s \sim d_{\pi^*} \\ a \sim \pi^*}}{\mathbb{E}}\left[\log Z_{\lambda, \nu}(s)\right]


            The first two terms are an affine function w.r.t. :math:`\nu`.
            Therefore, its derivative is :math:`\tilde{b}`. We will then focus on the expectation in the last term.
            To simplify our derivation, we will first calculate the derivative of :math:`\pi^*` w.r.t. :math:`\nu`,

            .. math::
                :label: focops-eq-25

                \frac{\partial \pi^*(a \mid s)}{\partial \nu} &=\frac{\pi_{{\boldsymbol{\theta}}_k}(a \mid s)}{Z_{\lambda, \nu}^2(s)}\left[Z_{\lambda, \nu}(s) \frac{\partial}{\partial \nu} \exp \left(\frac{1}{\lambda}\left(A^R_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)-\nu A^C_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)\right)\right)\right.\\
                &\left.-\exp \left(\frac{1}{\lambda}\left(A^R_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)-\nu A^C_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)\right)\right) \frac{\partial Z_{\lambda, \nu}(s)}{\partial \nu}\right] \\
                &=-\frac{A^C_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)}{\lambda} \pi^*(a \mid s)-\pi^*(a \mid s) \frac{\partial \log Z_{\lambda, \nu}(s)}{\partial \nu}\nonumber


            Therefore the derivative of the expectation in the last term of :math:`L(\pi^*,\lambda,\nu)` can be written as:

            .. _`focops-eq-22`:

            .. math::
                :label: focops-eq-26

                \frac{\partial}{\partial \nu} \underset{\substack{s \sim d_{\pi {\boldsymbol{\theta}}_k} \\
                a \sim \pi^*}}{\mathbb{E}}\left[\log Z_{\lambda, \nu}(s)\right]
                &= \underset{\substack{s \sim d_{\pi_{\boldsymbol{\theta}}} \\
                a \sim \pi_{{\boldsymbol{\theta}}_k}}}{\mathbb{E}}\left[\frac{\partial}{\partial \nu}\left(\frac{\pi^*(a \mid s)}{\pi_{{\boldsymbol{\theta}}_k}(a \mid s)} \log Z_{\lambda, \nu}(s)\right)\right] \\
                &= \underset{\substack{s \sim d_{\pi_{\boldsymbol{\theta}}} \\
                a \sim \pi_{{\boldsymbol{\theta}}_k}}}{\mathbb{E}}\left[\frac{1}{\pi_{{\boldsymbol{\theta}}_k}(a \mid s)}\left(\frac{\partial \pi^*(a \mid s)}{\partial \nu} \log Z_{\lambda, \nu}(s)+\pi^*(a \mid s) \frac{\partial \log Z_{\lambda, \nu}(s)}{\partial \nu}\right)\right] \\
                &= \underset{\substack{s \sim d_{\pi_{\boldsymbol{\theta}}} \\
                a \sim \pi^*}}{\mathbb{E}}\left[-(\frac{A^C_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)}{\lambda}+\frac{\partial \log Z_{\lambda, \nu}(s)}{\partial \nu}) \log Z_{\lambda, \nu}(s)+\frac{\partial \log Z_{\lambda, \nu}(s)}{\partial \nu}\right]


            Also:

            .. math::
                :label: focops-eq-27

                \frac{\partial Z_{\lambda, \nu}(s)}{\partial \nu} &=\frac{\partial}{\partial \nu} \sum_a \pi_{{\boldsymbol{\theta}}_k}(a \mid s) \exp \left(\frac{1}{\lambda}\left(A^R_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)-\nu A^C_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)\right)\right) \\
                &=\sum_a-\pi_{{\boldsymbol{\theta}}_k}(a \mid s) \frac{A^C_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)}{\lambda} \exp \left(\frac{1}{\lambda}\left(A^R_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)-\nu A^C_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)\right)\right) \\
                &=\sum_a-\frac{A^C_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)}{\lambda} \frac{\pi_{{\boldsymbol{\theta}}_k}(a \mid s)}{Z_{\lambda, \nu}(s)} \exp \left(\frac{1}{\lambda}\left(A^R_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)-\nu A^C_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)\right)\right) Z_{\lambda, \nu}(s) \\
                &=-\frac{Z_{\lambda, \nu}(s)}{\lambda} \underset{a \sim \pi^*(\cdot \mid s)}{\mathbb{E}}\left[A^C_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)\right]


            Therefore:

            .. _`focops-eq-24`:

            .. math::
                :label: focops-eq-28

                \frac{\partial \log Z_{\lambda, \nu}(s)}{\partial \nu}=\frac{\partial Z_{\lambda, \nu}(s)}{\partial \nu} \frac{1}{Z_{\lambda, \nu}(s)}=-\frac{1}{\lambda} \underset{a \sim \pi^*(\cdot \mid s)}{\mathbb{E}}\left[A^C_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)\right]

            Plugging :eq:`focops-eq-28`  into the last equality in :eq:`focops-eq-26`  gives us:

            .. _`focops-eq-25`:

            .. math::
                :label: focops-eq-29

                \frac{\partial}{\partial \nu} \underset{\substack{s \sim d_{\pi_{\boldsymbol{\theta}}} \\
                a \sim \pi^*}}{\mathbb{E}}\left[\log Z_{\lambda, \nu}(s)\right]
                &=\underset{\substack{s \sim d_{\pi^*} \\
                a \sim \pi^*}}{\mathbb{E}}\left[-\frac{A^C_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)}{\lambda} \log Z_{\lambda, \nu}(s)+\frac{A^C_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)}{\lambda} \log Z_{\lambda, \nu}(s)-\frac{1}{\lambda} A^C_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)\right] \\
                &=-\frac{1}{\lambda} \underset{\substack{s \sim d_{\pi_{{\boldsymbol{\theta}}_k}} \\
                a \sim \pi^*}}{\mathbb{E}}\left[A^C_{\pi_{{\boldsymbol{\theta}}_k}}(s, a)\right]


            Combining :eq:`focops-eq-29`  with the derivatives of the affine term give us the final desired result.
