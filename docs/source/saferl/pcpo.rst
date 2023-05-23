Projection-Based Constrained Policy Optimization
================================================

Quick Facts
-----------

.. card::
    :class-card: sd-outline-info  sd-rounded-1
    :class-body: sd-font-weight-bold

    #. PCPO is an :bdg-info-line:`on-policy` algorithm.
    #. PCPO can be used for environments with both :bdg-info-line:`discrete` and :bdg-info-line:`continuous` action spaces.
    #. PCPO is an improvement work done based on :bdg-info-line:`CPO` .
    #. The OmniSafe implementation of PCPO support :bdg-success-line:`parallelization`.
    #. An :bdg-ref-info-line:`API Documentation <pcpoapi>` is available for PCPO.


PCPO Theorem
------------

Background
~~~~~~~~~~

**Projection-Based Constrained Policy Optimization (PCPO)** is a two-stage
iterative method for optimizing policies. The first stage involves a local
reward improvement update, while the second stage reconciles any constraint
violation by projecting the policy back onto the constraint set.

PCPO is an improvement work based on **CPO** (:doc:`../saferl/cpo`),
which also provides lower bounds on reward improvement and upper bounds on
constraint violation.

In addition to these guarantees, PCPO characterizes its convergence based on
two metrics: :math:`L2` norm and :math:`KL` divergence. It is designed to
address the challenge of learning control policies that optimize a reward
function while satisfying constraints.

.. hint::

    If you are new to the CPO family of algorithms, we recommend reading CPO
    tutorial (:doc:`./cpo`) to fully understand the concepts
    introduced in this chapter.

------

Optimization Objective
~~~~~~~~~~~~~~~~~~~~~~

In the previous chapters, you learned that CPO solves the following
optimization problems:

.. math::
    :label: pcpo-eq-1

    \pi_{k+1} &= \arg\max_{\pi \in \Pi_{\boldsymbol{\theta}}}J^R(\pi)\\
    \text{s.t.}\quad D(\pi,\pi_k) & \le\delta\\
    J^{C_i}(\pi) &\le d_i\quad i=1,...m


where :math:`\Pi_{\boldsymbol{\theta}}\subseteq\Pi` denotes the set of parametrized policies
with parameters :math:`\boldsymbol{\theta}`, and :math:`D` is some distance measure.
In local policy search for CMDPs, we additionally require policy iterates to be
feasible, so instead of optimizing over :math:`\Pi_{\boldsymbol{\theta}}`, PCPO optimizes
over :math:`\Pi_{\boldsymbol{\theta}}\cap\Pi_{C}`.
Next, we will introduce you to how PCPO solves the above optimization problems.
For you to have a clearer understanding, we hope that you will read the next
section with the following questions:

.. card::
    :class-header: sd-bg-primary sd-text-white sd-font-weight-bold
    :class-card: sd-outline-primary  sd-rounded-1 sd-font-weight-bold

    Questions
    ^^^
    -  What is a two-stage policy update and how?

    -  What is performance bound for PCPO and how PCPO get it?

    -  How PCPO practically solve the optimal problem?

------

Two-stage Policy Update
~~~~~~~~~~~~~~~~~~~~~~~

PCPO updates policy in **two stages**. The first stage
is :bdg-ref-info-line:`Reward Improvement Stage<two stage update>`,
which maximizes reward using a trust region optimization method without
constraints. This might results in a new intermediate policy that fails to
satisfy the constraints. The second stage,
named :bdg-ref-info-line:`Projection Stage<two stage update>`,
reconciles the constraint violation (if any) by projecting the policy back onto
the constraint set, choosing the procedure in the constraint set that is
closest to the selected intermediate policy. Next, we will describe how PCPO
completes the two-stage update.

.. _`two stage update`:

.. tab-set::

    .. tab-item:: Stage 1

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Reward Improvement Stage
            ^^^
            First, PCPO optimizes the reward function by maximizing the reward advantage function :math:`A^R_{\pi}(s,a)` subject to :math:`KL`-Divergence constraint.
            This constraints the intermediate policy :math:`\pi_{k+\frac12}` to be within a :math:`\delta`-neighborhood of :math:`\pi_{k}`:

            .. math::
                :label: pcpo-eq-2

                &\pi_{k+\frac12}=\arg\underset{\pi}{\max}\underset{\substack{s\sim d_{\pi_k}\\ a\sim\pi}}{\mathbb{E}}[A^R_{\pi_k}(s,a)]\\
                \text{s.t.}\quad &\underset{s\sim d_{\pi_k}}{\mathbb{E}}[D_{KL}(\pi||\pi_k)[s]]\le\delta\nonumber


            This update rule with the trust region is called **TRPO** (sees in :doc:`../baserl/trpo`).
            It constraints the policy changes to a divergence neighborhood and guarantees reward improvement.

    .. tab-item:: Stage 2

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card:  sd-outline-info  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Projection Stage
            ^^^
            Second, PCPO projects the intermediate policy :math:`\pi_{k+\frac12}` onto the constraint set by minimizing a distance measure :math:`D` between :math:`\pi_{k+\frac12}` and :math:`\pi`:

            .. math::
                :label: pcpo-eq-3

                &\pi_{k+1}=\arg\underset{\pi}{\min} D(\pi,\pi_{k+\frac12})\\
                \text{s.t.}\quad &J^C\left(\pi_k\right)+\underset{\substack{s\sim d_{\pi_k}\\ a\sim\pi}}{\mathbb{E}}\left[A^C_{\pi_k}(s, a)\right] \leq d

The :bdg-ref-info-line:`Projection Stage<two stage update>` guarantees that the
constraint-satisfying policy :math:`\pi_{k+1}` remains in close proximity to
:math:`\pi_{k+\frac{1}{2}}`. On the other hand,
the :bdg-ref-info-line:`Reward Improvement Stage<two stage update>` ensures
that the agent's updates aim to maximize rewards without violating the distance
measure :math:`D`.
The :bdg-ref-info-line:`Projection Stage<two stage update>` prompts the agent
to update its policy in a direction that satisfies the constraint while
not across :math:`D`.

------

Policy Performance Bounds
~~~~~~~~~~~~~~~~~~~~~~~~~

In safety-critical applications, **how worse the performance of a system
evolves when applying a learning algorithm** is an important issue.
For the two cases where the agent satisfies the constraint and does not satisfy
the constraint, PCPO provides worst-case performance bound respectively.

.. _`performance bound`:

.. tab-set::

    .. tab-item:: Theorem 1

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-1
            :class-footer: sd-font-weight-bold
            :link: cards-clickable
            :link-type: ref

            Worst-case Bound on Updating Constraint-satisfying Policies
            ^^^
            Define :math:`\epsilon_{\pi_{k+1}}^{R}\doteq \max\limits_{s}\big|\underset{a\sim\pi_{k+1}}{\mathbb{E}}[A^{R}_{\pi_{k}}(s,a)]\big|`, and :math:`\epsilon_{\pi_{k+1}}^{C}\doteq \max\limits_{s}\big|\underset{a\sim\pi_{k+1}}{\mathbb{E}}[A^{C}_{\pi_{k}}(s,a)]\big|`.
            If the current policy :math:`\pi_k` satisfies the constraint, then under :math:`KL` divergence projection, the lower bound on reward improvement, and upper bound on constraint violation for each policy update are

            .. math::
                :label: pcpo-eq-4

                J^{R}(\pi_{k+1})-J^{R}(\pi_{k})&\geq-\frac{\sqrt{2\delta}\gamma\epsilon_{\pi_{k+1}}^{R}}{(1-\gamma)^{2}}\\
                J^{C}(\pi_{k+1})&\leq d+\frac{\sqrt{2\delta}\gamma\epsilon_{\pi_{k+1}}^{C}}{(1-\gamma)^{2}}


            where :math:`\delta` is the step size in the reward improvement step.
            +++
            The proof of the :bdg-info-line:`Theorem 1` can be seen in the :bdg-info:`CPO tutorial`, click on this :bdg-info-line:`card` to jump to view.

    .. tab-item:: Theorem 2

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card:  sd-outline-info  sd-rounded-1
            :class-footer: sd-font-weight-bold
            :link: pcpo-performance-bound-proof
            :link-type: ref

            Worst-case Bound on Updating Constraint-violating Policies
            ^^^
            Define :math:`\epsilon_{\pi_{k+1}}^{R}\doteq \max\limits_{s}\big|\underset{a\sim\pi_{k+1}}{\mathbb{E}}[A^{R}_{\pi_{k}}(s,a)]\big|`, :math:`\epsilon_{\pi_{k+1}}^{C}\doteq \max\limits_{s}\big|\underset{a\sim\pi_{k+1}}{\mathbb{E}}[A^{C}_{\pi_{k}}(s,a)]\big|`, :math:`b^{+}\doteq \max(0,J^{C}(\pi_k)-d),` and :math:`\alpha_{KL} \doteq \frac{1}{2a^T\boldsymbol{H}^{-1}a},` where :math:`a` is the gradient of the cost advantage function and :math:`\boldsymbol{H}` is the Hessian of the :math:`KL` divergence constraint.
            If the current policy :math:`\pi_k` violates the constraint, then under :math:`KL` divergence projection, the lower bound on reward improvement and the upper bound on constraint violation for each policy update are

            .. math::
                :label: pcpo-eq-5

                J^{R}(\pi_{k+1})-J^{R}(\pi_{k})\geq&-\frac{\sqrt{2(\delta+{b^+}^{2}\alpha_\mathrm{KL})}\gamma\epsilon_{\pi_{k+1}}^{R}}{(1-\gamma)^{2}}\\
                J^{C}(\pi_{k+1})\leq& ~d+\frac{\sqrt{2(\delta+{b^+}^{2}\alpha_\mathrm{KL})}\gamma\epsilon_{\pi_{k+1}}^{C}}{(1-\gamma)^{2}}


            where :math:`\delta` is the step size in the reward improvement step.
            +++
            The proof of the :bdg-info-line:`Theorem 2` can be seen in the :bdg-info:`Appendix`, click on this :bdg-info-line:`card` to jump to view.

------

Practical Implementation
------------------------

Implementation of a Two-stage Update
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a large neural network policy with thousands of parameters,
directly solving for the PCPO update in :eq:`pcpo-eq-2` and :eq:`pcpo-eq-3` is
impractical due to the computational cost.
PCPO proposes that with a small step size :math:`\delta`, the reward function
and the :math:`KL` divergence constraint in the reward improvement step can
be approximated with a first-order expansion, while the KL
divergence measure in the projection step can also be approximated with a
second-order expansion.

.. tab-set::

    .. tab-item:: Implementation of Stage 1

        .. card::
            :class-header: sd-bg-success  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success  sd-rounded-1
            :class-footer: sd-font-weight-bold
            :link: pcpo-code-with-omnisafe
            :link-type: ref

            Reward Improvement Stage
            ^^^
            Define:

            :math:`g\doteq\nabla_\boldsymbol{\theta}\underset{\substack{s\sim d_{\pi_k} \\ a\sim \pi}}{\mathbb{E}}[A_{\pi_k}^{R}(s,a)]` is the gradient of the reward advantage function,

            :math:`a\doteq\nabla_\boldsymbol{\theta}\underset{\substack{s\sim d_{\pi_k} \\ a\sim \pi}}{\mathbb{E}}[A_{\pi_k}^{C}(s,a)]` is the gradient of the cost advantage function,

            where :math:`\boldsymbol{H}_{i,j}\doteq \frac{\partial^2 \underset{s\sim d_{\pi_{k}}}{\mathbb{E}}\big[KL(\pi ||\pi_{k})[s]\big]}{\partial \boldsymbol{\theta}_j\partial \boldsymbol{\theta}_j}` is the Hessian of the :math:`KL` divergence constraint (:math:`\boldsymbol{H}` is also called the Fisher information matrix. It is symmetric positive semi-definite), :math:`b\doteq J^{C}(\pi_k)-d` is the constraint violation of the policy :math:`\pi_{k}`, and :math:`\boldsymbol{\theta}` is the parameter of the policy. PCPO linearize the objective function at :math:`\pi_k` subject to second order approximation of the :math:`KL` divergence constraint to obtain the following updates:

            .. math::
                :label: pcpo-eq-6

                &\boldsymbol{\theta}_{k+\frac{1}{2}} =\arg \underset{\boldsymbol{\theta}}{\max}g^{T}(\boldsymbol{\theta}-\boldsymbol{\theta}_k)  \\
                \text{s.t.}\quad &\frac{1}{2}(\boldsymbol{\theta}-\boldsymbol{\theta}_{k})^{T}\boldsymbol{H}(\boldsymbol{\theta}-\boldsymbol{\theta}_k)\le \delta . \label{eq:update1}


            The above problem is essentially an optimization problem presented in TRPO, which can be completely solved using the method we introduced in the TRPO tutorial.
            +++
            The OmniSafe code of the :bdg-success-line:`Implementation of Stage I` can be seen in the :bdg-success:`Code with OmniSafe`, click on this :bdg-success-line:`card` to jump to view.

    .. tab-item:: Implementation of Stage 2

        .. card::
            :class-header: sd-bg-success  sd-text-white sd-font-weight-bold
            :class-card:  sd-outline-success  sd-rounded-1
            :class-footer: sd-font-weight-bold
            :link: pcpo-code-with-omnisafe
            :link-type: ref

            Projection Stage
            ^^^
            PCPO provides a selection reference for distance measures: if the projection is defined in the parameter space, :math:`L2` norm projection is selected, while if the projection is defined in the probability space, :math:`KL` divergence projection is better.
            This can be approximated through the second-order expansion.
            Again, PCPO linearize the cost constraint at :math:`\pi_{k}`.
            This gives the following update for the projection step:

            .. math::
                :label: pcpo-eq-7

                &\boldsymbol{\theta}_{k+1} =\arg \underset{\boldsymbol{\theta}}{\min}\frac{1}{2}(\boldsymbol{\theta}-{\boldsymbol{\theta}}_{k+\frac{1}{2}})^{T}\boldsymbol{L}(\boldsymbol{\theta}-{\boldsymbol{\theta}}_{k+\frac{1}{2}})\\
                \text{s.t.}\quad & a^{T}(\boldsymbol{\theta}-\boldsymbol{\theta}_{k})+b\leq 0


            where :math:`\boldsymbol{L}=\boldsymbol{I}` for :math:`L2` norm projection, and :math:`\boldsymbol{L}=\boldsymbol{H}` for :math:`KL` divergence projection.
            +++
            The OmniSafe code of the :bdg-success-line:`Implementation of Stage II` can be seen in the :bdg-success:`Code with OmniSafe`, click on this :bdg-success-line:`card` to jump to view.

PCPO solves :eq:`cpo-eq-4` and :eq:`pcpo-eq-5` using
:bdg-success-line:`convex programming`, see detailed in
:bdg-ref-success:`Appendix<convex-programming>`.

For each policy update:

.. _pcpo-eq-10:

.. math::
    :label: pcpo-eq-8

    \boldsymbol{\theta}_{k+1}=\boldsymbol{\theta}_{k}+\sqrt{\frac{2\delta}{g^T\boldsymbol{H}^{-1}g}}\boldsymbol{H}^{-1}g
    -\max\left(0,\frac{\sqrt{\frac{2\delta}{g^T\boldsymbol{H}^{-1}g}}a^{T}\boldsymbol{H}^{-1}g+b}{a^T\boldsymbol{L}^{-1}a}\right)\boldsymbol{L}^{-1}a


.. hint::

    :math:`\boldsymbol{H}` is assumed invertible and PCPO requires to invert :math:`\boldsymbol{H}`, which is impractical for huge neural network policies.
    Hence it uses the conjugate gradient method.
    (See appendix for a discussion of the trade-off between the approximation error, and computational efficiency of the conjugate gradient method.)

.. grid:: 2

    .. grid-item::
        :columns: 12 6 6 5

        .. tab-set::

            .. tab-item:: Question I
                :sync: key1

                .. card::
                    :class-header: sd-bg-success  sd-text-white sd-font-weight-bold
                    :class-card:  sd-outline-success  sd-rounded-1 sd-font-weight-bold

                    Question
                    ^^^
                    Is using a linear approximation to the constraint set enough to ensure constraint satisfaction?

            .. tab-item:: Question II
                :sync: key2

                .. card::
                    :class-header: sd-bg-success  sd-text-white sd-font-weight-bold
                    :class-card:  sd-outline-success  sd-rounded-1 sd-font-weight-bold

                    Question
                    ^^^
                    Can PCPO solve the multi-constraint problem? And how PCPO do that?

    .. grid-item::
        :columns: 12 6 6 7

        .. tab-set::

            .. tab-item:: Answer I
                :sync: key1

                .. card::
                    :class-header: sd-bg-primary  sd-text-white sd-font-weight-bold
                    :class-card:  sd-outline-primary  sd-rounded-1 sd-font-weight-bold

                    Answer
                    ^^^
                    If the step size :math:`\delta` is small, then the linearization of the constraint set is accurate enough to locally approximate it.

            .. tab-item:: Answer II
                :sync: key2

                .. card::
                    :class-header: sd-bg-primary  sd-text-white sd-font-weight-bold
                    :class-card:  sd-outline-primary  sd-rounded-1 sd-font-weight-bold

                    Answer
                    ^^^
                    By sequentially projecting onto each of the sets,
                    the update in :eq:`pcpo-eq-5` can be extended by using alternating projections.

------

Analysis
~~~~~~~~

The update rule in :eq:`pcpo-eq-5` shows that the difference between PCPO with
:math:`KL` divergence and :math:`L2` norm projections are **the cost update
direction**, leading to a difference in reward improvement.
These two projections converge to different stationary points with different
convergence rates related to the smallest and largest singular values of the
Fisher information matrix shown in :bdg-info-line:`Theorem 3`.
PCPO assumes that: When policy tries to minimize the negative reward objective function
:math:`f: \mathbb{R}^n \rightarrow \mathbb{R}` .
The function :math:`f` is :math:`L`-smooth and twice continuously
differentiable over the closed and convex constraint set.

.. _Theorem 3:

.. card::
    :class-header: sd-bg-info sd-text-white sd-font-weight-bold
    :class-card: sd-outline-info  sd-rounded-1
    :class-footer: sd-font-weight-bold
    :link: pcpo-theorem3-proof
    :link-type: ref

    Theorem 3
    ^^^
    Let :math:`\eta\doteq \sqrt{\frac{2\delta}{g^{T}\boldsymbol{H}^{-1}g}}` in :eq:`pcpo-eq-5`, where :math:`\delta` is the step size for reward improvement, :math:`g` is the gradient of :math:`f`, and :math:`\boldsymbol{H}` is the Fisher information matrix.
    Let :math:`\sigma_\mathrm{max}(\boldsymbol{H})` be the largest singular value of :math:`\boldsymbol{H}`, and :math:`a` be the gradient of cost advantage function in :eq:`pcpo-eq-5`.
    Then PCPO with :math:`KL` divergence projection converges to a stationary point either inside the constraint set or in the boundary of the constraint set.
    In the latter case, the Lagrangian constraint :math:`g=-\alpha a, \alpha\geq0` holds.
    Moreover, at step :math:`k+1` the objective value satisfies

    .. math::
        :label: pcpo-eq-9

        f(\boldsymbol{\theta}_{k+1})\leq f(\boldsymbol{\theta}_{k})+||\boldsymbol{\theta}_{k+1}-\boldsymbol{\theta}_{k}||^2_{-\frac{1}{\eta}\boldsymbol{H}+\frac{L}{2}\boldsymbol{I}}.

    PCPO with :math:`L2`  norm projection converges to a stationary point either inside the constraint set or in the boundary of the constraint set.
    In the latter case, the Lagrangian constraint :math:`\boldsymbol{H}^{-1}g=-\alpha a, \alpha\geq0` holds.
    If :math:`\sigma_\mathrm{max}(\boldsymbol{H})\leq1,` then a step :math:`k+1` objective value satisfies.

    .. math::
        :label: pcpo-eq-10

        f(\boldsymbol{\theta}_{k+1})\leq f(\boldsymbol{\theta}_{k})+(\frac{L}{2}-\frac{1}{\eta})||\boldsymbol{\theta}_{k+1}-\boldsymbol{\theta}_{k}||^2_2.
    +++
    The proof of the :bdg-info-line:`Theorem 3` can be seen in the :bdg-info:`Appendix`, click on this :bdg-info-line:`card` to jump to view.

:bdg-info-line:`Theorem 3` shows that in the stationary point, :math:`g` is a
line that points to the opposite direction of :math:`a`.

Further, the improvement of the objective value is affected by the singular
value of the Fisher information matrix.
Specifically, the objective of :math:`KL` divergence projection decreases when
:math:`\frac{L\eta}{2}\boldsymbol{I}\prec\boldsymbol{H},` implying that
:math:`\sigma_\mathrm{min}(\boldsymbol{H})> \frac{L\eta}{2}`.
And the objective of :math:`L2` norm projection decreases when
:math:`\eta<\frac{2}{L},` implying that condition number of
:math:`\boldsymbol{H}` is upper bounded:
:math:`\frac{\sigma_\mathrm{max}(\boldsymbol{H})}{\sigma_\mathrm{min}(\boldsymbol{H})}<\frac{2||g||^2_2}{L^2\delta}`.
By observing the singular values of the Fisher information matrix, we can dynamically select the suitable projection, enabling us to achieve objective improvement.

In the Appendix, we further use an example to compare the
optimization trajectories and stationary points of :math:`KL` divergence and
:math:`L2` norm projections.

------

.. _pcpo-code-with-omnisafe:

Code with OmniSafe
~~~~~~~~~~~~~~~~~~

Quick start
"""""""""""


.. card::
    :class-header: sd-bg-success sd-text-white sd-font-weight-bold
    :class-card: sd-outline-success  sd-rounded-1 sd-font-weight-bold
    :class-footer: sd-font-weight-bold

    Run PCPO in OmniSafe
    ^^^
    Here are 3 ways to run PCPO in OmniSafe:

    * Run Agent from preset yaml file
    * Run Agent from custom config dict
    * Run Agent from custom terminal config

    .. tab-set::

        .. tab-item:: Yaml file style

            .. code-block:: python
                :linenos:

                import omnisafe


                env_id = 'SafetyPointGoal1-v0'

                agent = omnisafe.Agent('PCPO', env_id)
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

                agent = omnisafe.Agent('PCPO', env_id, custom_cfgs=custom_cfgs)
                agent.learn()


        .. tab-item:: Terminal config style

            We use ``train_policy.py`` as the entrance file. You can train the agent with PCPO simply using ``train_policy.py``, with arguments about PCPO and environments does the training.
            For example, to run PCPO in SafetyPointGoal1-v0 , with 1 torch thread, seed 0 and single environment, you can use the following command:

            .. code-block:: bash
                :linenos:

                cd examples
                python train_policy.py --algo PCPO --env-id SafetyPointGoal1-v0 --parallel 1 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 1

------

Architecture of functions
"""""""""""""""""""""""""

- ``PCPO.learn()``

  - ``PCPO._env.rollout()``
  - ``PCPO._update()``

    - ``PCPO._buf.get()``
    - ``PCPO._update_actor()``

      - ``PCPO._fvp()``
      - ``conjugate_gradients()``
      - ``PCPO._cpo_search_step()``

    - ``PCPO._update_cost_critic()``
    - ``PCPO._update_reward_critic()``

------

Documentation of algorithm specific functions
"""""""""""""""""""""""""""""""""""""""""""""

.. tab-set::

    .. tab-item:: pcpo._update_actor()

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success  sd-rounded-1 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            pcpo._update_actor()
            ^^^
            Update the policy network, flowing the next steps:

            (1) Get the policy reward performance gradient g (flat as vector)

            .. code-block:: python
                :linenos:

                theta_old = get_flat_params_from(self._actor_critic.actor)
                self._actor_critic.actor.zero_grad()
                loss_reward, info = self._loss_pi(obs, act, logp, adv_r)
                loss_reward_before = distributed.dist_avg(loss_reward).item()
                p_dist = self._actor_critic.actor(obs)

            (2) Get the policy cost performance gradient b (flat as vector)

            .. code-block:: python
                :linenos:

                self._actor_critic.zero_grad()
                loss_cost = self._loss_pi_cost(obs, act, logp, adv_c)
                loss_cost_before = distributed.dist_avg(loss_cost).item()

                loss_cost.backward()
                distributed.avg_grads(self._actor_critic.actor)

                b_grads = get_flat_gradients_from(self._actor_critic.actor)


            (3) Build the Hessian-vector product based on an approximation of the :math:`KL`-divergence, using ``conjugate_gradients``

            .. code-block:: python
                :linenos:

                p = conjugate_gradients(self._fvp, b_grads, self._cfgs.algo_cfgs.cg_iters)
                q = xHx
                r = grads.dot(p)
                s = b_grads.dot(p)

            (4) Determine step direction and apply SGD step after grads where set (By ``adjust_cpo_step_direction()``)

            .. code-block:: python
                :linenos:

                step_direction, accept_step = self._cpo_search_step(
                    step_direction=step_direction,
                    grads=grads,
                    p_dist=p_dist,
                    obs=obs,
                    act=act,
                    logp=logp,
                    adv_r=adv_r,
                    adv_c=adv_c,
                    loss_reward_before=loss_reward_before,
                    loss_cost_before=loss_cost_before,
                    total_steps=200,
                    violation_c=ep_costs,
                )

            (5) Update actor network parameters

            .. code-block:: python
                :linenos:

                theta_new = theta_old + step_direction
                set_param_values_to_model(self._actor_critic.actor, theta_new)

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

                The following configs are specific to PCPO algorithm.

                - cg_damping (float): Damping coefficient for conjugate gradient.
                - cg_iters (int): Number of iterations for conjugate gradient.
                - fvp_sample_freq (int): Frequency of sampling for Fisher vector product.

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

-  `Constrained Policy Optimization <https://arxiv.org/abs/1705.10528>`__
-  `Projection-Based Constrained Policy Optimization <https://arxiv.org/pdf/2010.03152.pdf>`__
-  `Trust Region Policy Optimization <https://arxiv.org/abs/1502.05477>`__
-  `Constrained Markov Decision Processes <https://www.semanticscholar.org/paper/Constrained-Markov-Decision-Processes-Altman/3cc2608fd77b9b65f5bd378e8797b2ab1b8acde7>`__

.. _`pcpo-performance-bound-proof`:

.. _`convex-programming`:

Appendix
--------

:bdg-ref-info-line:`Click here to jump to PCPO Theorem<performance bound>`  :bdg-ref-success-line:`Click here to jump to Code with OmniSafe<pcpo-code-with-omnisafe>`

Proof of Theorem 2
~~~~~~~~~~~~~~~~~~

To prove the policy performance bound when the current policy is infeasible
(constraint-violating), we first prove two lemmas of the :math:`KL` divergence
between :math:`\pi_{k}` and :math:`\pi_{k+1}`.
We then prove the main theorem for the worst-case performance degradation.

.. tab-set::

    .. tab-item:: Lemma 1
        :sync: key1

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Lemma 1
            ^^^
            If the current policy :math:`\pi_{k}` satisfies the constraint, the constraint set is closed and convex, and the :math:`KL` divergence constraint for the first step is :math:`\underset{s\sim d_{\pi_{k}}}{\mathbb{E}}\big[\mathrm{KL}(\pi_{k+\frac{1}{2}} ||\pi_{k})[s]\big]\leq \delta`, where :math:`\delta` is the step size in the reward improvement step, then under :math:`KL` divergence projection, we have

            .. math::
                :label: pcpo-eq-11

                \underset{s\sim d_{\pi_{k}}}{\mathbb{E}}\big[\mathrm{KL}(\pi_{k+1} ||\pi_{k})[s]\big]\leq \delta.


    .. tab-item:: Lemma 2
        :sync: key2

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Lemma 2
            ^^^
            If the current policy :math:`\pi_{k}` violates the constraint, the constraint set is closed and convex, the :math:`KL` divergence constraint for the first step is :math:`\underset{s\sim d_{\pi_{k}}}{\mathbb{E}}\big[\mathrm{KL}(\pi_{k+\frac{1}{2}} ||\pi_{k})[s]\big]\leq \delta`.
            where :math:`\delta` is the step size in the reward improvement step, then under the :math:`KL` divergence projection, we have

            .. math::
                :label: pcpo-eq-12

                \underset{s\sim d_{\pi_{k}}}{\mathbb{E}}\big[\mathrm{KL}(\pi_{k+1} ||\pi_{k})[s]\big]\leq \delta+{b^+}^2\alpha_\mathrm{KL},

            where :math:`\alpha_\mathrm{KL} \doteq \frac{1}{2a^T\boldsymbol{H}^{-1}a}`, :math:`a` is the gradient of the cost advantage function, :math:`\boldsymbol{H}` is the Hessian of the :math:`KL` divergence constraint, and :math:`b^+\doteq\max(0,J^{C}(\pi_k)-h)`.

.. _pcpo-eq-11:

.. tab-set::

    .. tab-item:: Proof of Lemma 1
        :sync: key1

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Proof of Lemma 1
            ^^^
            By the Bregman divergence projection inequality, :math:`\pi_{k}` being in the constraint set, and :math:`\pi_{k+1}` being the projection of the :math:`\pi_{k+\frac{1}{2}}` onto the constraint set, we have

            .. math::
                :label: pcpo-eq-13


                &\underset{s\sim d_{\pi_{k}}}{\mathbb{E}}\big[\mathrm{KL}(\pi_{k} ||\pi_{k+\frac{1}{2}})[s]\big]\geq
                \underset{s\sim d_{\pi_{k}}}{\mathbb{E}}\big[\mathrm{KL}(\pi_{k}||\pi_{k+1})[s]\big] \\
                &+
                \underset{s\sim d_{\pi_{k}}}{\mathbb{E}}\big[\mathrm{KL}(\pi_{k+1} ||\pi_{k+\frac{1}{2}})[s]\big]\\
                &\Rightarrow\delta\geq
                \underset{s\sim d_{\pi_{k}}}{\mathbb{E}}\big[\mathrm{KL}(\pi_{k} ||\pi_{k+\frac{1}{2}})[s]\big]\geq
                \underset{s\sim d_{\pi_{k}}}{\mathbb{E}}\big[\mathrm{KL}(\pi_{k}||\pi_{k+1})[s]\big].


            The derivation uses the fact that :math:`KL` divergence is always greater than zero.
            We know that :math:`KL` divergence is asymptotically symmetric when updating the policy within a local neighborhood.
            Thus, we have

            .. math::
                :label: pcpo-eq-14

                \delta\geq
                \underset{s\sim d_{\pi_{k}}}{\mathbb{E}}\big[\mathrm{KL}(\pi_{k+\frac{1}{2}} ||\pi_{k})[s]\big]\geq
                \underset{s\sim d_{\pi_{k}}}{\mathbb{E}}\big[\mathrm{KL}(\pi_{k+1}||\pi_{k})[s]\big].

    .. tab-item:: Proof of Lemma 2
      :sync: key2

      .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Proof of Lemma 2
            ^^^
            We define the sub-level set of cost constraint functions for the current infeasible policy :math:`\pi_k`:

            .. math::
                :label: pcpo-eq-15

                L_{\pi_k}=\{\pi~|~J^{C}(\pi_{k})+ \mathbb{E}_{\substack{s\sim d_{\pi_{k}}\\ a\sim \pi}}[A_{\pi_k}^{C}(s,a)]\leq J^{C}(\pi_{k})\}.

            This implies that the current policy :math:`\pi_k` lies in :math:`L_{\pi_k}`, and :math:`\pi_{k+\frac{1}{2}}` is projected onto the constraint set: :math:`\{\pi~|~J^{C}(\pi_{k})+ \mathbb{E}_{\substack{s\sim d_{\pi_{k}}\\ a\sim \pi}}[A_{\pi_k}^{C}(s,a)]\leq h\}`.
            Next, we define the policy :math:`\pi_{k+1}^l` as the projection of :math:`\pi_{k+\frac{1}{2}}` onto :math:`L_{\pi_k}`.

            For these three polices :math:`\pi_k, \pi_{k+1}` and :math:`\pi_{k+1}^l`, with :math:`\varphi(x)\doteq\sum_i x_i\log x_i`, we have

            .. math::
                :label: pcpo-eq-16

                \delta &\geq  \underset{s\sim d_{\pi_{k}}}{\mathbb{E}}\big[\mathrm{KL}(\pi_{k+1}^l ||\pi_{k})[s]\big]
                \\&=\underset{s\sim d_{\pi_{k}}}{\mathbb{E}}\big[\mathrm{KL}(\pi_{k+1} ||\pi_{k})[s]\big] -\underset{s\sim d_{\pi_{k}}}{\mathbb{E}}\big[\mathrm{KL} (\pi_{k+1} ||\pi_{k+1}^l)[s]\big]\\
                &+\underset{s\sim d_{\pi_{k}}}{\mathbb{E}}\big[(\nabla\varphi(\pi_k)-\nabla\varphi(\pi_{k+1}^{l}))^T(\pi_{k+1}-\pi_{k+1}^l)[s]\big] \nonumber \\



                \Rightarrow \underset{s\sim d_{\pi_{k}}}{\mathbb{E}}\big[\mathrm{KL} (\pi_{k+1} ||\pi_{k})[s]\big]&\leq \delta + \underset{s\sim d_{\pi_{k}}}{\mathbb{E}}\big[\mathrm{KL} (\pi_{k+1} ||\pi_{k+1}^l)[s]\big]\\
                &- \underset{s\sim d_{\pi_{k}}}{\mathbb{E}}\big[(\nabla\varphi(\pi_k)-\nabla\varphi(\pi_{k+1}^{l}))^T(\pi_{k+1}-\pi_{k+1}^l)[s]\big].


            The inequality :math:`\underset{s\sim d_{\pi_{k}}}{\mathbb{E}}\big[\mathrm{KL} (\pi_{k+1}^l ||\pi_{k})[s]\big]\leq\delta` comes from that :math:`\pi_{k}`, :math:`\pi_{k+1}^l` are in :math:`L_{\pi_k}`, and :bdg-info-line:`Lemma 1`.

            If the constraint violation of the current policy :math:`\pi_k` and :math:`b^+` are small, :math:`\underset{s\sim d_{\pi_{k}}}{\mathbb{E}}\big[\mathrm{KL} (\pi_{k+1} ||\pi_{k+1}^l)[s]\big]` can be approximated by the second order expansion.
            By the update rule in :eq:`pcpo-eq-5`, we have

            .. math::
                :label: pcpo-eq-17

                \underset{s\sim d_{\pi_{k}}}{\mathbb{E}}\big[\mathrm{KL}(\pi_{k+1} ||\pi_{k+1}^l)[s]\big] &\approx \frac{1}{2}(\boldsymbol{\theta}_{k+1}-\boldsymbol{\theta}_{k+1}^l)^{T}\boldsymbol{H}(\boldsymbol{\theta}_{k+1}-\boldsymbol{\theta}_{k+1}^l)\\
                &=\frac{1}{2} \Big(\frac{b^+}{a^T\boldsymbol{H}^{-1}a}\boldsymbol{H}^{-1}a\Big)^T\boldsymbol{H}\Big(\frac{b^+}{a^T\boldsymbol{H}^{-1}a}\boldsymbol{H}^{-1}a\Big)\\
                &=\frac{{b^+}^2}{2a^T\boldsymbol{H}^{-1}a}\\
                &={b^+}^2\alpha_\mathrm{KL},


            where :math:`\alpha_\mathrm{KL} \doteq \frac{1}{2a^T\boldsymbol{H}^{-1}a}.`

            And since :math:`\delta` is small, we have :math:`\nabla\varphi(\pi_k)-\nabla\varphi(\pi_{k+1}^{l})\approx \mathbf{0}` given :math:`s`.
            Thus, the third term in :eq:`pcpo-eq-8` can be eliminated.

            Combining :eq:`pcpo-eq-8` and :eq:`pcpo-eq-13`, we have :math:`[
            \underset{s\sim d_{\pi_{k}}}{\mathbb{E}}\big[\mathrm{KL}(\pi_{k+1}||\pi_{k})[s]\big]\leq \delta+{b^+}^2\alpha_\mathrm{KL}.]`


Now we use :bdg-info-line:`Lemma 2` to prove the :bdg-info-line:`Theorem 2`.
Following the same proof in :bdg-ref-info-line:`Theorem 1<cards-clickable>`, we
complete the proof.

.. _`appendix_proof_theorem_3`:

.. _`pcpo-theorem3-proof`:

Proof of Analytical Solution to PCPO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. card::
    :class-header: sd-bg-info sd-text-white sd-font-weight-bold
    :class-card: sd-outline-info  sd-rounded-1

    Analytical Solution to PCPO
    ^^^
    Consider the PCPO problem. In the first step, we optimize the reward:

    .. math::
        :label: pcpo-eq-18

        \boldsymbol{\theta}_{k+\frac{1}{2}} = & \arg \underset{\boldsymbol{\theta}}{\min} g^{T}(\boldsymbol{\theta}-\boldsymbol{\theta}_{k}) \\
        \text{s.t.}\quad&\frac{1}{2}(\boldsymbol{\theta}-\boldsymbol{\theta}_{k})^{T}\boldsymbol{H}(\boldsymbol{\theta}-\boldsymbol{\theta}_{k})\leq \delta,


    and in the second step, we project the policy onto the constraint set:

    .. math::
        :label: pcpo-eq-19

        \boldsymbol{\theta}_{k+1} = &\arg\underset{\boldsymbol{\theta}}{ \min} \frac{1}{2}(\boldsymbol{\theta}-{\boldsymbol{\theta}}_{k+\frac{1}{2}})^{T}\boldsymbol{L}(\theta-{\boldsymbol{\theta}}_{k+\frac{1}{2}}) \\
        \text{s.t.}\quad &a^{T}(\boldsymbol{\theta}-\boldsymbol{\theta}_{k})+b\leq 0,


    where :math:`g, a, \boldsymbol{\theta} \in \mathbb{R}^n, b, \delta\in \mathbb{R}, \delta>0,` and :math:`\boldsymbol{H},\boldsymbol{L}\in \mathbb{R}^{n\times n}, \boldsymbol{L}=\boldsymbol{H}`, if using the :math:`KL` divergence projection, and :math:`\boldsymbol{L}=\boldsymbol{I}` if using the :math:`L2`  norm projection.
    When there is at least one strictly feasible point, the optimal solution satisfies

    .. math::
        :label: pcpo-eq-20

        \boldsymbol{\theta}_{k+1}&=\boldsymbol{\theta}_{k}+\sqrt{\frac{2\delta}{g^T\boldsymbol{H}^{-1}g}}\boldsymbol{H}^{-1}g\nonumber\\
        &-\max(0,\frac{\sqrt{\frac{2\delta}{g^T\boldsymbol{H}^{-1}g}}a^{T}\boldsymbol{H}^{-1}g+b}{a^T\boldsymbol{L}^{-1}a})\boldsymbol{L}^{-1}a


    assuming that :math:`\boldsymbol{H}` is invertible to get a unique solution.

    .. dropdown:: Proof of Analytical Solution to PCPO (Click here)
        :color: info
        :class-body: sd-outline-info

        For the first problem, since :math:`\boldsymbol{H}` is the Fisher Information matrix, which automatically guarantees it is positive semi-definite, it is a convex program with quadratic inequality constraints.
        Hence if the primal problem has a feasible point, then `Slater's condition <https://en.wikipedia.org/wiki/Slater%27s_condition>`_ is satisfied and strong duality holds.
        Let :math:`\boldsymbol{\theta}^{*}` and :math:`\lambda^*` denote the solutions to the primal and dual problems, respectively.
        In addition, the primal objective function is continuously differentiable.
        Hence the `Karush-Kuhn-Tucker (KKT) conditions <https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions>`_ are necessary and sufficient for the optimality of :math:`\boldsymbol{\theta}^{*}` and :math:`\lambda^*.`
        We now form the Lagrangian:

        .. math:: \mathcal{L}(\boldsymbol{\theta},\lambda)=-g^{T}(\boldsymbol{\theta}-\boldsymbol{\theta}_{k})+\lambda\Big(\frac{1}{2}(\boldsymbol{\theta}-\boldsymbol{\theta}_{k})^{T}\boldsymbol{H}(\boldsymbol{\theta}-\boldsymbol{\theta}_{k})- \delta\Big).

        And we have the following KKT conditions:

        .. _`pcpo-eq-13`:

        .. math::
            :label: pcpo-eq-22

            -g + \lambda^*\boldsymbol{H}\boldsymbol{\theta}^{*}-\lambda^*\boldsymbol{H}\boldsymbol{\theta}_{k}=0~~~~&~~~\nabla_\boldsymbol{\theta}\mathcal{L}(\boldsymbol{\theta}^{*},\lambda^{*})=0 \\
            \frac{1}{2}(\boldsymbol{\theta}^{*}-\boldsymbol{\theta}_{k})^{T}\boldsymbol{H}(\boldsymbol{\theta}^{*}-\boldsymbol{\theta}_{k})- \delta=0~~~~&~~~\nabla_\lambda\mathcal{L}(\boldsymbol{\theta}^{*},\lambda^{*})=0 \\
            \frac{1}{2}(\boldsymbol{\theta}^{*}-\boldsymbol{\theta}_{k})^{T}\boldsymbol{H}(\boldsymbol{\theta}^{*}-\boldsymbol{\theta}_{k})-\delta\leq0~~~~&~~~\text{primal constraints}\label{KKT_3}\\
            \lambda^*\geq0~~~~&~~~\text{dual constraints}\\
            \lambda^*\Big(\frac{1}{2}(\boldsymbol{\theta}^{*}-\boldsymbol{\theta}_{k})^{T}\boldsymbol{H}(\boldsymbol{\theta}^{*}-\boldsymbol{\theta}_{k})-\delta\Big)=0~~~~&~~~\text{complementary slackness}


        By :eq:`pcpo-eq-22`, we have :math:`\boldsymbol{\theta}^{*}=\boldsymbol{\theta}_{k}+\frac{1}{\lambda^*}\boldsymbol{H}^{-1}g`.
        And :math:`\lambda^*=\sqrt{\frac{g^T\boldsymbol{H}^{-1}g}{2\delta}}` .
        Hence we have our optimal solution:

        .. _`pcpo-eq-18`:

        .. math::
            :label: pcpo-eq-23

            \boldsymbol{\theta}_{k+\frac{1}{2}}=\boldsymbol{\theta}^{*}=\boldsymbol{\theta}_{k}+\sqrt{\frac{2\delta}{g^T\boldsymbol{H}^{-1}g}}\boldsymbol{H}^{-1}g

        Following the same reasoning, we now form the Lagrangian of the second problem:

        .. math::
            :label: pcpo-eq-24

            \mathcal{L}(\boldsymbol{\theta},\lambda)=\frac{1}{2}(\boldsymbol{\theta}-{\boldsymbol{\theta}}_{k+\frac{1}{2}})^{T}\boldsymbol{L}(\boldsymbol{\theta}-{\boldsymbol{\theta}}_{k+\frac{1}{2}})+\lambda(a^T(\boldsymbol{\theta}-\boldsymbol{\theta}_{k})+b)


        And we have the following KKT conditions:

        .. _`pcpo-eq-20`:

        .. math::
            :label: pcpo-eq-25

            \boldsymbol{L}\boldsymbol{\theta}^*-\boldsymbol{L}\boldsymbol{\theta}_{k+\frac{1}{2}}+\lambda^*a=0~~~~&~~~\nabla_\boldsymbol{\theta}\mathcal{L}(\boldsymbol{\theta}^{*},\lambda^{*})=0   \\
                a^T(\boldsymbol{\theta}^*-\boldsymbol{\theta}_{k})+b=0~~~~&~~~\nabla_\lambda\mathcal{L}(\boldsymbol{\theta}^{*},\lambda^{*})=0   \\
                a^T(\boldsymbol{\theta}^*-\boldsymbol{\theta}_{k})+b\leq0~~~~&~~~\text{primal constraints}  \\
                \lambda^*\geq0~~~~&~~~\text{dual constraints}  \\
                \lambda^*(a^T(\boldsymbol{\theta}^*-\boldsymbol{\theta}_{k})+b)=0~~~~&~~~\text{complementary slackness}


        By :eq:`pcpo-eq-25`, we have :math:`\boldsymbol{\theta}^{*}=\boldsymbol{\theta}_{k+1}+\lambda^*\boldsymbol{L}^{-1}a`.
        And by solving :eq:`pcpo-eq-25`, we have :math:`\lambda^*=\max(0,\\ \frac{a^T(\boldsymbol{\theta}_{k+\frac{1}{2}}-\boldsymbol{\theta}_{k})+b}{a\boldsymbol{L}^{-1}a})`.
        Hence we have our optimal solution:

        .. _`pcpo-eq-25`:

        .. math::
            :label: pcpo-eq-26

            \boldsymbol{\theta}_{k+1}=\boldsymbol{\theta}^{*}=\boldsymbol{\theta}_{k+\frac{1}{2}}-\max(0,\frac{a^T(\boldsymbol{\theta}_{k+\frac{1}{2}}-\boldsymbol{\theta}_{k})+b}{a^T\boldsymbol{L}^{-1}a^T})\boldsymbol{L}^{-1}a

        we have

        .. math::
            :label: pcpo-eq-27

            \boldsymbol{\theta}_{k+1}&=\boldsymbol{\theta}_{k}+\sqrt{\frac{2\delta}{g^T\boldsymbol{H}^{-1}g}}\boldsymbol{H}^{-1}g\\
            &-\max(0,\frac{\sqrt{\frac{2\delta}{g^T\boldsymbol{H}^{-1}g}}a^{T}\boldsymbol{H}^{-1}g+b}{a^T\boldsymbol{L}^{-1}a})\boldsymbol{L}^{-1}a


Proof of Theorem 3
~~~~~~~~~~~~~~~~~~

For our analysis, we make the following assumptions: we minimize the negative
reward objective function :math:`f: \mathbb{R}^n \rightarrow \mathbb{R}`
(We follow the
convention of the literature that authors typically minimize the objective
function).
The function :math:`f` is :math:`L`-smooth and twice continuously
differentiable over the closed and convex constraint set :math:`\mathcal{C}`.
We have the following :bdg-info-line:`Lemma 3` to characterize the projection
and for the proof of :bdg-info-line:`Theorem 3`

.. card::
    :class-header: sd-bg-info sd-text-white sd-font-weight-bold
    :class-card: sd-outline-info  sd-rounded-1

    Lemma 3
    ^^^
    For any :math:`\boldsymbol{\theta}`, :math:`\boldsymbol{\theta}^{*}=\mathrm{Proj}^{\boldsymbol{L}}_{\mathcal{C}}(\boldsymbol{\theta})` if and only if :math:`(\boldsymbol{\theta}-\boldsymbol{\theta}^*)^T\boldsymbol{L}(\boldsymbol{\theta}'-\boldsymbol{\theta}^*)\leq0, \forall\boldsymbol{\theta}'\in\mathcal{C}`,
    where :math:`\mathrm{Proj}^{\boldsymbol{L}}_{\mathcal{C}}(\boldsymbol{\theta})\doteq \underset{\boldsymbol{\theta}' \in \mathrm{C}}{\arg \min}||\boldsymbol{\theta}-\boldsymbol{\theta}'||^2_{\boldsymbol{L}}` and :math:`\boldsymbol{L}=\boldsymbol{H}` if using the :math:`KL` divergence projection, and :math:`\boldsymbol{L}=\boldsymbol{I}` if using the :math:`L2` norm projection.

    .. dropdown:: Proof of Lemma 3 (Click here)
        :color: info
        :class-body: sd-outline-info

        :math:`(\Rightarrow)` Let
        :math:`\boldsymbol{\theta}^{*}=\mathrm{Proj}^{\boldsymbol{L}}_{\mathcal{C}}(\boldsymbol{\theta})`
        for a given :math:`\boldsymbol{\theta} \not\in\mathcal{C},`
        :math:`\boldsymbol{\theta}'\in\mathcal{C}` be such that
        :math:`\boldsymbol{\theta}'\neq\boldsymbol{\theta}^*,` and :math:`\alpha\in(0,1).` Then we have

        .. _`pcpo-eq-26`:

        .. math::
            :label: pcpo-eq-28

            \label{eq:appendix_lemmaD1_0}
                \left\|\boldsymbol{\theta}-\boldsymbol{\theta}^*\right\|_L^2
                & \leq\left\|\boldsymbol{\theta}-\left(\boldsymbol{\theta}^*+\alpha\left(\boldsymbol{\theta}^{\prime}-\boldsymbol{\theta}^*\right)\right)\right\|_L^2 \\
                &=\left\|\boldsymbol{\theta}-\boldsymbol{\theta}^*\right\|_L^2+\alpha^2\left\|\boldsymbol{\theta}^{\prime}-\boldsymbol{\theta}^*\right\|_{\boldsymbol{L}}^2\\
                ~~~~ &-2\alpha\left(\boldsymbol{\theta}-\boldsymbol{\theta}^*\right)^T \boldsymbol{L}\left(\boldsymbol{\theta}^{\prime}-\boldsymbol{\theta}^*\right) \\
                & \Rightarrow\left(\boldsymbol{\theta}-\boldsymbol{\theta}^*\right)^T \boldsymbol{L}\left(\boldsymbol{\theta}^{\prime}-\boldsymbol{\theta}^*\right) \leq \frac{\alpha}{2}\left\|\boldsymbol{\theta}^{\prime}-\boldsymbol{\theta}^*\right\|_{\boldsymbol{L}}^2


        Since the right-hand side of :eq:`pcpo-eq-28` can be made arbitrarily small for a given :math:`\alpha`, and hence we have:

        .. math::
            :label: pcpo-eq-29

            (\boldsymbol{\theta}-\boldsymbol{\theta}^*)^T\boldsymbol{L}(\boldsymbol{\theta}'-\boldsymbol{\theta}^*)\leq0, \forall\boldsymbol{\theta}'\in\mathcal{C}.

        Let :math:`\boldsymbol{\theta}^*\in\mathcal{C}` be such that :math:`(\boldsymbol{\theta}-\boldsymbol{\theta}^*)^T\boldsymbol{L}(\boldsymbol{\theta}'-\boldsymbol{\theta}^*)\leq0, \forall\boldsymbol{\theta}'\in\mathcal{C}`.
        We show that :math:`\boldsymbol{\theta}^*` must be the optimal solution.
        Let :math:`\boldsymbol{\theta}'\in\mathcal{C}` and :math:`\boldsymbol{\theta}'\neq\boldsymbol{\theta}^*`.
        Then we have

        .. math::
            :label: pcpo-eq-30

            \begin{split}
            &\left\|\boldsymbol{\theta}-\boldsymbol{\theta}^{\prime}\right\|_L^2-\left\|\boldsymbol{\theta}-\boldsymbol{\theta}^*\right\|_L^2\\ &=\left\|\boldsymbol{\theta}-\boldsymbol{\theta}^*+\boldsymbol{\theta}^*-\boldsymbol{\theta}^{\prime}\right\|_L^2-\left\|\boldsymbol{\theta}-\boldsymbol{\theta}^*\right\|_L^2 \\
            &=\left\|\boldsymbol{\theta}-\boldsymbol{\theta}^*\right\|_L^2+\left\|\boldsymbol{\theta}^{\prime}-\boldsymbol{\theta}^*\right\|_L^2-2\left(\boldsymbol{\theta}-\boldsymbol{\theta}^*\right)^T \boldsymbol{L}\left(\boldsymbol{\theta}^{\prime}-\boldsymbol{\theta}^*\right)\\
            &~~~~-\left\|\boldsymbol{\theta}-\boldsymbol{\theta}^*\right\|_{\boldsymbol{L}}^2 \\
            &>0 \\
            &\Rightarrow\left\|\boldsymbol{\theta}-\boldsymbol{\theta}^{\prime}\right\|_L^2 >\left\|\boldsymbol{\theta}-\boldsymbol{\theta}^*\right\|_L^2 .
            \end{split}

        Hence, :math:`\boldsymbol{\theta}^*` is the optimal solution to the optimization problem, and :math:`\boldsymbol{\theta}^*=\mathrm{Proj}^{\boldsymbol{L}}_{\mathcal{C}}(\boldsymbol{\theta})`.

Based on :bdg-info-line:`Lemma 3` we have the proof of following :bdg-info-line:`Theorem 3`.

.. card::
    :class-header: sd-bg-info sd-text-white sd-font-weight-bold
    :class-card: sd-outline-info  sd-rounded-1

    Theorem 3 (Stationary Points of PCPO with the :math:`KL` divergence and :math:`L2` Norm Projections)
    ^^^
    Let :math:`\eta\doteq \sqrt{\frac{2\delta}{g^{T}\boldsymbol{H}^{-1}g}}` in :eq:`pcpo-eq-5`, where :math:`\delta` is the step size for reward improvement, :math:`g` is the gradient of :math:`f`, :math:`\boldsymbol{H}` is the Fisher information matrix.
    Let :math:`\sigma_\mathrm{max}(\boldsymbol{H})` be the largest singular value of :math:`\boldsymbol{H}`, and :math:`a` be the gradient of cost advantage function in :eq:`pcpo-eq-5`.
    Then PCPO with the :math:`KL` divergence projection converges to stationary points with :math:`g\in-a` (i.e., the gradient of :math:`f` belongs to the negative gradient of the cost advantage function).
    The objective value changes by

    .. math::
        :label: pcpo-eq-31

        f(\boldsymbol{\theta}_{k+1})\leq f(\boldsymbol{\theta}_{k})+||\boldsymbol{\theta}_{k+1}-\boldsymbol{\theta}_{k}||^2_{-\frac{1}{\eta}\boldsymbol{H}+\frac{L}{2}\boldsymbol{I}}

    PCPO with the :math:`L2` norm projection converges to stationary points with :math:`\boldsymbol{H}^{-1}g\in-a` (i.e., the product of the inverse of :math:`\boldsymbol{H}` and gradient of :math:`f` belongs to the negative gradient of the cost advantage function).
    If :math:`\sigma_\mathrm{max}(\boldsymbol{H})\leq1`, then the objective value changes by

    .. math::
        :label: pcpo-eq-32

        f(\boldsymbol{\theta}_{k+1})\leq f(\boldsymbol{\theta}_{k})+(\frac{L}{2}-\frac{1}{\eta})||\boldsymbol{\theta}_{k+1}-\boldsymbol{\theta}_{k}||^2_2

    .. dropdown:: Proof of Theorem 3 (Click here)
        :color: info
        :class-body: sd-outline-info

        The proof of the theorem is based on working in a Hilbert space and the non-expansive property of the projection.
        We first prove stationary points for PCPO with the :math:`KL` divergence and :math:`L2` norm projections and then prove the change of the objective value.

        When in stationary points :math:`\boldsymbol{\theta}^*`, we have

        .. _`pcpo-eq-29`:

        .. math::
            :label: pcpo-eq-33

            \boldsymbol{\theta}^{*}&=\boldsymbol{\theta}^{*}-\sqrt{\frac{2\delta}{g^T\boldsymbol{H}^{-1}g}}\boldsymbol{H}^{-1}g
            -\max\left(0,\frac{\sqrt{\frac{2\delta}{g^T\boldsymbol{H}^{-1}g}}a^{T}\boldsymbol{H}^{-1}g+b}{a^T\boldsymbol{L}^{-1}a}\right)\boldsymbol{L}^{-1}a\\
            &\Leftrightarrow \sqrt{\frac{2\delta}{g^T\boldsymbol{H}^{-1}g}}\boldsymbol{H}^{-1}g  = -\max(0,\frac{\sqrt{\frac{2\delta}{g^T\boldsymbol{H}^{-1}g}}a^{T}\boldsymbol{H}^{-1}g+b}{a^T\boldsymbol{L}^{-1}a})\boldsymbol{L}^{-1}a\\
            &\Leftrightarrow  \boldsymbol{H}^{-1}g \in -\boldsymbol{L}^{-1}a.
            \label{eq:appendixStationary}


        For the :math:`KL` divergence projection (:math:`\boldsymbol{L}=\boldsymbol{H}`), :eq:`pcpo-eq-33` boils down to :math:`g\in-a`, and for the :math:`L2` norm projection (:math:`\boldsymbol{L}=\boldsymbol{I}`), :eq:`pcpo-eq-33` is equivalent to :math:`\boldsymbol{H}^{-1}g\in-a`.

        Now we prove the second part of the theorem. Based on :bdg-info-line:`Lemma 3`, for the :math:`KL` divergence projection, we have

        .. _`pcpo-eq-30`:

        .. math::
            :label: pcpo-eq-34

            \label{eq:appendix_converge_0}
            \left(\boldsymbol{\theta}_k-\boldsymbol{\theta}_{k+1}\right)^T \boldsymbol{H}\left(\boldsymbol{\theta}_k-\eta \boldsymbol{H}^{-1} \boldsymbol{g}-\boldsymbol{\theta}_{k+1}\right) \leq 0 \\
            \Rightarrow \boldsymbol{g}^T\left(\boldsymbol{\theta}_{k+1}-\boldsymbol{\theta}_k\right) \leq-\frac{1}{\eta}\left\|\boldsymbol{\theta}_{k+1}-\boldsymbol{\theta}_k\right\|_{\boldsymbol{H}}^2


        By :eq:`pcpo-eq-34`, and :math:`L`-smooth continuous function :math:`f,` we have

        .. math::
            :label: pcpo-eq-35

            f\left(\boldsymbol{\theta}_{k+1}\right) & \leq f\left(\boldsymbol{\theta}_k\right)+\boldsymbol{g}^T\left(\boldsymbol{\theta}_{k+1}-\boldsymbol{\theta}_k\right)+\frac{L}{2}\left\|\boldsymbol{\theta}_{k+1}-\boldsymbol{\theta}_k\right\|_2^2 \\
            & \leq f\left(\boldsymbol{\theta}_k\right)-\frac{1}{\eta}\left\|\boldsymbol{\theta}_{k+1}-\boldsymbol{\theta}_k\right\|_{\boldsymbol{H}}^2+\frac{L}{2}\left\|\boldsymbol{\theta}_{k+1}-\boldsymbol{\theta}_k\right\|_2^2 \\
            &=f\left(\boldsymbol{\theta}_k\right)+\left(\boldsymbol{\theta}_{k+1}-\boldsymbol{\theta}_k\right)^T\left(-\frac{1}{\eta} \boldsymbol{H}+\frac{L}{2} \boldsymbol{I}\right)\left(\boldsymbol{\theta}_{k+1}-\boldsymbol{\theta}_k\right) \\
            &=f\left(\boldsymbol{\theta}_k\right)+\left\|\boldsymbol{\theta}_{k+1}-\boldsymbol{\theta}_k\right\|_{-\frac{1}{\eta} \boldsymbol{H}+\frac{L}{2} \boldsymbol{I}}^2


        For the :math:`L2` norm projection, we have

        .. _`pcpo-eq-31`:

        .. math::
            :label: pcpo-eq-36

            (\boldsymbol{\theta}_{k}-\boldsymbol{\theta}_{k+1})^T(\boldsymbol{\theta}_{k}-\eta\boldsymbol{H}^{-1}g-\boldsymbol{\theta}_{k+1})\leq0\\
            \Rightarrow g^T\boldsymbol{H}^{-1}(\boldsymbol{\theta}_{k+1}-\boldsymbol{\theta}_{k})\leq -\frac{1}{\eta}||\boldsymbol{\theta}_{k+1}-\boldsymbol{\theta}_{k}||^2_2


        By :eq:`pcpo-eq-36`, :math:`L`-smooth continuous function :math:`f`, and if :math:`\sigma_\mathrm{max}(\boldsymbol{H})\leq1`, we have

        .. math::
            :label: pcpo-eq-37

            f(\boldsymbol{\theta}_{k+1})&\leq f(\boldsymbol{\theta}_{k})+g^T(\boldsymbol{\theta}_{k+1}-\boldsymbol{\theta}_{k})+\frac{L}{2}||\boldsymbol{\theta}_{k+1}-\boldsymbol{\theta}_{k}||^2_2 \nonumber\\
            &\leq f(\boldsymbol{\theta}_{k})+(\frac{L}{2}-\frac{1}{\eta})||\boldsymbol{\theta}_{k+1}-\boldsymbol{\theta}_{k}||^2_2.\nonumber


        To see why we need the assumption of :math:`\sigma_\mathrm{max}(\boldsymbol{H})\leq1`, we define :math:`\boldsymbol{H}=\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{U}^T` as the singular value decomposition of :math:`\boldsymbol{H}` with :math:`u_i` being the column vector of :math:`\boldsymbol{U}`.
        Then we have

        .. math::
            :label: pcpo-eq-38

            g^T\boldsymbol{H}^{-1}(\boldsymbol{\theta}_{k+1}-\boldsymbol{\theta}_{k})
            &=g^T\boldsymbol{U}\boldsymbol{\Sigma}^{-1}\boldsymbol{U}^T(\boldsymbol{\theta}_{k+1}-\boldsymbol{\theta}_{k}) \nonumber\\
            &=g^T(\sum_{i}\frac{1}{\sigma_i(\boldsymbol{H})}u_iu_i^T)(\boldsymbol{\theta}_{k+1}-\boldsymbol{\theta}_{k})\nonumber\\
            &=\sum_{i}\frac{1}{\sigma_i(\boldsymbol{H})}g^T(\boldsymbol{\theta}_{k+1}-\boldsymbol{\theta}_{k}).\nonumber


        If we want to have

        .. math::
            :label: pcpo-eq-39

            g^T(\boldsymbol{\theta}_{k+1}-\boldsymbol{\theta}_{k})\leq g^T\boldsymbol{H}^{-1}(\boldsymbol{\theta}_{k+1}-\boldsymbol{\theta}_{k})\leq -\frac{1}{\eta}||\boldsymbol{\theta}_{k+1}-\boldsymbol{\theta}_{k}||^2_2,

        then every singular value :math:`\sigma_i(\boldsymbol{H})` of :math:`\boldsymbol{H}` needs to be smaller than :math:`1`, and hence :math:`\sigma_\mathrm{max}(\boldsymbol{H})\leq1`, which justifies the assumption we use to prove the bound.

        .. hint::

            To make the objective value for PCPO with the :math:`KL` divergence projection improves, the right-hand side of :eq:`pcpo-eq-26` needs to be negative.
            Hence we have :math:`\frac{L\eta}{2}\boldsymbol{I}\prec\boldsymbol{H}`, implying that :math:`\sigma_\mathrm{min}(\boldsymbol{H})>\frac{L\eta}{2}`.
            And to make the objective value for PCPO with the :math:`L2` norm projection improves, the right-hand side of :eq:`pcpo-eq-28` needs to be negative.
            Hence we have :math:`\eta<\frac{2}{L}`, implying that

            .. math::
                :label: pcpo-eq-40

                &\eta = \sqrt{\frac{2\delta}{g^T\boldsymbol{H}^{-1}g}}<\frac{2}{L}\nonumber\\
                \Rightarrow& \frac{2\delta}{g^T\boldsymbol{H}^{-1}g} < \frac{4}{L^2} \nonumber\\
                \Rightarrow& \frac{g^{T}\boldsymbol{H}^{-1}g}{2\delta}>\frac{L^2}{4}\nonumber\\
                \Rightarrow& \frac{L^2\delta}{2}<g^T\boldsymbol{H}^{-1}g\nonumber\\
                &\leq||g||_2||\boldsymbol{H}^{-1}g||_2\nonumber\\
                &\leq||g||_2||\boldsymbol{H}^{-1}||_2||g||_2\nonumber\\
                &=\sigma_\mathrm{max}(\boldsymbol{H}^{-1})||g||^2_2\nonumber\\
                &=\sigma_\mathrm{min}(\boldsymbol{H})||g||^2_2\nonumber\\
                \Rightarrow&\sigma_\mathrm{min}(\boldsymbol{H})>\frac{L^2\delta}{2||g||^2_2}.
                \label{eqnarray}
