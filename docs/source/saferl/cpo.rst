Constrained Policy Optimization
===============================

Quick Facts
-----------

.. card::
    :class-card: sd-outline-info  sd-rounded-1
    :class-body: sd-font-weight-bold

    #. CPO is an :bdg-info-line:`on-policy` algorithm.
    #. CPO can be thought of as being :bdg-info-line:`TRPO in Safe RL areas` .
    #. The OmniSafe implementation of CPO support :bdg-info-line:`parallelization`.
    #. An :bdg-ref-info-line:`API Documentation <cpoapi>` is available for CPO.

CPO Theorem
-----------

Background
~~~~~~~~~~

**Constrained policy optimization (CPO)** is a policy search algorithm for safe
reinforcement learning that guarantees near-constraint satisfaction at each
iteration. CPO builds upon the ideas of TRPO( :doc:`../baserl/trpo`)
to construct surrogate functions that approximate the objectives and
constraints, and it is easy to estimate using samples from the current policy.
CPO provides tighter bounds for policy search using trust regions, making it
the first general-purpose policy search algorithm for safe RL.

.. hint::

    CPO can train neural network policies for high-dimensional control while ensuring that they behave within specified constraints throughout training.

CPO aims to provide an approach for policy search in continuous CMDP. It uses
the result from TRPO and
NPG to derive a policy improvement step that guarantees both an increase in
reward and satisfaction of constraints. Although CPO is slightly inferior in
performance, it offers a solid theoretical foundation for solving constrained
optimization problems in safe RL.

.. hint::

    CPO is very complex in terms of implementation, but OmniSafe provides a highly readable code implementation to help you get up to speed quickly.

------

Optimization Objective
~~~~~~~~~~~~~~~~~~~~~~

In the previous chapters, we introduced that TRPO solves the following
optimization problems:

.. math::
    :label: cpo-eq-1

    &\pi_{k+1}=\arg\max_{\pi \in \Pi_{\boldsymbol{\theta}}}J^R(\pi)\\
    \text{s.t.}\quad &D(\pi,\pi_k)\le\delta


where :math:`\Pi_{\boldsymbol{\theta}} \subseteq \Pi` denotes the set of
parametrized policies with parameters :math:`\boldsymbol{\theta}`, and
:math:`D` is some distance measure.

In local policy search, we additionally require policy iterates to be feasible
for the CMDP. So instead of optimizing over :math:`\Pi_{\boldsymbol{\theta}}`,
CPO optimizes over :math:`\Pi_{\boldsymbol{\theta}} \cap \Pi_{C}`.

.. math::
    :label: cpo-eq-2

    \pi_{k+1} &= \arg\max_{\pi \in \Pi_{\boldsymbol{\theta}}}J^R(\pi)\\
    \text{s.t.} \quad  D(\pi,\pi_k) &\le\delta\\
    J^{C_i}(\pi) &\le d_i\quad i=1,...m



.. hint::

    This update is difficult to implement because it requires evaluating the constraint functions to determine whether a proposed policy :math:`\pi` is feasible.


.. tab-set::

    .. tab-item:: CPO Contribution I

        .. card::
            :class-header: sd-bg-success  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Contribution I
            ^^^
            CPO develops a principled approximation with a particular choice of :math:`D`,
            where the objective and constraints are replaced with surrogate functions.

    .. tab-item:: CPO Contribution II

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card:  sd-outline-success  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Contribution II
            ^^^
            CPO proposes that with those surrogates, the update's worst-case performance
            and worst-case constraint violation can be bounded with values that depend on a
            hyperparameter of the algorithm.

------

Policy Performance Bounds
~~~~~~~~~~~~~~~~~~~~~~~~~

CPO presents the theoretical foundation for its approach, a new bound on the
difference in returns between two arbitrary policies.

The following :bdg-info-line:`Theorem 1` connects the difference in returns (or
constraint costs) between two arbitrary policies to an average divergence
between them.

.. _Theorem 1:

.. card::
    :class-header: sd-bg-info sd-text-white sd-font-weight-bold
    :class-card: sd-outline-info  sd-rounded-1
    :class-footer: sd-font-weight-bold
    :link: cards-clickable
    :link-type: ref

    Theorem 1 (Difference between two arbitrary policies)
    ^^^
    **For any function** :math:`f : S \rightarrow \mathbb{R}` and any policies :math:`\pi` and :math:`\pi'`, define :math:`\delta^f(s,a,s') \doteq R(s,a,s') + \gamma f(s')-f(s)`,

    .. math::
        :label: cpo-eq-3

        \epsilon^{f}_{\pi'} &\doteq \max_s \left|\underset{\substack{a\sim\pi' \\ s'\sim P} }{\mathbb{E}}\left[\delta^f(s,a,s')\right] \right|\\
        L_{\pi, f}\left(\pi'\right) &\doteq \underset{{\tau \sim \pi}}{\mathbb{E}}\left[\left(\frac{\pi'(a | s)}{\pi(a|s)}-1\right)\delta^f\left(s, a, s'\right)\right] \\
        D_{\pi, f}^{\pm}\left(\pi^{\prime}\right) &\doteq \frac{L_{\pi, f}\left(\pi' \right)}{1-\gamma} \pm \frac{2 \gamma \epsilon^{f}_{\pi'}}{(1-\gamma)^2} \underset{s \sim d_{\pi}}{\mathbb{E}}\left[D_{T V}\left(\pi^{\prime} \| \pi\right)[s]\right]


    where :math:`D_{T V}\left(\pi'|| \pi\right)[s]=\frac{1}{2} \sum_a\left|\pi'(a|s)-\pi(a|s)\right|` is the total variational divergence between action distributions at :math:`s`.
    The conclusion is as follows:

    .. math::
        :label: cpo-eq-4

        D_{\pi, f}^{+}\left(\pi'\right) \geq J\left(\pi'\right)-J(\pi) \geq D_{\pi, f}^{-}\left(\pi'\right)

    Furthermore, the bounds are tight (when :math:`\pi=\pi^{\prime}`, all three expressions are identically zero).
    +++
    The proof of the :bdg-info-line:`Theorem 1` can be seen in the :bdg-info:`Appendix`, click on this :bdg-info-line:`card` to jump to view.

By picking :math:`f=V^{R}_\pi` or :math:`f=V^{C}_\pi`,
we obtain a :bdg-info-line:`Corollary 1`,
:bdg-info-line:`Corollary 2`, :bdg-info-line:`Corollary 3` below:

.. _Corollary 1:

.. _Corollary 2:

.. tab-set::

    .. tab-item:: Corollary 1

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Corollary 1
            ^^^
            For any policies :math:`\pi'`, :math:`\pi`, with :math:`\epsilon^{R}_{\pi'}=\max _s|\underset{a \sim \pi'}{\mathbb{E}}[A^{R}_\pi(s, a)]|`, the following bound holds:

            .. math::
                :label: cpo-eq-5

                J^R\left(\pi^{\prime}\right)-J^R(\pi) \geq \frac{1}{1-\gamma} \underset{\substack{s \sim d_{\pi} \\ a \sim \pi'}}{\mathbb{E}}\left[A^R_\pi(s, a)-\frac{2 \gamma \epsilon^{R}_{\pi'}}{1-\gamma} D_{T V}\left(\pi' \| \pi\right)[s]\right]

    .. tab-item:: Corollary 2

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card:  sd-outline-info  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Corollary 2
            ^^^
            For any policies :math:`\pi'` and :math:`\pi`,
            with :math:`\epsilon^{C_i}_{\pi'}=\max _s|\underset{a \sim \pi'}{\mathbb{E}}[A^{C_i}_\pi(s, a)]|`

            the following bound holds:

            .. math::
                :label: cpo-eq-6

                J^{C_i}\left(\pi^{\prime}\right)-J^{C_i}(\pi) \geq \frac{1}{1-\gamma} \underset{\substack{s \sim d_{\pi} \\ a \sim \pi'}}{\mathbb{E}}\left[A^{C_i}_\pi(s, a)-\frac{2 \gamma \epsilon^{C_i}_{\pi'}}{1-\gamma} D_{T V}\left(\pi' \| \pi\right)[s]\right]

    .. tab-item:: Corollary 3

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Corollary 3
            ^^^
            Trust region methods prefer to constrain the :math:`KL` divergence between policies, so CPO use `Pinsker's inequality <https://en.wikipedia.org/wiki/Pinsker%27s_inequality>`_ to connect the :math:`D_{TV}` with :math:`D_{KL}`

            .. math::
                :label: cpo-eq-7

                D_{TV}(p \| q) \leq \sqrt{D_{KL}(p \| q) / 2}

            Combining this with `Jensen's inequality <https://en.wikipedia.org/wiki/Jensen%27s_inequality>`_, we obtain our final :bdg-info-line:`Corollary 3` :

            In bound :bdg-ref-info-line:`Theorem 1<Theorem 1>` , :bdg-ref-info-line:`Corollary 1<Corollary 1>`, :bdg-ref-info-line:`Corollary 2<Corollary 2>`,
            make the substitution:

            .. math::
                :label: cpo-eq-8

                \underset{s \sim d_{\pi}}{\mathbb{E}}\left[D_{T V}\left(\pi'|| \pi\right)[s]\right] \rightarrow \sqrt{\frac{1}{2} \underset{s \sim d_{\pi}}{\mathbb{E}}\left[D_{K L}\left(\pi^{\prime} \| \pi\right)[s]\right]}


------

Trust Region Methods
~~~~~~~~~~~~~~~~~~~~

For parameterized stationary policy, trust region algorithms for reinforcement
learning have policy updates of the following form:

.. _cpo-eq-11:

.. math::
    :label: cpo-eq-9

    &\boldsymbol{\theta}_{k+1}=\arg\max_{\pi \in \Pi_{\boldsymbol{\theta}}} \underset{\substack{s \sim d_{\pi_k}\\a \sim \pi}}{\mathbb{E}}[A^R_{\boldsymbol{\theta}_k}(s, a)]\\
    \text{s.t.}\quad &\bar{D}_{K L}\left(\pi \| \pi_k\right) \le \delta


where
:math:`\bar{D}_{K L}(\pi \| \pi_k)=\underset{s \sim \pi_k}{\mathbb{E}}[D_{K L}(\pi \| \pi_k)[s]]`
and :math:`\delta \ge 0` is the step size.
The set :math:`\left\{\pi_{\boldsymbol{\theta}} \in \Pi_{\boldsymbol{\theta}}: \bar{D}_{K L}\left(\pi \| \pi'\right) \leq \delta\right\}` is called trust
region.
The success motivation for this update is that it approximates optimizing the
lower bound on policy performance given in :bdg-info-line:`Corollary 1`, which
would guarantee monotonic performance improvements.

.. _cpo-eq-12:

.. math::
    :label: cpo-eq-10

    \pi_{k+1}&=\arg \max _{\pi \in \Pi_{\boldsymbol{\theta}}} \underset{\substack{s \sim d_{\pi_k}\\a \sim \pi}}{\mathbb{E}}[A^R_{\pi_k}(s, a)]\\
    \text{s.t.} \quad J^{C_i}\left(\pi_k\right) &\leq d_i-\frac{1}{1-\gamma} \underset{\substack{s \sim d_{\pi_k}\\a \sim \pi}}{\mathbb{E}}\left[A^{C_i}_{\pi_k}(s, a)\right] \quad \forall i  \\
    \bar{D}_{K L}\left(\pi \| \pi_k\right) &\leq \delta


.. hint::
    In a word, CPO uses a trust region instead of penalties on policy divergence to enable larger step size.

------

Worst-case Performance of CPO Update
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we will introduce the propositions proposed by the CPO, one describes the
worst-case performance degradation guarantee that depends on :math:`\delta`,
and the other discusses the worst-case constraint violation in the CPO update.


.. tab-set::

    .. tab-item:: Proposition 1

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Trust Region Update Performance
            ^^^
            Suppose :math:`\pi_k, \pi_{k+1}` are related by :eq:`cpo-eq-9`, and that :math:`\pi_k \in \Pi_{\boldsymbol{\theta}}`.
            A lower bound on the policy performance difference between :math:`\pi_k` and :math:`\pi_{k+1}` is:

            .. math::
                :label: cpo-eq-11

                J^{R}\left(\pi_{k+1}\right)-J^{R}(\pi_{k}) \geq \frac{-\sqrt{2 \delta} \gamma \epsilon^R_{\pi_{k+1}}}{(1-\gamma)^2}

            where :math:`\epsilon^R_{\pi_{k+1}}=\max_s\left|\mathbb{E}_{a \sim \pi_{k+1}}\left[A^R_{\pi_k}(s, a)\right]\right|`.

    .. tab-item:: Proposition 2

        .. card::
            :class-header: sd-bg-info sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-1
            :class-footer: sd-font-weight-bold

            CPO Update Worst-Case Constraint Violation
            ^^^
            Suppose :math:`\pi_k, \pi_{k+1}` are related by :eq:`cpo-eq-9`, and that :math:`\pi_k \in \Pi_{\boldsymbol{\theta}}`.
            An upper bound on the :math:`C_i`-return of :math:`\pi_{k+1}` is:

            .. math::
                :label: cpo-eq-12

                    J^{C_i}\left(\pi_{k+1}\right) \leq d_i+\frac{\sqrt{2 \delta} \gamma \epsilon^{C_i}_{\pi_{k+1}}}{(1-\gamma)^2}

            where :math:`\epsilon^{C_i}_{\pi_{k+1}}=\max _s\left|\mathbb{E}_{a \sim \pi_{k+1}}\left[A^{C_i}_{\pi_k}(s, a)\right]\right|`

.. _Proposition 2:

------

Summary
~~~~~~~

We mainly introduced the essential inequalities in CPO.
Based on those inequalities, CPO presents optimization problems that ultimately
need to be solved and propose two proposition about the worst case in the CPO
update.
Next section, we will discuss how to solve this problem practically.
You may be confused when you first read these theoretical
derivation processes, and we have given detailed proof of the above formulas in
the appendix, which we believe you can understand by reading them a few times.

------

Practical Implementation
------------------------

.. grid:: 2

    .. grid-item-card::
        :columns: 12 4 4 6
        :class-header: sd-bg-success sd-text-white sd-font-weight-bold
        :class-card: sd-outline-success  sd-rounded-1

        Overview
        ^^^
        In this section, we show how CPO implements an approximation to the update :eq:`cpo-eq-10`, even when optimizing policies with thousands of parameters.
        To address the issue of approximation and sampling errors that arise in practice and the potential violations described by :bdg-ref-info-line:`Proposition 2<Proposition 2>`, CPO proposes to tighten the constraints by constraining the upper bounds of the extra costs instead of the extra costs themselves.

    .. grid-item-card::
        :columns: 12 8 8 6
        :class-header: sd-bg-success sd-text-white sd-font-weight-bold
        :class-card: sd-outline-success  sd-rounded-1

        Navigation
        ^^^
        Approximately Solving the CPO Update

        :bdg-ref-success-line:`Click here<Approximately_Solving_the_CPO_Update>`

        Feasibility

        :bdg-ref-success-line:`Click here<Feasibility>`

        Tightening Constraints via Cost Shaping

        :bdg-ref-success-line:`Click here<Tightening_Constraints_via_Cost_Shaping>`

        Code With OmniSafe

        :bdg-ref-success-line:`Click here<Code_with_OmniSafe>`



------

.. _Approximately_Solving_the_CPO_Update:

Approximately Solving the CPO Update
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For policies with high-dimensional parameter spaces like neural networks, :eq:`cpo-eq-10` can be impractical to solve directly because of the computational cost.

.. hint::
    However, for small step sizes :math:`\delta`, the objective and cost constraints are well-approximated by linearizing around :math:`\pi_k`, and the KL-Divergence constraint is well-approximated by second-order expansion.

Denoting the gradient of the objective as :math:`g`, the gradient of constraint :math:`i` as :math:`b_i`, the Hessian of the :math:`KL` divergence as :math:`H`, and defining :math:`c_i=J^{C_i}\left(\pi_k\right)-d_i`, the approximation to :eq:`cpo-eq-10` is:

.. _cpo-eq-13:

.. math::
    :label: cpo-eq-13

    &\boldsymbol{\theta}_{k+1}=\arg \max _{\boldsymbol{\theta}} g^T\left(\boldsymbol{\theta}-\boldsymbol{\theta}_k\right)\\
    \text{s.t.}\quad  &c_i+b_i^T\left(\boldsymbol{\theta}-\boldsymbol{\theta}_k\right) \leq 0 ~~~ i=1, \ldots m \\
    &\frac{1}{2}\left(\boldsymbol{\theta}-\boldsymbol{\theta}_k\right)^T H\left(\boldsymbol{\theta}-\boldsymbol{\theta}_k\right) \leq \delta


With :math:`B=\left[b_1, \ldots, b_m\right]` and :math:`c=\left[c_1, \ldots, c_m\right]^T`, a dual to :eq:`cpo-eq-13` can be express as:

.. math::
    :label: cpo-eq-14

    \max_{\lambda \geq 0, \nu \geq 0} \frac{-1}{2 \lambda}\left(g^T H^{-1} g-2 r^T v+v^T S v\right)+v^T c-\frac{\lambda \delta}{2}

where :math:`r=g^T H^{-1} B, S=B^T H^{-1} B`. If :math:`\lambda^*, v^*` are a solution to the dual, the solution to the primal is

.. _cpo-eq-14:

.. math::
    :label: cpo-eq-15

    {\boldsymbol{\theta}}^*={\boldsymbol{\theta}}_k+\frac{1}{\lambda^*} H^{-1}\left(g-B v^*\right)


In a word, CPO solves the dual for :math:`\lambda^*, \nu^*` and uses it to
propose the policy update :eq:`cpo-eq-15`, thus solving :eq:`cpo-eq-10` in a
particular way.
In the experiment,
CPO also uses two tricks to promise the update's performance.

.. warning::
    Because of the approximation error, the proposed update may not satisfy the constraints in :eq:`cpo-eq-10`; A backtracking line search is used to ensure surrogate constraint satisfaction.

For high-dimensional policies, it is impractically expensive to invert the
Fisher information matrix.
This poses a challenge for computing :math:`H^{-1} \mathrm{~g}` and
:math:`H^{-1} b`, which appears in the dual.
Like TRPO, CPO computes them approximately using the conjugate gradient method.

------

.. _Feasibility:

Feasibility
~~~~~~~~~~~

CPO may occasionally produce an infeasible iterate :math:`\pi_k` due to
approximation errors. To handle such cases, CPO proposes an update that purely
decreases the constraint value.

.. math::
    :label: cpo-eq-16

    \boldsymbol{\theta}^*=\boldsymbol{\theta}_k-\sqrt{\frac{2 \delta}{b^T H^{-1} b}} H^{-1} b

This is followed by a line search, similar to
before. This approach is principled because it uses the limiting search
direction as the intersection of the trust region and the constraint region
shrinks to zero.

------

.. _Tightening_Constraints_via_Cost_Shaping:

Tightening Constraints via Cost Shaping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To build a factor of safety into the algorithm minimizing the chance of
constraint violations, CPO chooses to constrain upper bounds on the original
constraints, :math:`C_i^{+}`, instead of the original constraints themselves.
CPO does this by cost shaping:

.. math::
    :label: cpo-eq-17

    C_i^{+}\left(s, a, s^{\prime}\right)=C_i\left(s, a, s^{\prime}\right)+\triangle_i\left(s, a, s^{\prime}\right)

where
:math:`\delta_i: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow R_{+}`
correlates in
some useful way with :math:`C_i`.
Because CPO has only one constraint, it partitions states into safe and unsafe
states, and the agent suffers a safety cost of 1 for being in an unsafe state.

CPO chooses :math:`\triangle` to be the probability of entering an unsafe state
within a fixed time horizon, according to a learned model that is updated at
each iteration.
This choice confers the additional benefit of smoothing out sparse constraints.

------

.. _Code_with_OmniSafe:

Code with OmniSafe
~~~~~~~~~~~~~~~~~~

Quick start
"""""""""""

.. card::
    :class-header: sd-bg-success sd-text-white sd-font-weight-bold
    :class-card: sd-outline-success  sd-rounded-1 sd-font-weight-bold
    :class-footer: sd-font-weight-bold

    Run CPO in OmniSafe
    ^^^^^^^^^^^^^^^^^^^
    Here are 3 ways to run CPO in OmniSafe:

    * Run Agent from preset yaml file
    * Run Agent from custom config dict
    * Run Agent from custom terminal config

    .. tab-set::

        .. tab-item:: Yaml file style

            .. code-block:: python
                :linenos:

                import omnisafe


                env_id = 'SafetyPointGoal1-v0'

                agent = omnisafe.Agent('CPO', env_id)
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

                agent = omnisafe.Agent('CPO', env_id, custom_cfgs=custom_cfgs)
                agent.learn()


        .. tab-item:: Terminal config style

            We use ``train_policy.py`` as the entrance file. You can train the agent with CPO simply using ``train_policy.py``, with arguments about CPO and environments does the training.
            For example, to run CPO in SafetyPointGoal1-v0 , with 1 torch thread, seed 0 and single environment, you can use the following command:

            .. code-block:: bash
                :linenos:

                cd examples
                python train_policy.py --algo CPO --env-id SafetyPointGoal1-v0 --parallel 1 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 1

------

Here is the documentation of CPO in PyTorch version.


Architecture of functions
"""""""""""""""""""""""""

- ``CPO.learn()``

  - ``CPO._env.rollout()``
  - ``CPO._update()``

    - ``CPO._buf.get()``
    - ``CPO._update_actor()``

      - ``CPO._fvp()``
      - ``conjugate_gradients()``
      - ``CPO._cpo_search_step()``

    - ``CPO._update_cost_critic()``
    - ``CPO._update_reward_critic()``


------

Documentation of algorithm specific functions
"""""""""""""""""""""""""""""""""""""""""""""

.. tab-set::

    .. tab-item:: cpo._update_actor()

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success  sd-rounded-1 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            cpo._update_actor()
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

                loss_reward.backward()
                distributed.avg_grads(self._actor_critic.actor)

                grads = -get_flat_gradients_from(self._actor_critic.actor)


            (2) Get the policy cost performance gradient b and ep_costs (flat as vector)

            .. code-block:: python
                :linenos:

                self._actor_critic.zero_grad()
                loss_cost = self._loss_pi_cost(obs, act, logp, adv_c)
                loss_cost_before = distributed.dist_avg(loss_cost).item()

                loss_cost.backward()
                distributed.avg_grads(self._actor_critic.actor)

                b_grads = get_flat_gradients_from(self._actor_critic.actor)
                ep_costs = self._logger.get_stats('Metrics/EpCost')[0] - self._cfgs.algo_cfgs.cost_limit

            (3) Build the Hessian-vector product based on an approximation of the :math:`KL` divergence, using ``conjugate_gradients``.

            .. code-block:: python
                :linenos:

                p = conjugate_gradients(self._fvp, b_grads, self._cfgs.algo_cfgs.cg_iters)
                q = xHx
                r = grads.dot(p)
                s = b_grads.dot(p)

            (4) Divide the optimization case into 5 kinds to compute.

            (5) Determine step direction and apply SGD step after grads where set (By ``search_step_size()``)

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
                    total_steps=20,
                    violation_c=ep_costs,
                    optim_case=optim_case,
                )

            (6) Update actor network parameters

            .. code-block:: python
                :linenos:

                theta_new = theta_old + step_direction
                set_param_values_to_model(self._actor_critic.actor, theta_new)

    .. tab-item:: cpo._cpo_search_step()

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success  sd-rounded-1 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            cpo._search_step_size()
            ^^^
            CPO algorithm performs line-search to ensure constraint satisfaction for rewards and costs, flowing the next steps:

            (1) Initialize the step size and get the old flat parameters of the policy network.

            .. code-block:: python
               :linenos:

                # get distance each time theta goes towards certain direction
                step_frac = 1.0
                # get and flatten parameters from pi-net
                theta_old = get_flat_params_from(self._actor_critic.actor)
                # reward improvement, g-flat as gradient of reward
                expected_reward_improve = grad.dot(step_direction)

            (1) Calculate the expected reward improvement.

            .. code-block:: python
               :linenos:

               expected_rew_improve = g_flat.dot(step_dir)

            (2) Performs line-search to find a step to improve the surrogate while not violating the trust region.

            - Search acceptance step ranging from 0 to total step

            .. code-block:: python
               :linenos:

               for j in range(total_steps):
                  new_theta = _theta_old + step_frac * step_dir
                  set_param_values_to_model(self.ac.pi.net, new_theta)
                  acceptance_step = j + 1

            - In each step of for loop, calculate the policy performance and KL divergence.

            .. code-block:: python
               :linenos:

               with torch.no_grad():
                   loss_pi_rew, _ = self.compute_loss_pi(data=data)
                   loss_pi_cost, _ = self.compute_loss_cost_performance(data=data)
                   q_dist = self.ac.pi.dist(data['obs'])
                   torch_kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).mean().item()
               loss_rew_improve = self.loss_pi_before - loss_pi_rew.item()
               cost_diff = loss_pi_cost.item() - self.loss_pi_cost_before

            - Step only if the surrogate is improved and within the trust region.

            .. code-block:: python
               :linenos:

               if not torch.isfinite(loss_pi_rew) and not torch.isfinite(loss_pi_cost):
                   self.logger.log('WARNING: loss_pi not finite')
               elif loss_rew_improve < 0 if optim_case > 1 else False:
                   self.logger.log('INFO: did not improve improve <0')

               elif cost_diff > max(-c, 0):
                   self.logger.log(f'INFO: no improve {cost_diff} > {max(-c, 0)}')
               elif torch_kl > self.target_kl * 1.5:
                   self.logger.log(f'INFO: violated KL constraint {torch_kl} at step {j + 1}.')
               else:
                   self.logger.log(f'Accept step at i={j + 1}')
                   break

            (3) Return appropriate step direction and acceptance step.


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

                The following configs are specific to CPO algorithm.

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
-  `A Natural Policy Gradient <https://proceedings.neurips.cc/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf>`__
-  `Trust Region Policy Optimization <https://arxiv.org/abs/1502.05477>`__
-  `Constrained Markov Decision Processes <https://www.semanticscholar.org/paper/Constrained-Markov-Decision-Processes-Altman/3cc2608fd77b9b65f5bd378e8797b2ab1b8acde7>`__

.. _Appendix:

.. _cards-clickable:

Appendix
--------

:bdg-ref-info-line:`Click here to jump to CPO Theorem<Theorem 1>`  :bdg-ref-success-line:`Click here to jump to Code with OmniSafe<Code_with_OmniSafe>`

Proof of theorem 1 (Difference between two arbitrary policies)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Our analysis will begin with the discounted future state distribution,
:math:`d_\pi`, which is defined as:

.. math::
    :label: cpo-eq-18

    d_\pi(s)=(1-\gamma) \sum_{t=0}^{\infty} \gamma^t P\left(s_t=s|\pi\right)

Let :math:`p_\pi^t \in \mathbb{R}^{|\mathcal{S}|}`
denote the vector with components
:math:`p_\pi^t(s)=P\left(s_t=s \mid \pi\right)`, and let
:math:`P_\pi \in \mathbb{R}^{|\mathcal{S}| \times|\mathcal{S}|}`
denotes the transition
matrix with components
:math:`P_\pi\left(s^{\prime} \mid s\right)=\int d a P\left(s^{\prime} \mid s, a\right) \pi(a \mid s)`,
which shown as below:

.. math::
    :label: cpo-eq-19

    &\left[\begin{array}{c}
    p_\pi^t\left(s_1\right) \\
    p_\pi^t\left(s_2\right) \\
    \vdots\nonumber \\
    p_\pi^t\left(s_n\right)
    \end{array}\right]
    =\left[\begin{array}{cccc}
    P_\pi\left(s_1 \mid s_1\right) & P_\pi\left(s_1 \mid s_2\right) & \cdots & P_\pi\left(s_1 \mid s_n\right) \\
    P_\pi\left(s_2 \mid s_1\right) & P_\pi\left(s_2 \mid s_2\right) & \cdots & P_\pi\left(s_2 \mid s_n\right) \\
    \vdots & \vdots & \ddots & \vdots \\
    P_\pi\left(s_n \mid s_1\right) & P_\pi\left(s_n \mid s_2\right) & \cdots & P_\pi\left(s_n \mid s_n\right)
    \end{array}\right]\left[\begin{array}{c}
    p_\pi^{t-1}\left(s_1\right) \\
    p_\pi^{t-1}\left(s_2\right) \\
    \vdots \\
    p_\pi^{t-1}\left(s_n\right)
    \end{array}\right]

Then :math:`p_\pi^t=P_\pi p_\pi^{t-1}=P_\pi^2 p_\pi^{t-2}=\ldots=P_\pi^t \mu`,
where :math:`\mu` represents the state distribution of the system at the moment.
That is, the initial state distribution, then :math:`d_\pi` can then be
rewritten as:

.. math::
    :label: cpo-eq-20

    d_\pi&=\left[\begin{array}{c}
    d_\pi\left(s_1\right) \\
    d_\pi\left(s_2\right) \\
    \vdots \\
    d_\pi\left(s_n\right)
    \end{array}\right]
    =(1-\gamma)\left[\begin{array}{c}
    \gamma^0 p_\pi^0\left(s_1\right)+\gamma^1 p_\pi^1\left(s_1\right)+\gamma^2 p_\pi^2\left(s_1\right)+\ldots \\
    \gamma^0 p_\pi^0\left(s_2\right)+\gamma^1 p_\pi^1\left(s_2\right)+\gamma^2 p_\pi^2\left(s_2\right)+\ldots \\
    \vdots \\
    \gamma^0 p_\pi^0\left(s_3\right)+\gamma^1 p_\pi^1\left(s_3\right)+\gamma^2 p_\pi^2\left(s_3\right)+\ldots
    \end{array}\right]

.. _cpo-eq-17:

.. math::
    :label: cpo-eq-21

    d_\pi=(1-\gamma) \sum_{t=0}^{\infty} \gamma^t p_\pi^t=(1-\gamma)\left(1-\gamma P_\pi\right)^{-1} \mu


.. tab-set::

    .. tab-item:: Lemma 1

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Lemma 1
            ^^^
            For any function :math:`f: \mathcal{S} \rightarrow \mathbb{R}` and any policy :math:`\pi` :

            .. math::
                :label: cpo-eq-22

                (1-\gamma) \underset{s \sim \mu}{\mathbb{E}}[f(s)]+\underset{\tau \sim \pi}{\mathbb{E}}\left[\gamma f\left(s'\right)\right]-\underset{s \sim d_\pi}{\mathbb{E}}[f(s)]=0

            where :math:`\tau \sim \pi` denotes :math:`s \sim d_\pi, a \sim \pi` and :math:`s^{\prime} \sim P`.

    .. tab-item:: Lemma 2

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Lemma 2
            ^^^
            For any function :math:`f: \mathcal{S} \rightarrow \mathbb{R}` and any policies
            :math:`\pi` and :math:`\pi'`, define

            .. math::
                :label: cpo-eq-23

                L_{\pi, f}\left(\pi'\right)\doteq \underset{\tau \sim \pi}{\mathbb{E}}\left[\left(\frac{\pi^{\prime}(a \mid s)}{\pi(a \mid s)}-1\right)\left(R\left(s, a, s^{\prime}\right)+\gamma f\left(s^{\prime}\right)-f(s)\right)\right]

            and :math:`\epsilon^{f}_{\pi^{\prime}}\doteq \max_s \left|\underset{\substack{a \sim \pi \\ s'\sim P}}{\mathbb{E}}\left[R\left(s, a, s^{\prime}\right)+\gamma f\left(s^{\prime}\right)-f(s)\right]\right|`.
            Then the following bounds hold:

            .. math::
                :label: cpo-eq-24

                &J\left(\pi'\right)-J(\pi) \geq \frac{1}{1-\gamma}\left(L_{\pi, f}\left(\pi'\right)-2 \epsilon^{f}_{\pi'} D_{T V}\left(d_\pi \| d_{\pi^{\prime}}\right)\right) \\
                &J\left(\pi^{\prime}\right)-J(\pi) \leq \frac{1}{1-\gamma}\left(L_{\pi, f}\left(\pi'\right)+2 \epsilon^{f}_{\pi'} D_{T V}\left(d_\pi \| d_{\pi'}\right)\right)


            where :math:`D_{T V}` is the total variational divergence. Furthermore, the bounds are tight when :math:`\pi^{\prime}=\pi`, and the LHS and RHS are identically zero.

    .. tab-item:: Lemma 3

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Lemma 3
            ^^^
            The divergence between discounted future state visitation
            distributions, :math:`\Vert d_{\pi'}-d_\pi\Vert_1`, is bounded by an
            average divergence of the policies :math:`\pi` and :math:`\pi'` :

            .. math::
                :label: cpo-eq-25

                \Vert d_{\pi'}-d_\pi\Vert_1 \leq \frac{2 \gamma}{1-\gamma} \underset{s \sim d_\pi}{\mathbb{E}}\left[D_{T V}\left(\pi^{\prime} \| \pi\right)[s]\right]


            where :math:`D_{\mathrm{TV}}(\pi' \| \pi)[s] = \frac{1}{2}\sum_a \Vert\pi'(a|s) - \pi(a|s)\Vert`

    .. tab-item:: Corollary 4

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Corollary 4
            ^^^
            Define the matrices
            :math:`G \doteq\left(I-\gamma P_\pi\right)^{-1}, \bar{G} \doteq\left(I-\gamma P_{\pi^{\prime}}\right)^{-1}`,
            and :math:`\Delta=P_{\pi^{\prime}}-P_\pi`. Then:

            .. math::
                :label: cpo-eq-26

                G^{-1}-\bar{G}^{-1} &=\left(I-\gamma P_\pi\right)-\left(I-\gamma P_{\pi^{\prime}}\right) \\
                G^{-1}-\bar{G}^{-1} &=\gamma \Delta \\
                \bar{G}\left(G^{-1}-\bar{G}^{-1}\right) G &=\gamma \bar{G} \Delta G \\
                \bar{G}-G &=\gamma \bar{G} \Delta G

            Thus, with :eq:`cpo-eq-21`

            .. math::
                :label: cpo-eq-27

                d_{\pi^{\prime}}-d_{\pi} &=(1-\gamma)(\bar{G}-G) \mu \\
                &=\gamma(1-\gamma) \bar{G} \Delta G \mu\\
                &=\gamma \bar{G} \Delta d_{\pi}


    .. tab-item:: Corollary 5

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Corollary 5
            ^^^
            .. math::
                :label: cpo-eq-28

                \left\|P_{\pi^{\prime}}\right\|_1=\max _{s \in \mathcal{S}}\left\{\sum_{s^{\prime} \in \mathcal{S}} P_\pi\left(s^{\prime} \mid s\right)\right\}=1

Begin with the bounds from :bdg-info-line:`Lemma 2` and bound the divergence by
:bdg-info-line:`Lemma 3`, :bdg-info-line:`Theorem 1` can be finally proved.

.. _cpo-eq-18:

.. tab-set::

    .. tab-item:: Proof of Lemma 1

        .. card::
            :class-header: sd-bg-info sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Proof
            ^^^
            Multiply both sides of :eq:`cpo-eq-21` by :math:`\left(I-\gamma P_\pi\right)`, we get:

            .. math::
                :label: cpo-eq-29

                \left(I-\gamma P_\pi\right) d_\pi=(1-\gamma) \mu

            Then take the inner product with the vector :math:`f \in \mathbb{R}^{|S|}` and notice that the vector :math:`f`
            can be arbitrarily picked.

            .. math::
                :label: cpo-eq-30

                <f,\left(I-\gamma P_\pi\right) d_\pi>=<f,(1-\gamma) \mu>

            Both sides of the above equation can be rewritten separately by:

            .. math::
                :label: cpo-eq-31

                &<f,\left(I-\gamma P_\pi\right) d_\pi>\\
                &=\left[\sum_s f(s) d_\pi(s)\right]-
                \left[\sum_{s^{\prime}} f\left(s^{\prime}\right) \gamma \sum_s \sum_a \pi(a \mid s) P\left(s^{\prime} \mid s, a\right) d_\pi(s)\right] \\
                &=\underset{s \sim d_\pi}{\mathbb{E}}[f(s)]-\underset{\tau \sim \pi}{\mathbb{E}}\left[\gamma f\left(s^{\prime}\right)\right]

            .. math::
                :label: cpo-eq-32

                <f,(1-\gamma) \mu>=\sum_s f(s)(1-\gamma) \mu(s)=(1-\gamma) \underset{s \sim \mu}{\mathbb{E}}[f(s)]

            Finally, we obtain:

            .. math::
                :label: cpo-eq-33

                (1-\gamma) \underset{s \sim \mu}{\mathbb{E}}[f(s)]+\underset{\tau \sim \pi}{\mathbb{E}}\left[\gamma f\left(s^{\prime}\right)\right]-\underset{s \sim d_\pi}{\mathbb{E}}[f(s)] = 0

            .. hint::

                **Supplementary details**

                .. math::
                    :label: cpo-eq-34

                    d_{\pi} &=(1-\gamma)\left(I-\gamma P_\pi\right)^{-1} \mu \\\left(I-\gamma P_\pi\right) d_{\pi} &=(1-\gamma)  \mu \\ \int_{s \in \mathcal{S}}\left(I-\gamma P_\pi\right) d_{\pi} f(s) d s &=\int_{s \in \mathcal{S}} (1-\gamma) \mu f(s) d s \\ \int_{s \in \mathcal{S}} d_{\pi} f(s) d s-\int_{s \in \mathcal{S}} \gamma P_\pi  d_{\pi} f(s) d s &=\int_{s \in \mathcal{S}}(1-\gamma) \mu f(s) d s \\ \underset{s \sim d_\pi}{\mathbb{E}}[f(s)] -\underset{\tau \sim \pi}{\mathbb{E}}\left[\gamma f\left(s^{\prime}\right)\right]] &= (1-\gamma) \underset{s \sim \mu}{\mathbb{E}}[f(s)]


    .. tab-item:: Proof of Lemma 2

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Proof
            ^^^
            Note that the objective function can be represented as:

            .. math::
                :label: cpo-eq-35

                J(\pi)&=\frac{1}{1-\gamma} \underset{\tau \sim \pi}{\mathbb{E}}[R(s, a, s^{\prime})]  \\
                &=\underset{s \sim \mu}{\mathbb{E}}[f(s)]+\frac{1}{1-\gamma} \underset{\tau \sim \pi}{\mathbb{E}}[R(s, a, s^{\prime})+\gamma f(s^{\prime})-f(s)]


            Let :math:`\delta^f(s, a, s^{\prime})\doteq R(s, a, s^{\prime})+\gamma f(s^{\prime})-f(s)`, then by :eq:`cpo-eq-29`, we easily obtain that:

            .. math::
                :label: cpo-eq-36

                J(\pi')-J(\pi)=\frac{1}{1-\gamma}\underset{\tau \sim \pi^{\prime}}{\mathbb{E}}[\delta^f(s, a, s^{\prime})]-\underset{\tau \sim \pi}{\mathbb{E}}[\delta^f(s, a, s^{\prime}]\}.

            For the first term of the equation, let :math:`\bar{\delta}^{f}_{\pi'} \in \mathbb{R}^{|S|}` denotes the vector of components :math:`\bar{\delta}^{f}_{\pi'}(s)=\underset{\substack{a \sim \pi' \\ s' \sim P}}{\mathbb{E}}\left[\delta^f\left(s, a, s'|s\right)\right]`, then

            .. math::
                :label: cpo-eq-37

                \underset{\tau \sim \pi^{\prime}}{\mathbb{E}}\left[d_f\left(s, a, s'\right)\right]=<d_{\pi'}, \bar{\delta}^f_{\pi'}>=<d_\pi,\bar{\delta}^f_{\pi'}>+<d_{\pi'}-d_\pi, \hat{d}^f_{\pi'}>

            By using Hölder's inequality, for any :math:`p, q \in[1, \infty]`, such that :math:`\frac{1}{p}+\frac{1}{q}=1`.
            We have

            .. math::
                :label: cpo-eq-38

                & \underset{\tau \sim \pi^{\prime}}{\mathbb{E}}\left[\delta^f\left(s, a, s^{\prime}\right)\right] \leq \langle d_\pi, \bar{\delta}^{f}_{\pi^{\prime}} \rangle+\Vert d_{\pi'}-d_\pi \Vert_p \Vert \bar{\delta}^{f}_{\pi^{\prime}}\Vert_q  \\
                &\underset{\tau \sim \pi^{\prime}}{\mathbb{E}}\left[\delta^f\left(s, a, s'\right)\right] \geq \langle d_\pi, \bar{\delta}^{f}_{\pi^{\prime}}\rangle-\Vert d_{\pi'}-d_\pi \Vert_p \Vert \bar{\delta}^{f}_{\pi^{\prime}}\Vert_q

            .. hint::

                **Hölder's inequality**:

                Let :math:`(\mathcal{S}, \sum, \mu)` be a measure space and let :math:`p, q \in [1, \infty]` with :math:`\frac{1}{p} + \frac{1}{q} = 1`. Then for all measurable real or complex-valued function :math:`f` and :math:`g` on :math:`s`, :math:`\|f g\|_1 \leq\|f\|_p\|g\|_q`.

                If, in addition, :math:`p, q \in(1, \infty)` and :math:`f \in L^p(\mu)` and :math:`g \in L^q(\mu)`, then
                Hölder's inequality becomes an equality if and only if :math:`|f|^p` and :math:`|g|^q` are linearly dependent in :math:`L^1(\mu)`, meaning that there exists real numbers :math:`\alpha, \beta \geq 0`, not both of them zero, such that :math:`\alpha|f|^p=\beta|g|^q \mu` almost everywhere.

            The last step is to observe that, by the importance of sampling identity,

            .. math::
                :label: cpo-eq-39

                \left\langle d_{\pi}, \bar{\delta}^{f}_{\pi^{\prime}}\right\rangle &=\underset{\substack{s \sim d_{\pi} \\ a \sim \pi^{\prime} \\ s^{\prime} \sim P}}{\mathbb{E}}\left[\delta^f\left(s, a, s^{\prime}\right)\right] \\
                &=\underset{\substack{s \sim d_{\pi} \\ a \sim \pi^{\prime} \\ s^{\prime} \sim P}}{\mathbb{E}}\left[\left(\frac{\pi^{\prime}(a \mid s)}{\pi(a \mid s)}\right) \delta^f\left(s, a, s^{\prime}\right)\right]

            After grouping terms, the bounds are obtained.

            .. math::
                :label: cpo-eq-40

                &\left\langle d_{\pi}, \bar{\delta}^{f}_{\pi^{\prime}}\right\rangle \pm\Vert d_{\pi^{\prime}}-d_{\pi}\Vert_p\Vert\bar{\delta}^{f}_{\pi^{\prime}}\Vert_q\\
                &=\underset{\tau \sim \pi}{\mathbb{E}}\left[\left(\frac{\pi'(a|s)}{\pi(a|s)}\right) \delta^f\left(s, a, s^{\prime}\right)\right] \pm 2 \epsilon^{f}_{\pi^{\prime}} D_{T V}\left(d_{\pi'} \| d_\pi\right)

            .. math::
                :label: cpo-eq-41

                &J(\pi')-J(\pi)\\
                &\leq \frac{1}{1-\gamma}\underset{\tau \sim \pi}{\mathbb{E}}[(\frac{\pi^{\prime}(a|s)}{\pi(a|s)}) \delta^f(s, a, s^{\prime})]+2 \epsilon^{f}_{\pi^{\prime}} D_{T V}(d_{\pi^{\prime}} \| d_{\pi})-\underset{\tau \sim \pi}{\mathbb{E}}[\delta^f(s, a, s^{\prime})]\\
                &=\frac{1}{1-\gamma}(\underset{\tau \sim \pi}{\mathbb{E}}[(\frac{\pi^{\prime}(a|s)}{\pi(a|s)}) \delta^f(s, a, s^{\prime})]-\underset{\tau \sim \pi}{\mathbb{E}}[\delta^f(s, a, s^{\prime})]+2 \epsilon^{f}_{\pi^{\prime}} D_{T V}(d_{\pi^{\prime}} \| d_{\pi}))\\
                &=\frac{1}{1-\gamma}(\underset{\tau \sim \pi}{\mathbb{E}}[(\frac{\pi^{\prime}(a \mid s)}{\pi(a \mid s)}-1) \delta^f(s, a, s^{\prime})]+2 \epsilon^{f}_{\pi^{\prime}} D_{T V}(d_{\pi^{\prime}} \| d_{\pi}))

            The lower bound is the same.

            .. math::
                :label: cpo-eq-42

                J\left(\pi^{\prime}\right)-J(\pi) \geq\underset{\tau \sim \pi}{\mathbb{E}}\left[\left(\frac{\pi^{\prime}(a|s)}{\pi(a|s)}-1\right) \delta^f\left(s, a, s^{\prime}\right)\right]-2 \epsilon^{f}_{\pi^{\prime}} D_{T V}\left(d_{\pi^{\prime}} \| d_{\pi}\right)

    .. tab-item:: Proof of Lemma 3

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Proof
            ^^^
            First, using Corollary 4, we obtain

            .. math::
                :label: cpo-eq-43

                \left\|d_{\pi^{\prime}}-d_{\pi}\right\|_1 &=\gamma\left\|\bar{G} \Delta d_{\pi}\right\|_1 \\
                & \leq \gamma\|\bar{G}\|_1\left\|\Delta d_{\pi}\right\|_1

            Meanwhile,

            .. math::
                :label: cpo-eq-44

                \|\bar{G}\|_1 &=\left\|\left(I-\gamma P_{\pi^{\prime}}\right)^{-1}\right\|_1 \\ &=\left\|\sum_{t=0}^{\infty} \gamma^t P_{\pi^{\prime}}^t\right\|_1 \\ & \leq \sum_{t=0}^{\infty} \gamma^t\left\|P_{\pi^{\prime}}\right\|_1^t \\ &=\left(1-\gamma\left\|P_{\pi^{\prime}}\right\|_1\right)^{-1} \\ &=(1-\gamma)^{-1}

            And, using Corollary 5, we have,

            .. math::
                :label: cpo-eq-45

                \Delta d_{\pi}\left[s^{\prime}\right] &= \sum_s \Delta\left(s^{\prime} \mid s\right) d_{\pi}(s) \\
                &=\sum_s \left\{ P_{\pi^{\prime}}\left(s^{\prime} \mid s\right)-P_\pi\left(s^{\prime} \mid s\right)  \right\} d_{\pi}(s) \\
                &=\sum_s \left\{ P\left(s^{\prime} \mid s, a\right) \pi^{\prime}(a \mid s)-P\left(s^{\prime} \mid s, a\right) \pi(a \mid s)  \right\} d_{\pi}(s)\\
                &=\sum_s \left\{ P\left(s^{\prime} \mid s, a\right)\left[\pi^{\prime}(a \mid s)-\pi(a \mid s)\right] \right\} d_{\pi}(s)


            .. hint::

                **Total variation distance of probability measures**

                :math:`\Vert d_{\pi'}-d_\pi \Vert_1=\sum_{a \in \mathcal{A}}\left|d_{\pi_{{\boldsymbol{\theta}}^{\prime}}}(a|s)-d_{\pi_{\boldsymbol{\theta}}}(a|s)\right|=2 D_{\mathrm{TV}}\left(d_{\pi_{{\boldsymbol{\theta}}'}}, d_\pi\right)[s]`

            Finally, using :ref:`(20) <cpo-eq-18>`, we obtain,

            .. math::
                :label: cpo-eq-46

                \left\|\Delta d_{\pi}\right\|_1 &=\sum_{s^{\prime}}\left|\sum_s \Delta\left(s^{\prime} \mid s\right) d_{\pi}(s)\right| \\ & \leq \sum_{s, s^{\prime}}\left|\Delta\left(s^{\prime} \mid s\right)\right| d_{\pi}(s) \\ &=\sum_{s, s^{\prime}}\left|\sum_a P\left(s^{\prime} \mid s, a\right)\left(\pi^{\prime}(a \mid s)-\pi(a \mid s)\right)\right| d_{\pi}(s) \\ & \leq \sum_{s, a, s^{\prime}} P\left(s^{\prime} \mid s, a\right)\left|\pi^{\prime}(a \mid s)-\pi(a \mid s)\right| d_{\pi}(s) \\ &=\sum_{s^{\prime}} P\left(s^{\prime} \mid s, a\right) \sum_{s, a}\left|\pi^{\prime}(a \mid s)-\pi(a \mid s)\right| d_{\pi}(s) \\ &=\sum_{s, a}\left|\pi^{\prime}(a \mid s)-\pi(a \mid s)\right| d_{\pi}(s) \\ &=\sum_a \underset{s \sim d_{\pi}}{ } \mathbb{E}^{\prime}|(a \mid s)-\pi(a \mid s)| \\ &=2 \underset{s \sim d_{\pi}}{\mathbb{E}}\left[D_{T V}\left(\pi^{\prime}|| \pi\right)[s]\right]


------

Proof of Analytical Solution to LQCLP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. card::
    :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
    :class-card: sd-outline-info  sd-rounded-1
    :class-footer: sd-font-weight-bold

    Theorem 2 (Optimizing Linear Objective with Linear, Quadratic Constraints)
    ^^^
    Consider the problem

    .. math::
        :label: cpo-eq-47

        p^*&=\min_x g^T x \\
        \text { s.t. }\quad & b^T x+c \leq 0 \\
        & x^T H x \leq \delta


    where
    :math:`g, b, x \in \mathbb{R}^n, c, \delta \in \mathbb{R}, \delta>0, H \in \mathbb{S}^n`,
    and :math:`H \succ 0`. When there is at least one strictly feasible
    point, the optimal point :math:`x^*` satisfies

    .. math::
        :label: cpo-eq-48

        x^*=-\frac{1}{\lambda^*} H^{-1}\left(g+\nu^* b\right)


    where :math:`\lambda^*` and :math:`\nu^*` are defined by

    .. math::
        :label: cpo-eq-49

        &\nu^*=\left(\frac{\lambda^* c-r}{s}\right)_{+}, \\
        &\lambda^*=\arg \max _{\lambda \geq 0} \begin{cases}f_a(\lambda) \doteq \frac{1}{2 \lambda}\left(\frac{r^2}{s}-q\right)+\frac{\lambda}{2}\left(\frac{c^2}{s}-\delta\right)-\frac{r c}{s} & \text { if } \lambda c-r>0 \\
        f_b(\lambda) \doteq-\frac{1}{2}\left(\frac{q}{\lambda}+\lambda \delta\right) & \text { otherwise }\end{cases}


    with :math:`q=g^T H^{-1} g, r=g^T H^{-1} b`, and
    :math:`s=b^T H^{-1} b`.

    Furthermore, let
    :math:`\Lambda_a \doteq\{\lambda \mid \lambda c-r>0, \lambda \geq 0\}`,
    and
    :math:`\Lambda_b \doteq\{\lambda \mid \lambda c-r \leq 0, \lambda \geq 0\}`.
    The value of :math:`\lambda^*` satisfies

    .. math::
        :label: cpo-eq-50

        \lambda^* \in\left\{\lambda_a^* \doteq \operatorname{Proj}\left(\sqrt{\frac{q-r^2 / s}{\delta-c^2 / s}}, \Lambda_a\right), \lambda_b^* \doteq \operatorname{Proj}\left(\sqrt{\frac{q}{\delta}}, \Lambda_b\right)\right\}

    with :math:`\lambda^*=\lambda_a^*` if
    :math:`f_a\left(\lambda_a^*\right)>f_b\left(\lambda_b^*\right)` and
    :math:`\lambda = \lambda_b^*` otherwise, and
    :math:`\operatorname{Proj}(a, S)` is the projection of a point
    :math:`x` on to a set :math:`S`.

    .. hint::
        the projection of a point
        :math:`x \in \mathbb{R}` onto a convex segment of
        :math:`\mathbb{R},[a, b]`, has value
        :math:`\operatorname{Proj}(x,[a, b])=\max (a, \min (b, x))`.

.. dropdown:: Proof for Theorem 2 (Click here)
    :color: info
    :class-body: sd-outline-info

    This is a convex optimization problem. When there is at least one strictly feasible point, strong duality holds by Slater's theorem.
    We exploit strong duality to solve the problem analytically.
    First using the method of Lagrange multipliers, :math:`\exists \lambda, \mu \geq 0`

    .. math::
        :label: cpo-eq-51

        \mathcal{L}(x, \lambda, \nu)=g^T x+\frac{\lambda}{2}\left(x^T H x-\delta\right)+\nu\left(b^T x+c\right)

    Because of strong duality,

    :math:`p^*=\min_x\max_{\lambda \geq 0, \nu \geq 0} \mathcal{L}(x, \lambda, \nu)`

    .. math::
        :label: cpo-eq-52

        \nabla_x \mathcal{L}(x, \lambda, \nu)=\lambda H x+(g+\nu b)

    Plug in :math:`x^*`,

    :math:`H \in \mathbb{S}^n \Rightarrow H^T=H \Rightarrow\left(H^{-1}\right)^T=H^{-1}`

    .. math::
        :label: cpo-eq-53

        x^T H x
        &=\left(-\frac{1}{\lambda} H^{-1}(g+\nu b)\right)^T H\left(-\frac{1}{\lambda} H^{-1}(g+\nu b)\right)\\
        &=\frac{1}{\lambda^2}(g+\nu b)^T H^{-1}(g+\nu b) -\frac{1}{2 \lambda}(g+\nu b)^T H^{-1}(g+\nu b)\\
        &=-\frac{1}{2 \lambda}\left(g^T H^{-1} g+\nu g^T H^{-1} b+\nu b^T H^{-1} g+\nu^2 b^T H^{-1} b\right)\\
        &=-\frac{1}{2 \lambda}\left(q+2 \nu r+\nu^2 s\right)


    .. math::
        :label: cpo-eq-54

        p^*
        &=\min_x \underset{\begin{subarray}{c} \lambda \geq 0 \\ \nu \geq 0\end{subarray}}{\max}
        \; g^T x + \frac{\lambda}{2} \left( x^T H x - \delta \right) + \nu \left(b^Tx +c \right) \\
        &\xlongequal[duality]{strong} \underset{\begin{subarray}{c} \lambda \geq 0 \\ \nu \geq 0\end{subarray}}{\max} \min_x  \; \frac{\lambda}{2} x^T H x + \left(g + \nu b\right)^T x + \left( \nu c - \frac{1}{2} \lambda \delta \right)\\
        & \;\;\; \implies x^* = -\frac{1}{\lambda} H^{-1} \left(g + \nu b \right) ~~~ \nabla_x \mathcal L(x,\lambda, \nu) =0\\
        &\xlongequal{\text{Plug in } x^*} \underset{\begin{subarray}{c} \lambda \geq 0 \\ \nu \geq 0\end{subarray}}{\max}  \; -\frac{1}{2\lambda} \left(g + \nu b \right)^T H^{-1} \left(g + \nu b \right) + \left( \nu c - \frac{1}{2} \lambda \delta \right)\\
        &\xlongequal[s \doteq b^T H^{-1} b]{
            q \doteq g^T H^{-1} g,
            r \doteq g^T H^{-1} b
        } \underset{\begin{subarray}{c} \lambda \geq 0 \\ \nu \geq 0\end{subarray}}{\max}  \; -\frac{1}{2\lambda} \left(q + 2 \nu r + \nu^2 s\right) + \left( \nu c - \frac{1}{2} \lambda \delta \right)\\
        & \;\;\; \implies \frac {\partial\mathcal L}{\partial\nu} = -\frac{1}{2\lambda}\left( 2r + 2 \nu s \right) + c \\
        &~~ \text{Optimizing single-variable convex quadratic function over } \mathbb R_+ \\
        & \;\;\; \implies \nu = \left(\frac{\lambda c - r}{s} \right)_+ \\
        &= \max_{\lambda \geq 0} \;  \left\{ \begin{array}{ll}
        \frac{1}{2\lambda} \left(\frac{r^2}{s} -q\right) + \frac{\lambda}{2}\left(\frac{c^2}{s} - \delta\right) - \frac{rc}{s}  & \text{if } \lambda \in \Lambda_a  \\
        -\frac{1}{2} \left(\frac{q}{\lambda}  + \lambda \delta\right) & \text{if } \lambda \in \Lambda_b
        \end{array}\right.\\
        &~~~~ \text{where} \begin{array}{ll}
        \Lambda_a \doteq \{\lambda | \lambda c - r  > 0, \;\; \lambda \geq 0\}, \\ \Lambda_b \doteq \{\lambda | \lambda c - r \leq 0, \;\; \lambda \geq 0\}
        \end{array}


    :math:`\lambda \in \Lambda_a \Rightarrow \nu>0`, then plug in
    :math:`\nu=\frac{\lambda c-r}{s} ; \lambda \in \Lambda_a \Rightarrow \nu \leq 0`,
    then plug in :math:`\nu=0`
