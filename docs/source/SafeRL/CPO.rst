Constrained Policy Optimization
===============================

Quick Facts
-----------

.. card::
    :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3
    :class-body: sd-font-weight-bold

    #. CPO is an :bdg-success-line:`on-policy` algorithm.
    #. CPO can be used for environments with both :bdg-success-line:`discrete` and :bdg-success-line:`continuous` action spaces.
    #. CPO can be thought of as being :bdg-success-line:`TRPO in SafeRL areas` .
    #. The OmniSafe implementation of CPO support :bdg-success-line:`parallelization`.

------

.. contents:: Table of Contents
    :depth: 3

CPO Theorem
-----------

Background
~~~~~~~~~~

**Constrained policy optimization (CPO)** is a policy search algorithm for constrained reinforcement learning with
guarantees for near-constraint satisfaction at each iteration.
Motivated by TRPO( :doc:`../BaseRL/TRPO`).
CPO develops surrogate functions to be good local approximations for objectives and constraints and easy to estimate using samples from current policy.
Moreover, it provides tighter bounds for policy search using trust regions.

.. note::

    CPO is the **first general-purpose policy search algorithm** for safe reinforcement learning with guarantees for near-constraint satisfaction at each iteration.

CPO trains neural network policies for high-dimensional control while making guarantees about policy behavior throughout training.
CPO aims to provide an approach for policy search in continuous CMDP.
It uses the result from TRPO and NPG to derive a policy improvement step that guarantees both an increase in reward and satisfaction of constraints.
Although CPO is slightly inferior in performance, it provides a solid theoretical foundation for solving constrained optimization problems in the field of safe reinforcement learning.

.. note::

    CPO is very complex in terms of implementation, but omnisafe provides a highly readable code implementation to help you get up to speed quickly.

------

Optimization Objective
~~~~~~~~~~~~~~~~~~~~~~

In the previous chapters, we introduced that TRPO solves the following optimization problems:

.. math::
    :nowrap:

    \begin{eqnarray}
        &&\pi_{k+1}=\arg\max_{\pi \in \Pi_{\boldsymbol{\theta}}}J^R(\pi)\\
        \text{s.t.}\quad&&D(\pi,\pi_k)\le\delta\tag{1}
    \end{eqnarray}

where :math:`\Pi_{\boldsymbol{\theta}} \subseteq \Pi` denotes the set of parametrized policies with parameters :math:`\boldsymbol{\theta}`, and :math:`D` is some distance measure.
In local policy search, we additionally require policy iterates to be feasible for the CMDP, so instead of optimizing over :math:`\Pi_{\boldsymbol{\theta}}`, CPO optimizes over :math:`\Pi_{\boldsymbol{\theta}} \cap \Pi_{C}`.

.. math::
    :nowrap:

    \begin{eqnarray}
        &&\pi_{k+1} = \arg\max_{\pi \in \Pi_{\boldsymbol{\theta}}}J^R(\pi)\\
        \text{s.t.}\quad&&D(\pi,\pi_k)\le\delta\tag{2}\\
        &&J^{C_i}(\pi)\le d_i\quad i=1,...m
    \end{eqnarray}


.. note::

    This update is difficult to implement because it requires evaluating the constraint functions to determine whether a proposed policy :math:`\pi` is feasible.

CPO develops a principled approximation with a particular choice of :math:`D`, where the objective and constraints are replaced with surrogate functions.
CPO proposes that with those surrogates, the update's worst-case performance and worst-case constraint violation can be bounded with values that depend on a hyperparameter of the algorithm.

------

Policy Performance Bounds
~~~~~~~~~~~~~~~~~~~~~~~~~

CPO presents the theoretical foundation for its approach, a new bound on the difference in returns between two arbitrary policies.
The following :bdg-info-line:`Theorem 1` connects the difference in returns (or constraint costs) between two arbitrary policies to an average divergence between them.

.. _cpo-eq-3:

.. _Theorem 1:

.. card::
    :class-header: sd-bg-info sd-text-white sd-font-weight-bold
    :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3
    :class-footer: sd-font-weight-bold
    :link: cards-clickable
    :link-type: ref

    Theorem 1 (Difference between two arbitrary policies)
    ^^^
    **For any function** :math:`f : S \rightarrow \mathbb{R}` and any policies :math:`\pi` and :math:`\pi'`, define :math:`\delta_f(s,a,s') \doteq R(s,a,s') + \gamma f(s')-f(s)`,

    .. math::
        :nowrap:
        :label: cpo-eq-3 cpo-eq-4 cpo-eq-5

        \begin{eqnarray}
            \epsilon_f^{\pi'} &\doteq& \max_s \left|\mathbb{E}_{a\sim\pi'~,s'\sim P }\left[\delta_f(s,a,s')\right] \right|\tag{3}\\
            L_{\pi, f}\left(\pi'\right) &\doteq& \mathbb{E}_{\tau \sim \pi}\left[\left(\frac{\pi'(a | s)}{\pi(a|s)}-1\right)\delta_f\left(s, a, s'\right)\right]\tag{4} \\
            D_{\pi, f}^{\pm}\left(\pi^{\prime}\right) &\doteq& \frac{L_{\pi, f}\left(\pi' \right)}{1-\gamma} \pm \frac{2 \gamma \epsilon_f^{\pi'}}{(1-\gamma)^2} \mathbb{E}_{s \sim d^\pi}\left[D_{T V}\left(\pi^{\prime} \| \pi\right)[s]\right]\tag{5}
        \end{eqnarray}

    where :math:`D_{T V}\left(\pi'|| \pi\right)[s]=\frac{1}{2} \sum_a\left|\pi'(a|s)-\pi(a|s)\right|` is the total variational divergence between action distributions at :math:`s`.
    The conclusion is as follows: :ref:`(11) <cpo-eq-3>`

    .. math:: D_{\pi, f}^{+}\left(\pi'\right) \geq J\left(\pi'\right)-J(\pi) \geq D_{\pi, f}^{-}\left(\pi'\right)\tag{6}

    Furthermore, the bounds are tight (when :math:`\pi=\pi^{\prime}`, all three expressions are identically zero).
    +++
    The proof of the :bdg-info-line:`Theorem 1`` can be seen in the :bdg-info:`Appendix`, click on this :bdg-info-line:`card` to jump to view.

By picking :math:`f=V_\pi`, we obtain a :bdg-info-line:`Corollary 1`, :bdg-info-line:`Corollary 2`, :bdg-info-line:`Corollary 3` below:

.. _Corollary 1:

.. _Corollary 2:

.. tab-set::

    .. tab-item:: Corollary 1

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info sd-border-{3} sd-shadow-sm sd-rounded-3
            :class-footer: sd-font-weight-bold

            Corollary 1
            ^^^
            For any policies :math:`\pi'`, :math:`\pi`, with :math:`\epsilon_{\pi'}=\max _s|\mathbb{E}_{a \sim \pi'}[A_\pi(s, a)]|`, the following bound holds:

            .. math:: J^R\left(\pi^{\prime}\right)-J^R(\pi) \geq \frac{1}{1-\gamma} \mathbb{E}_{s \sim d^\pi\,a \sim \pi'}\left[A^R_\pi(s, a)-\frac{2 \gamma \epsilon_{\pi'}}{1-\gamma} D_{T V}\left(\pi' \| \pi\right)[s]\right]\tag{7}

    .. tab-item:: Corollary 2

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card:  sd-outline-info sd-border-{3} sd-shadow-sm sd-rounded-3
            :class-footer: sd-font-weight-bold

            Corollary 2
            ^^^
            For any policies :math:`\pi'` and :math:`\pi`,
            with :math:`\epsilon^{C_i}_{\pi'}=\max _s|E_{a \sim \pi^{\prime}}[A^{C_i}_\pi(s, a)]|`

            the following bound holds:

            .. math:: J^{C_i}\left(\pi^{\prime}\right)-J^{C_i}(\pi) \geq \frac{1}{1-\gamma} \mathbb{E}_{s \sim d^\pi a \sim \pi'}\left[A^{C_i}_\pi(s, a)-\frac{2 \gamma \epsilon^{C_i}_{\pi'}}{1-\gamma} D_{T V}\left(\pi' \| \pi\right)[s]\right]\tag{8}

    .. tab-item:: Corollary 3

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info sd-border-{3} sd-shadow-sm sd-rounded-3
            :class-footer: sd-font-weight-bold

            Corollary 3
            ^^^
            Trust region methods prefer to constrain the KL-divergence between policies, so CPO use Pinsker's inequality to connect the :math:`D_{TV}` with :math:`D_{KL}`

            .. math:: D_{TV}(p \| q) \leq \sqrt{D_{KL}(p \| q) / 2}\tag{9}

            Combining this with Jensen's inequality, we obtain our final :bdg-info-line:`Corollary 3` :
            In bound :bdg-ref-info-line:`Theorem 1<Theorem 1>` , :bdg-ref-info-line:`Corollary 1<Corollary 1>`, :bdg-ref-info-line:`Corollary 2<Corollary 2>`,
            make the substitution:

            .. math:: \mathbb{E}_{s \sim d^\pi}\left[D_{T V}\left(\pi'|| \pi\right)[s]\right] \rightarrow \sqrt{\frac{1}{2} \mathbb{E}_{s \sim d^\pi}\left[D_{K L}\left(\pi^{\prime} \| \pi\right)[s]\right]}\tag{10}


------

Trust Region Methods
~~~~~~~~~~~~~~~~~~~~

For parameterized stationary policy, trust region algorithms for reinforcement learning have policy updates of the following form:

.. _cpo-eq-11:

.. math::
    :nowrap:
    :label: cpo-eq-11

    \begin{eqnarray}
        &&\boldsymbol{\theta}_{k+1}=\arg\max_{\pi \in \Pi_{\boldsymbol{\theta}}} \mathbb{E}_{\substack{s \sim d_{\pi_k}\\a \sim \pi}}[A^R_{\boldsymbol{\theta}_k}(s, a)]\\
        \text{s.t.}\quad &&\bar{D}_{K L}\left(\pi \| \pi_k\right) \le \delta\tag{11}
    \end{eqnarray}


where :math:`\bar{D}_{K L}(\pi \| \pi_k)=\mathbb{E}_{s \sim \pi_k}[D_{K L}(\pi \| \pi_k)[s]]` and :math:`\delta \ge 0` is the step size.
The set :math:`\left\{\pi_{\boldsymbol{\theta}} \in \Pi_{\boldsymbol{\theta}}: \bar{D}_{K L}\left(\pi \| \pi'\right) \leq \delta\right\}` is called trust region.
The success motivation for this update is that,
it approximates optimizing the lower bound on policy performance given in :bdg-info-line:`Corollary 1`, which would guarantee monotonic performance improvements.

.. _cpo-eq-12:

.. math::
    :nowrap:

    \begin{eqnarray}
        &&\pi_{k+1}=\arg \max _{\pi \in \Pi_{\boldsymbol{\theta}}} \mathbb{E}_{\substack{s \sim d_{\pi_k}\\a \sim \pi}}[A^R_{\pi_k}(s, a)]\\
        \text{s.t.} \quad &&J^{C_i}\left(\pi_k\right) \leq d_i-\frac{1}{1-\gamma} \mathbb{E}_{\substack{s \sim d_{\pi_k} \\ a \sim \pi}}\left[A^{C_i}_{\pi_k}(s, a)\right] \quad \forall i \tag{12} \\
        &&\bar{D}_{K L}\left(\pi \| \pi_k\right) \leq \delta
    \end{eqnarray}

.. note::
    In a word, CPO proposes the final optimization problem, which uses a trust region instead of penalties on policy divergence to enable larger step sizes.

------

Worst Performance of CPO Update
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we will introduce the propositions proposed by the CPO, one describes the worst-case performance degradation guarantee that depends on :math:`\delta`, and the other discusses the worst-case constraint violation in the CPO update.

.. tab-set::

    .. tab-item:: Proposition 1

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3
            :class-footer: sd-font-weight-bold

            Trust Region Update Performance
            ^^^
            Suppose :math:`\pi_k, \pi_{k+1}` are related by :ref:`(11) <cpo-eq-11>`, and that :math:`\pi_k \in \Pi_{\boldsymbol{\theta}}`.
            A lower bound on the policy performance difference between :math:`\pi_k` and :math:`\pi_{k+1}` is:

            .. math::

                \begin{aligned}
                    J^{R}\left(\pi_{k+1}\right)-J^{R}(\pi_{k}) \geq \frac{-\sqrt{2 \delta} \gamma \epsilon^R_{\pi_{k+1}}}{(1-\gamma)^2}
                \end{aligned}

            where :math:`\epsilon^R_{\pi_{k+1}}=\max_s\left|\mathbb{E}_{a \sim \pi_{k+1}}\left[A^R_{\pi_k}(s, a)\right]\right|`.

    .. tab-item:: Proposition 2

        .. card::
            :class-header: sd-bg-info sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3
            :class-footer: sd-font-weight-bold

            CPO Update Worst-Case Constraint Violation
            ^^^
            Suppose :math:`\pi_k, \pi_{k+1}` are related by :ref:`(11) <cpo-eq-11>`, and that :math:`\pi_k \in \Pi_{\boldsymbol{\theta}}`.
            An upper bound on the :math:`C_i`-return of :math:`\pi_{k+1}` is:

            .. math::

                \begin{aligned}
                    J^{C_i}\left(\pi_{k+1}\right) \leq d_i+\frac{\sqrt{2 \delta} \gamma \epsilon^{C_i}_{\pi_{k+1}}}{(1-\gamma)^2}
                \end{aligned}

            where :math:`\epsilon^{C_i}_{\pi_{k+1}}=\max _s\left|\mathbb{E}_{a \sim \pi_{k+1}}\left[A^{C_i}_{\pi_k}(s, a)\right]\right|`

------

Summary
~~~~~~~

We mainly introduce the essential inequalities in CPO.
Based on those inequalities, CPO presents optimization problems that ultimately need to be solved and propose two proposition about the worst case in the CPO update.
Next section, we will discuss how to solve this problem practically.
It is expected that you may be confused when you first read these theoretical derivation processes, and we have given detailed proof of the above formulas in the appendix, which we believe you can understand by reading them a few times.

------

Practical Implementation
------------------------

.. grid:: 2

    .. grid-item-card::
        :class-item: sd-font-weight-bold
        :columns: 12 4 4 6
        :class-header: sd-bg-success sd-text-white sd-font-weight-bold
        :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3

        Overview
        ^^^
        In this section, we show how CPO implements an approximation to the update :ref:`(12) <cpo-eq-12>` that can be efficiently computed, even when optimizing policies with thousands of parameters.
        To address the issue of approximation and sampling errors that arise in practice and the potential violations described by Proposition 2, CPO proposes to tighten the constraints by constraining the upper bounds of the extra costs instead of the extra costs themselves.

    .. grid-item-card::
        :class-item: sd-font-weight-bold sd-fs-6
        :columns: 12 8 8 6
        :class-header: sd-bg-success sd-text-white sd-font-weight-bold
        :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3

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

For policies with high-dimensional parameter spaces like neural networks, :ref:`(12) <cpo-eq-12>` can be impractical to solve directly because of the computational cost.

.. hint::
    However, for small step sizes :math:`\delta`, the objective and cost constraints are well-approximated by linearizing around :math:`\pi_k`, and the KL-Divergence constraint is well-approximated by second-order expansion.

Denoting the gradient of the objective as :math:`g`, the gradient of constraint :math:`i` as :math:`b_i`, the Hessian of the KL-divergence as :math:`H`, and defining :math:`c_i=J^{C_i}\left(\pi_k\right)-d_i`, the approximation to :ref:`(12) <cpo-eq-12>` is:

.. _cpo-eq-13:

.. math::
    :nowrap:

    \begin{eqnarray}
        &&\boldsymbol{\theta}_{k+1}=\arg \max _{\boldsymbol{\theta}} g^T\left(\boldsymbol{\theta}-\boldsymbol{\theta}_k\right)\\
        \text{s.t.}\quad  &&c_i+b_i^T\left(\boldsymbol{\theta}-\boldsymbol{\theta}_k\right) \leq 0 ~~~ i=1, \ldots m \tag{13}\\
        &&\frac{1}{2}\left(\boldsymbol{\theta}-\boldsymbol{\theta}_k\right)^T H\left(\boldsymbol{\theta}-\boldsymbol{\theta}_k\right) \leq \delta
    \end{eqnarray}

With :math:`B=\left[b_1, \ldots, b_m\right]` and :math:`c=\left[c_1, \ldots, c_m\right]^T`, a dual to :ref:`(13) <cpo-eq-13>` can be express as:

.. math:: \max_{\lambda \geq 0, \nu \geq 0} \frac{-1}{2 \lambda}\left(g^T H^{-1} g-2 r^T v+v^T S v\right)+v^T c-\frac{\lambda \delta}{2}

where :math:`r=g^T H^{-1} B, S=B^T H^{-1} B`. If :math:`\lambda^*, v^*` are a solution to the dual, the solution to the primal is

.. _cpo-eq-14:

.. math::
    :nowrap:

    \begin{eqnarray}
        {\boldsymbol{\theta}}^*={\boldsymbol{\theta}}_k+\frac{1}{\lambda^*} H^{-1}\left(g-B v^*\right)\tag{14}
    \end{eqnarray}

In a word, CPO solves the dual for :math:`\lambda^*, \nu^*` and uses it to propose the policy update :ref:`(14) <cpo-eq-14>`, thus solving :ref:`(12) <cpo-eq-12>` in a particular way.
In the experiment, CPO also uses two tricks to promise the update's performance.

.. warning::
    Because of the approximation error, the proposed update may not satisfy the constraints in :ref:`(12) <cpo-eq-12>`; a backtracking line search is used to ensure surrogate constraint satisfaction.

For high-dimensional policies, it is impractically expensive to invert the FIM.
This poses a challenge for computing :math:`\mathrm{H}^{-1} \mathrm{~g}` and :math:`H^{-1} b`, which appear in the dual.
Like TRPO, CPO computes them approximately using the conjugate gradient method.

------

.. _Feasibility:

Feasibility
~~~~~~~~~~~

Due to approximation errors, CPO may take a bad step and produce an infeasible iterate :math:`\pi_k`.
CPO recovers the update from an infeasible case by proposing an update to decrease the constraint value purely:

.. math:: \boldsymbol{\theta}^*=\boldsymbol{\theta}_k-\sqrt{\frac{2 \delta}{b^T H^{-1} b}} H^{-1} b\tag{15}

As before, this is followed by a line search. This approach is principled, because it uses the limiting search direction as the intersection of the trust region and the constraint region shrinks to zero.

------

.. _Tightening_Constraints_via_Cost_Shaping:

Tightening Constraints via Cost Shaping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To build a factor of safety into the algorithm to minimize the chance of constraint violations, CPO chooses to constrain upper bounds on the original constraints,
:math:`C_i^{+}`, instead of the original constraints themselves. CPO does this by cost shaping:

.. math:: C_i^{+}\left(s, a, s^{\prime}\right)=C_i\left(s, a, s^{\prime}\right)+\triangle_i\left(s, a, s^{\prime}\right)\tag{16}

where :math:`\delta_i: S \times A \times S \rightarrow R_{+}`\  correlates in some useful way with :math:`C_i`.
Because CPO has only one constraint, it partitions states into safe and unsafe states, and the agent suffers a safety cost of 1 for being in an unsafe state.
CPO chooses :math:`\triangle` to be the probability of entering an unsafe state within a fixed time horizon, according to a learned model that is updated at each iteration.
This choice confers the additional benefit of smoothing out sparse constraints.

------

.. _Code_with_OmniSafe:

Code with OmniSafe
~~~~~~~~~~~~~~~~~~

Quick start
"""""""""""

.. card::
    :class-header: sd-bg-success sd-text-white sd-font-weight-bold
    :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3 sd-font-weight-bold
    :class-footer: sd-font-weight-bold

    Run CPO in Omnisafe
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

                env = omnisafe.Env('SafetyPointGoal1-v0')

                agent = omnisafe.Agent('CPO', env)
                agent.learn()

                obs = env.reset()
                for i in range(1000):
                    action, _states = agent.predict(obs, deterministic=True)
                    obs, reward, cost, done, info = env.step(action)
                    env.render()
                    if done:
                        obs = env.reset()
                env.close()

        .. tab-item:: Config dict style

            .. code-block:: python
                :linenos:

                import omnisafe

                env = omnisafe.Env('SafetyPointGoal1-v0')

                custom_dict = {'epochs': 1, 'data_dir': './runs'}
                agent = omnisafe.Agent('CPO', env, custom_cfgs=custom_dict)
                agent.learn()

                obs = env.reset()
                for i in range(1000):
                    action, _states = agent.predict(obs, deterministic=True)
                    obs, reward, done, info = env.step(action)
                    env.render()
                    if done:
                        obs = env.reset()
                env.close()

        .. tab-item:: Terminal config style

            We use ``train_on_policy.py`` as the entrance file. You can train the agent with CPO simply using ``train_on_policy.py``, with arguments about CPO and enviroments does the training.
            For example, to run CPO in SafetyPointGoal1-v0 , with 4 cpu cores and seed 0, you can use the following command:

            .. code-block:: bash
                :linenos:

                cd omnisafe/examples
                python train_on_policy.py --env-id SafetyPointGoal1-v0 --algo CPO --parallel 5 --epochs 1


------

Here are the documentation of CPO in PyTorch version.


Architecture of functions
"""""""""""""""""""""""""

- ``cpo.learn()``

  - ``env.roll_out()``
  - ``cpo.update()``

    - ``cpo.buf.get()``
    - ``cpo.update_policy_net()``

      - ``Fvp()``
      - ``conjugate_gradients()``
      - ``search_step_size()``

    - ``cpo.update_cost_net()``
    - ``cpo.update_value_net()``

- ``cpo.log()``

------

Documentation of basic functions
""""""""""""""""""""""""""""""""

.. card-carousel:: 3

    .. card::
        :class-header: sd-bg-success sd-text-white sd-font-weight-bold
        :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3 sd-font-weight-bold
        :class-footer: sd-font-weight-bold

        env.roll_out()
        ^^^
        Collect data and store to experience buffer.

    .. card::
        :class-header: sd-bg-success sd-text-white sd-font-weight-bold
        :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3 sd-font-weight-bold
        :class-footer: sd-font-weight-bold

        cpo.update()
        ^^^
        Update actor, critic, running statistics

    .. card::
        :class-header: sd-bg-success sd-text-white sd-font-weight-bold
        :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3 sd-font-weight-bold
        :class-footer: sd-font-weight-bold

        cpo.buf.get()
        ^^^
        Call this at the end of an epoch to get all of the data from the buffer

    .. card::
        :class-header: sd-bg-success sd-text-white sd-font-weight-bold
        :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3 sd-font-weight-bold
        :class-footer: sd-font-weight-bold

        cpo.update_value_net()
        ^^^
        Update Critic network for estimating reward.

    .. card::
        :class-header: sd-bg-success sd-text-white sd-font-weight-bold
        :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3 sd-font-weight-bold
        :class-footer: sd-font-weight-bold

        cpo.update_cost_net()
        ^^^
        Update Critic network for estimating cost.

    .. card::
        :class-header: sd-bg-success sd-text-white sd-font-weight-bold
        :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3 sd-font-weight-bold
        :class-footer: sd-font-weight-bold

        cpo.log()
        ^^^
        Get the trainning log and show the performance of the algorithm

Documentation of new functions
""""""""""""""""""""""""""""""

.. tab-set::

    .. tab-item:: cpo.update_policy_net()

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            cpo.update_policy_net()
            ^^^
            Update the policy network, flowing the next steps:

            (1) Get the policy reward performance gradient g (flat as vector)

            .. code-block:: python
                :linenos:

                self.pi_optimizer.zero_grad()
                loss_pi, pi_info = self.compute_loss_pi(data=data)
                loss_pi.backward()
                g_flat = get_flat_gradients_from(self.ac.pi.net)
                g_flat *= -1


            (2) Get the policy cost performance gradient b (flat as vector)

            .. code-block:: python
                :linenos:

                self.pi_optimizer.zero_grad()
                loss_cost, _ = self.compute_loss_cost_performance(data=data)
                loss_cost.backward()
                b_flat = get_flat_gradients_from(self.ac.pi.net)


            (3) Build the Hessian-vector product based on an approximation of the KL-divergence, using ``conjugate_gradients``

            .. code-block:: python
                :linenos:

                p = conjugate_gradients(self.Fvp, b_flat, self.cg_iters)
                q = xHx
                r = g_flat.dot(p)  # g^T H^{-1} b
                s = b_flat.dot(p)  # b^T H^{-1} b

            (4) Divide the optimization case into 5 kinds to compute.

            (5) Determine step direction and apply SGD step after grads where set (By ``search_step_size()``)

            .. code-block:: python
                :linenos:

                final_step_dir, accept_step = self.search_step_size(
                    step_dir,
                    g_flat,
                    c=c,
                    optim_case=optim_case,
                    p_dist=p_dist,
                    data=data,
                    total_steps=20,
                )

            (6) Update actor network parameters

            .. code-block:: python
                :linenos:

                new_theta = theta_old + final_step_dir
                set_param_values_to_model(self.ac.pi.net, new_theta)

    .. tab-item:: cpo.search_step_size()

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            cpo.search_step_size()
            ^^^
            CPO algorithm performs line-search to ensure constraint satisfaction for rewards and costs, flowing the next steps:

            (1) Calculate the expected reward improvement.

            .. code-block:: python
               :linenos:

               expected_rew_improve = g_flat.dot(step_dir)

            (2) Performs line-search to find a step improve the surrogate while not violating trust region.

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

            - Step only if surrogate is improved and within the trust region.

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

Parameters
""""""""""

.. tab-set::

    .. tab-item:: Specific Parameters

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3 sd-font-weight-bold
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
            :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            Basic parameters
            ^^^
            -  algo (string): The name of algorithm corresponding to current class,
               it does not actually affect any things which happen in the following.
            -  actor (string): The type of network in actor, discrete or continuous.
            -  model_cfgs (dictionary) : successrmation about actor and critic's net work configuration,
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
                        ``v``    ``nn.Module``        Gives the current estimate of **V** for states in ``s``.
                        ``pi``   ``nn.Module``        Deterministically or continuously computes an action from the agent,
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
            :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3 sd-font-weight-bold
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
            :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3 sd-font-weight-bold
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
            -  standardized_reward (int):  Use standarized reward or not.
            -  standardized_cost (bool): Use standarized cost or not.

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

Proof of theorem 1 (Difference between two arbitrarily policies)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Our analysis will begin with the discounted future future state distribution, :math:`d_\pi`, which is defined as:

.. math:: d_\pi(s)=(1-\gamma) \sum_{t=0}^{\infty} \gamma^t P\left(s_t=s|\pi\right)

Let :math:`p_\pi^t \in R^{|S|}` denote the vector with components :math:`p_\pi^t(s)=P\left(s_t=s \mid \pi\right)`, and let :math:`P_\pi \in R^{|S| \times|S|}` denote the transition matrix with components :math:`P_\pi\left(s^{\prime} \mid s\right)=\int d a P\left(s^{\prime} \mid s, a\right) \pi(a \mid s)`, which shown as below:

.. math::

    \begin{aligned}
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
    \end{aligned}

then :math:`p_\pi^t=P_\pi p_\pi^{t-1}=P_\pi^2 p_\pi^{t-2}=\ldots=P_\pi^t \mu`, where :math:`\mu` represents the state distribution of the system at the moment.
That is, the initial state distribution, then :math:`d_\pi` can then be rewritten as:

.. math::

    \begin{aligned}
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
    \end{aligned}

.. _cpo-eq-17:

.. math::
    :nowrap:

    \begin{eqnarray}
        d_\pi=(1-\gamma) \sum_{t=0}^{\infty} \gamma^t p_\pi^t=(1-\gamma)\left(1-\gamma P_\pi\right)^{-1} \mu\tag{17}
    \end{eqnarray}

.. tab-set::

    .. tab-item:: Lemma 1

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3
            :class-footer: sd-font-weight-bold

            Lemma 1
            ^^^
            For any function :math:`f: S \rightarrow \mathbb{R}` and any policy :math:`\pi` :

            .. math:: (1-\gamma) E_{s \sim \mu}[f(s)]+E_{\tau \sim \pi}\left[\gamma f\left(s'\right)\right]-E_{s \sim d_\pi}[f(s)]=0

            where :math:`\tau \sim \pi` denotes :math:`s \sim d_\pi, a \sim \pi` and :math:`s^{\prime} \sim P`.


    .. tab-item:: Lemma 2

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3
            :class-footer: sd-font-weight-bold

            Lemma 2
            ^^^
            For any function :math:`f: S \rightarrow \mathbb{R}` and any policies
            :math:`\pi` and :math:`\pi'`, define

            .. math:: L_{\pi, f}\left(\pi'\right)\doteq \mathbb{E}_{\tau \sim \pi}\left[\left(\frac{\pi^{\prime}(a \mid s)}{\pi(a \mid s)}-1\right)\left(R\left(s, a, s^{\prime}\right)+\gamma f\left(s^{\prime}\right)-f(s)\right)\right]

            and :math:`\epsilon_f^{\pi^{\prime}}\doteq \max_s \left|\mathbb{E}_{\substack{a \sim \pi , s'\sim P}} \left[R\left(s, a, s^{\prime}\right)+\gamma f\left(s^{\prime}\right)-f(s)\right]\right|`.
            Then the following bounds hold:

            .. math::

               \begin{aligned}
               &J\left(\pi'\right)-J(\pi) \geq \frac{1}{1-\gamma}\left(L_{\pi, f}\left(\pi'\right)-2 \epsilon_f^{\pi'} D_{T V}\left(d_\pi \| d_{\pi^{\prime}}\right)\right) \\
               &J\left(\pi^{\prime}\right)-J(\pi) \leq \frac{1}{1-\gamma}\left(L_{\pi, f}\left(\pi'\right)+2 \epsilon_f^{\pi'} D_{T V}\left(d_\pi \| d_{\pi'}\right)\right)
               \end{aligned}

            where :math:`D_{T V}` is the total variational divergence. Furthermore, the bounds are tight when :math:`\pi^{\prime}=\pi`, and the LHS and RHS are identically zero.

    .. tab-item:: Lemma 3

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3
            :class-footer: sd-font-weight-bold

            Lemma 3
            ^^^
            The divergence between discounted future state visitation
            distributions, :math:`\Vert d_{\pi'}-d_\pi\Vert_1`, is bounded by an
            average divergence of the policies :math:`\pi` and :math:`\pi` :

            .. math::

               \begin{aligned}
                   \Vert d_{\pi'}-d_\pi\Vert_1 \leq \frac{2 \gamma}{1-\gamma} \mathbb{E}_{s \sim d_\pi}\left[D_{T V}\left(\pi^{\prime} \| \pi\right)[s]\right]
               \end{aligned}

            where :math:`D_{\mathrm{TV}}(\pi' \| \pi)[s] = \frac{1}{2}\sum_a \Vert\pi'(a|s) - \pi(a|s)\Vert`

    .. tab-item:: Corollary 4

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3
            :class-footer: sd-font-weight-bold

            Corollary 4
            ^^^
            Define the matrices
            :math:`G \doteq\left(I-\gamma P_\pi\right)^{-1}, \bar{G} \doteq\left(I-\gamma P_{\pi^{\prime}}\right)^{-1}`,
            and :math:`\Delta=P_{\pi^{\prime}}-P_\pi`. Then:

            .. math::

               \begin{aligned}
               G^{-1}-\bar{G}^{-1} &=\left(I-\gamma P_\pi\right)-\left(I-\gamma P_{\pi^{\prime}}\right) \\
               G^{-1}-\bar{G}^{-1} &=\gamma \Delta \\
               \bar{G}\left(G^{-1}-\bar{G}^{-1}\right) G &=\gamma \bar{G} \Delta G \\
               \bar{G}-G &=\gamma \bar{G} \Delta G
               \end{aligned}

            Thus, with :ref:`(17) <cpo-eq-17>`

            .. math::
              :nowrap:

              \begin{eqnarray}
                   d^{\pi^{\prime}}-d^\pi &=&(1-\gamma)(\bar{G}-G) \mu \\
                   &=&\gamma(1-\gamma) \bar{G} \Delta G \mu\tag{19}\\
                   &=&\gamma \bar{G} \Delta d^\pi
              \end{eqnarray}

    .. tab-item:: Corollary 5

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3
            :class-footer: sd-font-weight-bold

            Corollary 5
            ^^^
            .. math:: \left\|P_{\pi^{\prime}}\right\|_1=\max _{s \in \mathcal{S}}\left\{\sum_{s^{\prime} \in \mathcal{S}} P_\pi\left(s^{\prime} \mid s\right)\right\}=1

Begin with the bounds from :bdg-info-line:`Lemma 2` and bound the divergence by :bdg-info-line:`Lemma 3`, :bdg-info-line:`Theorem 1` can be finally proved.

.. _cpo-eq-18:

.. tab-set::

    .. tab-item:: Proof of Lemma 1

        .. card::
            :class-header: sd-bg-info sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3
            :class-footer: sd-font-weight-bold

            Proof
            ^^^
            Multiply both sides of :ref:`(17) <cpo-eq-17>` by :math:`\left(I-\gamma P_\pi\right)`, we get:

            .. math:: \left(I-\gamma P_\pi\right) d_\pi=(1-\gamma) \mu

            Then take the inner product with the vector :math:`f \in \mathbb{R}^{|S|}` and notice that the vector :math:`f`
            can be arbitrarily picked.

            .. math:: <f,\left(I-\gamma P_\pi\right) d_\pi>=<f,(1-\gamma) \mu>

            Both sides of the above equation can be rewritten separately by:

            .. math::

                begin{aligned}
                    &<f,\left(I-\gamma P_\pi\right) d_\pi>=\left[\sum_s f(s) d_\pi(s)\right]-\\
                    &\left[\sum_{s^{\prime}} f\left(s^{\prime}\right) \gamma \sum_s \sum_a \pi(a \mid s) P\left(s^{\prime} \mid s, a\right) d_\pi(s)\right] \\
                    &=\mathbb{E}_{s \sim d_\pi}[f(s)]-\mathbb{E}_{\tau \sim \pi}\left[\gamma f\left(s^{\prime}\right)\right]
                end{aligned}

            .. math::

                \begin{aligned}
                    <f,(1-\gamma) \mu>=\sum_s f(s)(1-\gamma) \mu(s)=(1-\gamma) \mathbb{E}_{s \sim \mu}[f(s)]
                \end{aligned}

            Finally, we obtain:

            .. math:: (1-\gamma) \mathbb{E}_{s \sim \mu}[f(s)]+\mathbb{E}_{\tau \sim \pi}\left[\gamma f\left(s^{\prime}\right)\right]-\mathbb{E}_{s \sim d_\pi}[f(s)] = 0

            .. note::

                **Supplementary details**

                .. math::

                    \begin{aligned}
                        d^\pi &=(1-\gamma)\left(I-\gamma P_\pi\right)^{-1} \mu \\\left(I-\gamma P_\pi\right) d^\pi &=(1-\gamma)  \mu \\ \int_{s \in \mathcal{S}}\left(I-\gamma P_\pi\right) d^\pi f(s) d s &=\int_{s \in \mathcal{S}} (1-\gamma) \mu f(s) d s \\ \int_{s \in \mathcal{S}} d^\pi f(s) d s-\int_{s \in \mathcal{S}} \gamma P_\pi  d^\pi f(s) d s &=\int_{s \in \mathcal{S}}(1-\gamma) \mu f(s) d s \\ \mathbb{E}_{s \sim d^\pi}[f(s)] -\mathbb{E}_{s \sim d^\pi, a \sim \pi, s^{\prime} \sim P}\left[\gamma f\left(s^{\prime}\right)\right] &= (1-\gamma) \mathbb{E}_{s \sim \mu}[f(s)]
                    \end{aligned}


    .. tab-item:: Proof of Lemma 2

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3
            :class-footer: sd-font-weight-bold

            Proof
            ^^^
            note that the objective function can be represented as:

            .. math::
                :nowrap:

                \begin{eqnarray}
                    J(\pi)&=&\frac{1}{1-\gamma} \mathbb{E}_{\tau \sim \pi}\left[R\left(s, a, s^{\prime}\right)\right]\tag{18}  \\
                    &=&\mathbb{E}_{s \sim \mu}[f(s)]+\frac{1}{1-\gamma} \mathbb{E}_{\tau \sim \pi}\left[R\left(s, a, s^{\prime}\right)+\gamma f\left(s^{\prime}\right)-f(s)\right]
                \end{eqnarray}

            Let :math:`\delta_f\left(s, a, s^{\prime}\right)\doteq R\left(s, a, s^{\prime}\right)+\gamma f\left(s^{\prime}\right)-f(s)`, then by :ref:`(18) <cpo-eq-18>`, we easily obtain that:

            .. math:: J\left(\pi'\right)-J(\pi)=\frac{1}{1-\gamma}\left\{\mathbb{E}_{\tau \sim \pi^{\prime}}\left[\delta_f\left(s, a, s^{\prime}\right)\right]-\mathbb{E}_{\tau \sim \pi}\left[\delta_f\left(s, a, s^{\prime}\right]\right\}\right.

            For the first term of the equation, let :math:`\bar{\delta}_f^{\pi'} \in \mathbb{R}^{|S|}` denote the vector of components :math:`\bar{\delta}_f^{\pi'}(s)=\mathbb{E}_{a \sim \pi', s' \sim P}\left[\delta_f\left(s, a, s'|s\right)\right]`, then

            .. math:: \mathbb{E}_{\tau \sim \pi'}\left[d_f\left(s, a, s'\right)\right]=<d_{\pi'}, \bar{\delta}^f_{\pi'}>=<d_\pi,\bar{\delta}^f_{\pi'}>+<d_{\pi'}-d_\pi, \hat{d}^f_{\pi'}>

            By using Holder's inequality, for any :math:`p, q \in[1, \infty]`, such that :math:`\frac{1}{p}+\frac{1}{q}=1`.
            We have

            .. math::

                \begin{aligned}
                    & \mathbb{E}_{\tau \sim \pi^{\prime}}\left[\delta_f\left(s, a, s^{\prime}\right)\right] \leq \langle d_\pi, \bar{\delta}_f^{\pi^{\prime}} \rangle+\Vert d_{\pi'}-d_\pi \Vert_p \Vert \bar{\delta}_f^{\pi'}\Vert_q  \\
                    &\mathbb{E}_{\tau \sim \pi'}\left[\delta_f\left(s, a, s'\right)\right] \geq \langle d_\pi, \bar{\delta}_f^{\pi'}\rangle-\Vert d_{\pi'}-d_\pi \Vert_p \Vert \bar{\delta}_f^{\pi'}\Vert_q
                \end{aligned}

            .. note::

                **Hölder's inequality**:

                Let :math:`(\mathcal{S}, \sum, \mu)` be a measure space and let :math:`p, q \in [1, \infty]` with :math:`\frac{1}{p} + \frac{1}{q} = 1`. Then for all measurable real- or complex-valued function :math:`f` and :math:`g` on :math:`s`, :math:`\|f g\|_1 \leq\|f\|_p\|g\|_q`.

                If, in addition, :math:`p, q \in(1, \infty)` and :math:`f \in L^p(\mu)` and :math:`g \in L^q(\mu)`, then
                Hölder's inequality becomes an equality if and only if :math:`|f|^p` and :math:`|g|^q` are linearly dependent in :math:`L^1(\mu)`, meaning that there exist real numbers :math:`\alpha, \beta \geq 0`, not both of them zero, such that :math:`\alpha|f|^p=\beta|g|^q \mu`-almost everywhere.

            The last step is to observe that, by the importance of sampling identity,

            .. math::

                \begin{aligned}
                \left\langle d^\pi, \bar{\delta}_f^{\pi^{\prime}}\right\rangle &=\underset{s \sim d^\pi, a \sim \pi^{\prime}, s^{\prime} \sim P}{\mathbb{E}}\left[\delta_f\left(s, a, s^{\prime}\right)\right] \\
                &=\underset{s \sim d^\pi, a \sim \pi, s^{\prime} \sim P}{\mathbb{E}}\left[\left(\frac{\pi^{\prime}(a \mid s)}{\pi(a \mid s)}\right) \delta_f\left(s, a, s^{\prime}\right)\right]
                \end{aligned}

            After grouping terms, the bounds are obtained.

            .. math::

                \begin{aligned}
                &\left\langle d^\pi, \bar{\delta}_f^{\pi^{\prime}}\right\rangle \pm\Vert d^{\pi^{\prime}}-d^\pi\Vert_p\Vert\bar{\delta}_f^{\pi^{\prime}}\Vert_q\\
                &=\mathbb{E}_{\substack{s \sim d^\pi\\ a \sim \pi\\ s^{\prime} \sim P}}\left[\left(\frac{\pi'(a|s)}{\pi(a|s)}\right) \delta_f\left(s, a, s^{\prime}\right)\right] \pm 2 \epsilon_f^{\pi^{\prime}} D_{T V}\left(d_{\pi'} \| d_\pi\right)
                \end{aligned}

            .. math::

                \begin{aligned}
                    &J(\pi')-J(\pi)\\
                    &\leq \frac{1}{1-\gamma}\mathbb{E}_{\substack{s \sim d^\pi \\ a \sim \pi \\ s' \sim P}}[(\frac{\pi^{\prime}(a|s)}{\pi(a|s)}) \delta_f(s, a, s^{\prime})]+2 \epsilon_f^{\pi^{\prime}} D_{T V}(d^{\pi^{\prime}} \| d^\pi)-\mathbb{E}_{\substack{s \sim d^\pi \\ a \sim \pi \\ s' \sim P}}[\delta_f(s, a, s^{\prime})]\\
                    &=\frac{1}{1-\gamma}(\mathbb{E}_{\substack{s \sim d^\pi \\ a \sim \pi \\ s' \sim P}}[(\frac{\pi^{\prime}(a|s)}{\pi(a|s)}) \delta_f(s, a, s^{\prime})]-\mathbb{E}_{\substack{s \sim d^\pi \\ a \sim \pi \\ s' \sim P}}[\delta_f(s, a, s^{\prime})]+2 \epsilon_f^{\pi^{\prime}} D_{T V}(d^{\pi^{\prime}} \| d^\pi))\\
                    &=\frac{1}{1-\gamma}(\mathbb{E}_{\substack{s \sim d^\pi \\ a \sim \pi \\ s' \sim P}}[(\frac{\pi^{\prime}(a \mid s)}{\pi(a \mid s)}-1) \delta_f(s, a, s^{\prime})]+2 \epsilon_f^{\pi^{\prime}} D_{T V}(d^{\pi^{\prime}} \| d^\pi))
                \end{aligned}

            The lower bound is the same.

            .. math::

                \begin{aligned}
                J\left(\pi^{\prime}\right)-J(\pi) \geq \mathbb{E}_{\substack{s \sim d^\pi \\ a \sim \pi \\ s' \sim P}}\left[\left(\frac{\pi^{\prime}(a|s)}{\pi(a|s)}-1\right) \delta_f\left(s, a, s^{\prime}\right)\right]-2 \epsilon_f^{\pi^{\prime}} D_{T V}\left(d^{\pi^{\prime}} \| d^\pi\right)
                \end{aligned}

    .. tab-item:: Proof of Lemma 3

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3
            :class-footer: sd-font-weight-bold

            Proof
            ^^^
            First, using Corollary 4, we obtain

            .. math::

                \begin{aligned}
                    \left\|d^{\pi^{\prime}}-d^\pi\right\|_1 &=\gamma\left\|\bar{G} \Delta d^\pi\right\|_1 \\
                    & \leq \gamma\|\bar{G}\|_1\left\|\Delta d^\pi\right\|_1
                \end{aligned}

            Meanwhile,

            .. math::

                \begin{aligned}
                    \|\bar{G}\|_1 &=\left\|\left(I-\gamma P_{\pi^{\prime}}\right)^{-1}\right\|_1 \\ &=\left\|\sum_{t=0}^{\infty} \gamma^t P_{\pi^{\prime}}^t\right\|_1 \\ & \leq \sum_{t=0}^{\infty} \gamma^t\left\|P_{\pi^{\prime}}\right\|_1^t \\ &=\left(1-\gamma\left\|P_{\pi^{\prime}}\right\|_1\right)^{-1} \\ &=(1-\gamma)^{-1}
                \end{aligned}

            And, using Corollary 5, we have,

            .. math::
                :nowrap:

                    \begin{eqnarray}
                        \Delta d^\pi\left[s^{\prime}\right] &=& \sum_s \Delta\left(s^{\prime} \mid s\right) d^\pi(s) \\
                        &=&\sum_s \left\{ P_{\pi^{\prime}}\left(s^{\prime} \mid s\right)-P_\pi\left(s^{\prime} \mid s\right)  \right\} d_{\pi}(s)\tag{20} \\
                        &=&\sum_s \left\{ P\left(s^{\prime} \mid s, a\right) \pi^{\prime}(a \mid s)-P\left(s^{\prime} \mid s, a\right) \pi(a \mid s)  \right\} d_{\pi}(s)\\
                        &=&\sum_s \left\{ P\left(s^{\prime} \mid s, a\right)\left[\pi^{\prime}(a \mid s)-\pi(a \mid s)\right] \right\} d_{\pi}(s)
                    \end{eqnarray}

            .. note::

                **Total variation distance of probability measures**

                :math:`\Vert d_{\pi'}-d_\pi \Vert_1=\sum_{a \in \mathcal{A}}\left|d_{\pi_{{\boldsymbol{\theta}}^{\prime}}}(a|s)-d_{\pi_{\boldsymbol{\theta}}}(a|s)\right|=2 D_{\mathrm{TV}}\left(d_{\pi_{{\boldsymbol{\theta}}'}}, d_\pi\right)[s]`

            Finally, using :ref:`(20) <cpo-eq-18>`, we obtain,

            .. math::

                \begin{aligned}
                \left\|\Delta d^\pi\right\|_1 &=\sum_{s^{\prime}}\left|\sum_s \Delta\left(s^{\prime} \mid s\right) d^\pi(s)\right| \\ & \leq \sum_{s, s^{\prime}}\left|\Delta\left(s^{\prime} \mid s\right)\right| d^\pi(s) \\ &=\sum_{s, s^{\prime}}\left|\sum_a P\left(s^{\prime} \mid s, a\right)\left(\pi^{\prime}(a \mid s)-\pi(a \mid s)\right)\right| d^\pi(s) \\ & \leq \sum_{s, a, s^{\prime}} P\left(s^{\prime} \mid s, a\right)\left|\pi^{\prime}(a \mid s)-\pi(a \mid s)\right| d^\pi(s) \\ &=\sum_{s^{\prime}} P\left(s^{\prime} \mid s, a\right) \sum_{s, a}\left|\pi^{\prime}(a \mid s)-\pi(a \mid s)\right| d^\pi(s) \\ &=\sum_{s, a}\left|\pi^{\prime}(a \mid s)-\pi(a \mid s)\right| d^\pi(s) \\ &=\sum_a \underset{s \sim d^\pi}{ } \mathbb{E}^{\prime}|(a \mid s)-\pi(a \mid s)| \\ &=2 \underset{s \sim d^\pi}{\mathbb{E}}\left[D_{T V}\left(\pi^{\prime}|| \pi\right)[s]\right]
                \end{aligned}

------

Proof of Analytical Solution to LQCLP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. card::
    :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
    :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3
    :class-footer: sd-font-weight-bold

    Theorem 2 (Optimizing Linear Objective with Linear, Quadratic Constraints)
    ^^^
    Consider the problem

    .. math::
        :nowrap:

        \begin{eqnarray}
            p^*&=&\min_x g^T x \\
            \text { s.t. } b^T x+c &\leq& 0 \\
            x^T H x &\leq& \delta
        \end{eqnarray}

    where
    :math:`g, b, x \in \mathbb{R}^n, c, \delta \in \mathbb{R}, \delta>0, H \in \mathbb{S}^n`,
    and :math:`H \succ 0`. When there is at least one strictly feasible
    point, the optimal point :math:`x^*` satisfies

    .. math::

        \begin{aligned}
        x^*=-\frac{1}{\lambda^*} H^{-1}\left(g+\nu^* b\right)
        \end{aligned}

    where :math:`\lambda^*` and :math:`\nu^*` are defined by

    .. math::

        \begin{aligned}
        &\nu^*=\left(\frac{\lambda^* c-r}{s}\right)_{+}, \\
        &\lambda^*=\arg \max _{\lambda \geq 0} \begin{cases}f_a(\lambda) \doteq \frac{1}{2 \lambda}\left(\frac{r^2}{s}-q\right)+\frac{\lambda}{2}\left(\frac{c^2}{s}-\delta\right)-\frac{r c}{s} & \text { if } \lambda c-r>0 \\
        f_b(\lambda) \doteq-\frac{1}{2}\left(\frac{q}{\lambda}+\lambda \delta\right) & \text { otherwise }\end{cases}
        \end{aligned}

    with :math:`q=g^T H^{-1} g, r=g^T H^{-1} b`, and
    :math:`s=b^T H^{-1} b`.

    Furthermore, let
    :math:`\Lambda_a \doteq\{\lambda \mid \lambda c-r>0, \lambda \geq 0\}`,
    and
    :math:`\Lambda_b \doteq\{\lambda \mid \lambda c-r \leq 0, \lambda \geq 0\}`.
    The value of :math:`\lambda^*` satisfies

    .. math:: \lambda^* \in\left\{\lambda_a^* \doteq \operatorname{Proj}\left(\sqrt{\frac{q-r^2 / s}{\delta-c^2 / s}}, \Lambda_a\right), \lambda_b^* \doteq \operatorname{Proj}\left(\sqrt{\frac{q}{\delta}}, \Lambda_b\right)\right\}

    with :math:`\lambda^*=\lambda_a^*` if
    :math:`f_a\left(\lambda_a^*\right)>f_b\left(\lambda_b^*\right)` and
    :math:`\lambda = \lambda_b^*` otherwise, and
    :math:`\operatorname{Proj}(a, S)` is the projection of a point
    :math:`x` on to a set :math:`S`. hint: the projection of a point
    :math:`x \in \mathbb{R}` onto a convex segment of
    :math:`\mathbb{R},[a, b]`, has value
    :math:`\operatorname{Proj}(x,[a, b])=\max (a, \min (b, x))`.

.. dropdown:: Proof for Theorem 2 (Click here)
    :color: info
    :class-body: sd-border-{3}

    This is a convex optimization problem. When there is at least one strictly feasible point, strong duality holds by Slater's theorem.
    We exploit strong duality to solve the problem analytically.
    irst using the method of Lagrange multipliers, :math:`\exists \lambda, \mu \geq 0`

    .. math:: \mathcal{L}(x, \lambda, \nu)=g^T x+\frac{\lambda}{2}\left(x^T H x-\delta\right)+\nu\left(b^T x+c\right)

    Because of strong duality,

    :math:`p^*=\min_x\max_{\lambda \geq 0, \nu \geq 0} \mathcal{L}(x, \lambda, \nu)`

    .. math:: \nabla_x \mathcal{L}(x, \lambda, \nu)=\lambda H x+(g+\nu b)

    Plug in :math:`x^*`,

    :math:`H \in \mathbb{S}^n \Rightarrow H^T=H \Rightarrow\left(H^{-1}\right)^T=H^{-1}`

    .. math::

        \begin{aligned}
        x^T H x
        &=\left(-\frac{1}{\lambda} H^{-1}(g+\nu b)\right)^T H\left(-\frac{1}{\lambda} H^{-1}(g+\nu b)\right)\\
        &=\frac{1}{\lambda^2}(g+\nu b)^T H^{-1}(g+\nu b) -\frac{1}{2 \lambda}(g+\nu b)^T H^{-1}(g+\nu b)\\
        &=-\frac{1}{2 \lambda}\left(g^T H^{-1} g+\nu g^T H^{-1} b+\nu b^T H^{-1} g+\nu^2 b^T H^{-1} b\right)\\
        &=-\frac{1}{2 \lambda}\left(q+2 \nu r+\nu^2 s\right)
        \end{aligned}

    .. math::

        \begin{aligned}
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
        \end{aligned}

    :math:`\lambda \in \Lambda_a \Rightarrow \nu>0`, then plug in
    :math:`\nu=\frac{\lambda c-r}{s} ; \lambda \in \Lambda_a \Rightarrow \nu \leq 0`,
    then plug in :math:`\nu=0`
