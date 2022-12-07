Projection-Based Constrained Policy Optimization
================================================

Quick Facts
-----------

.. card::
    :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3
    :class-body: sd-font-weight-bold

    #. PCPO is an :bdg-success-line:`on-policy` algorithm.
    #. PCPO can be used for environments with both :bdg-success-line:`discrete` and :bdg-success-line:`continuous` action spaces.
    #. PCPO is an improvement work done on the basis of :bdg-success-line:`CPO` .
    #. The OmniSafe implementation of PCPO support :bdg-success-line:`parallelization`.

------

.. contents:: Table of Contents
   :depth: 3

PCPO Theorem
------------

Background
~~~~~~~~~~

**Projection-Based Constrained Policy Optimization (PCPO)** is an iterative method for optimizing policy in a **two-stage process**: the first stage performs a local reward improvement update, while the second stage reconciles any constraint violation by projecting the policy back onto the constraint set.

PCPO is an improvement work done on the basis of **CPO** (:doc:`../SafeRL/CPO`).
It provides a lower bound on reward improvement,
and an upper bound on constraint violation, for each policy update just like CPO does.
PCPO further characterizes the convergence of PCPO based on two different metrics: :math:`L2` norm and KL divergence.

In a word, PCPO is a CPO-based algorithm dedicated to solving problem of learning control policies that optimize a reward function, while satisfying constraints due to considerations of safety, fairness, or other costs.

.. note::

    If you have not previously learned the CPO type of algorithm, in order to facilitate your complete understanding of the PCPO algorithm ideas introduced in this section, we strongly recommend that you read this article after reading the CPO tutorial (:doc:`./CPO`) we wrote.

------

Optimization Objective
~~~~~~~~~~~~~~~~~~~~~~

In the previous chapters, you learned that CPO solves the following optimization problems:

.. math::
    :nowrap:

    \begin{eqnarray}
        &&\pi_{k+1} = \arg\max_{\pi \in \Pi_{\boldsymbol{\theta}}}J^R(\pi)\\
        \text{s.t.}\quad&&D(\pi,\pi_k)\le\delta\tag{1}\\
        &&J^{C_i}(\pi)\le d_i\quad i=1,...m
    \end{eqnarray}

where :math:`\Pi_{\theta}\subseteq\Pi` denotes the set of parametrized policies with parameters :math:`\theta`, and :math:`D` is some distance measure.
In local policy search for CMDPs, we additionally require policy iterates to be feasible for the CMDP, so instead of optimizing over :math:`\Pi_{\theta}`, PCPO optimizes over :math:`\Pi_{\theta}\cap\Pi_{C}`.
Next, we will introduce you to how PCPO solves the above optimization problems.
In order for you to have a clearer understanding, we hope that you will read the next section with the following questions:

.. card::
    :class-header: sd-bg-primary sd-text-white sd-font-weight-bold
    :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3 sd-font-weight-bold

    Questions
    ^^^
    -  What is two-stage policy update and how?

    -  What is performance bound for PCPO and how PCPO get it?

    -  How PCPO practically solve the optimal problem?

------

Two-stage Policy Update
~~~~~~~~~~~~~~~~~~~~~~~

PCPO performs policy update in **two stages**.
The first stage is :bdg-ref-info-line:`Reward Improvement Stage<two stage update>` which maximizes reward using a trust region optimization method without constraints.
This might result in a new intermediate policy that does not satisfy the constraints.
The second stage named :bdg-ref-info-line:`Projection Stage<two stage update>` reconciles the constraint violation (if any) by projecting the policy back onto the constraint set, i.e., choosing the policy in the constraint set that is closest to the selected intermediate policy.
Next, we will describe how PCPO completes the two-stage update.

.. _`two stage update`:

.. tab-set::

    .. tab-item:: Stage 1

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info sd-border-{3} sd-shadow-sm sd-rounded-3
            :class-footer: sd-font-weight-bold

            Reward Improvement Stage
            ^^^
            First, PCPO optimizes the reward function by maximizing the reward advantage function :math:`A_{\pi}(s,a)` subject to KL-Divergence constraint.
            This constraints the intermediate policy :math:`\pi_{k+\frac12}` to be within a :math:`\delta`-neighbourhood of :math:`\pi_{k}`:

            .. math::
                :nowrap:

                \begin{eqnarray}
                &&\pi_{k+\frac12}=\underset{\pi}{\arg\max}\underset{s\sim d^{\pi_k}, a\sim\pi}{\mathbb{E}}[A^R_{\pi_k}(s,a)]\tag{2}\\
                \text{s.t.}\quad &&\underset{s\sim d^{\pi_k}}{\mathbb{E}}[D_{KL}(\pi||\pi_k)[s]]\le\delta\nonumber
                \end{eqnarray}

            This update rule with the trust region is called **TRPO** (sees in :doc:`../BaseRL/TRPO`).
            It constraints the policy changes to a divergence neighborhood and guarantees reward improvement.

    .. tab-item:: Stage 2

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card:  sd-outline-info sd-border-{3} sd-shadow-sm sd-rounded-3
            :class-footer: sd-font-weight-bold

            Projection Stage
            ^^^
            Second, PCPO projects the intermediate policy :math:`\pi_{k+\frac12}` onto the constraint set by minimizing a distance measure :math:`D` between :math:`\pi_{k+\frac12}` and :math:`\pi`:

            .. math::
                :nowrap:

                \begin{eqnarray}
                &&\pi_{k+1}=\underset{\pi}{\arg\min}\quad D(\pi,\pi_{k+\frac12})\tag{3}\\
                \text{s.t.}\quad &&J^C\left(\pi_k\right)+\underset{\substack{s \sim d^{\pi_k} , a \sim \pi}}{\mathbb{E}}\left[A^C_{\pi_k}(s, a)\right] \leq d
                \end{eqnarray}

The :bdg-ref-info-line:`Projection Stage<two stage update>` ensures that the constraint-satisfying policy :math:`\pi_{k+1}` is close to :math:`\pi_{k+\frac{1}{2}}`.
The :bdg-ref-info-line:`Reward Improvement Stage<two stage update>` ensures that the agent's updates are in the direction of maximizing rewards, so as not to violate the step size of distance measure :math:`D`.
:bdg-ref-info-line:`Projection Stage<two stage update>` causes the agent to update in the direction of satisfying the constraint while avoiding crossing :math:`D` as much as possible.

------

Policy Performance Bounds
~~~~~~~~~~~~~~~~~~~~~~~~~

In safety-critical applications, **how worse the performance of a system evolves when applying a learning algorithm** is an important issue.
For the two cases where the agent satisfies the constraint and does not satisfy the constraint, PCPO provides worst-case performance bound respectively.

.. _`performance bound`:

.. tab-set::

    .. tab-item:: Theorem 1

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info sd-border-{3} sd-shadow-sm sd-rounded-3
            :class-footer: sd-font-weight-bold
            :link: cards-clickable
            :link-type: ref

            Worst-case Bound on Updating Constraint-satisfying Policies
            ^^^
            Define :math:`\epsilon_{\pi_{k+1}}^{R}\doteq \max\limits_{s}\big|\mathbb{E}_{a\sim\pi_{k+1}}[A^{R}_{\pi_{k}}(s,a)]\big|`, and :math:`\epsilon_{\pi_{k+1}}^{C}\doteq \max\limits_{s}\big|\mathbb{E}_{a\sim\pi_{k+1}}[A^{C}_{\pi_{k}}(s,a)]\big|`.
            If the current policy :math:`\pi_k` satisfies the constraint, then under KL divergence projection, the lower bound on reward improvement, and upper bound on constraint violation for each policy update are

            .. math::
                :nowrap:

                \begin{eqnarray}
                J^{R}(\pi_{k+1})-J^{R}(\pi_{k})&\geq&-\frac{\sqrt{2\delta}\gamma\epsilon_{\pi_{k+1}}^{R}}{(1-\gamma)^{2}}\tag{4}\\
                J^{C}(\pi_{k+1})&\leq& d+\frac{\sqrt{2\delta}\gamma\epsilon_{\pi_{k+1}}^{C}}{(1-\gamma)^{2}}\tag{5}
                \end{eqnarray}

            where :math:`\delta` is the step size in the reward improvement step.
            +++
            The proof of the :bdg-info-line:`Theorem 1` can be seen in the :bdg-info:`CPO tutorial`, click on this :bdg-info-line:`card` to jump to view.

    .. tab-item:: Theorem 2

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card:  sd-outline-info sd-border-{3} sd-shadow-sm sd-rounded-3
            :class-footer: sd-font-weight-bold
            :link: pcpo-performance-bound-proof
            :link-type: ref

            Worst-case Bound on Updating Constraint-violating Policies
            ^^^
            Define :math:`\epsilon_{\pi_{k+1}}^{R}\doteq \max\limits_{s}\big|\mathbb{E}_{a\sim\pi_{k+1}}[A^{R}_{\pi_{k}}(s,a)]\big|`, :math:`\epsilon_{\pi_{k+1}}^{C}\doteq \max\limits_{s}\big|\mathbb{E}_{a\sim\pi_{k+1}}[A^{C}_{\pi_{k}}(s,a)]\big|`, :math:`b^{+}\doteq \max(0,J^{C}(\pi_k)-d),` and :math:`\alpha_{KL} \doteq \frac{1}{2a^T\boldsymbol{H}^{-1}a},` where :math:`a` is the gradient of the cost advantage function and :math:`\boldsymbol{H}` is the Hessian of the KL divergence constraint.
            If the current policy :math:`\pi_k` violates the constraint, then under KL divergence projection, the lower bound on reward improvement and the upper bound on constraint violation for each policy update are

            .. math::
                :nowrap:

                \begin{eqnarray}
                    J^{R}(\pi_{k+1})-J^{R}(\pi_{k})\geq&-\frac{\sqrt{2(\delta+{b^+}^{2}\alpha_\mathrm{KL})}\gamma\epsilon_{\pi_{k+1}}^{R}}{(1-\gamma)^{2}}\tag{6}\\
                    J^{C}(\pi_{k+1})\leq& ~d+\frac{\sqrt{2(\delta+{b^+}^{2}\alpha_\mathrm{KL})}\gamma\epsilon_{\pi_{k+1}}^{C}}{(1-\gamma)^{2}}\tag{7}
                \end{eqnarray}

            where :math:`\delta` is the step size in the reward improvement step.
            +++
            The proof of the :bdg-info-line:`Theorem 2` can be seen in the :bdg-info:`Appendix`, click on this :bdg-info-line:`card` to jump to view.

------

Practical Implementation
------------------------

Implementation of Two-stage Update
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a large neural network policy with hundreds of thousands of parameters, directly solving for the PCPO update in :ref:`(2) <two stage update>` and :ref:`(3) <two stage update>` is impractical due to the computational cost.
PCPO proposes that with a small step size :math:`\delta`, the reward function and constraints and the KL divergence constraint in the reward improvement step can be approximated with a first order expansion, while the KL divergence measure in the projection step can also be approximated with a second order expansion.

.. tab-set::

    .. tab-item:: Implementation of Stage 1

        .. card::
            :class-header: sd-bg-success  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info sd-border-{3} sd-shadow-sm sd-rounded-3
            :class-footer: sd-font-weight-bold
            :link: pcpo-code-with-omnisafe
            :link-type: ref

            Reward Improvement Stage
            ^^^
            Define:

            :math:`g\doteq\nabla_\theta\underset{\substack{s\sim d^{\pi_k}a\sim \pi}}{\mathbb{E}}[A_{\pi_k}^{R}(s,a)]` is the gradient of the reward advantage function,

            :math:`a\doteq\nabla_\theta\underset{\substack{s\sim d^{\pi_k}a\sim \pi}}{\mathbb{E}}[A_{\pi_k}^{C}(s,a)]` is the gradient of the cost advantage function,

            where :math:`\boldsymbol{H}_{i,j}\doteq \frac{\partial^2 \underset{s\sim d^{\pi_{k}}}{\mathbb{E}}\big[KL(\pi ||\pi_{k})[s]\big]}{\partial \theta_j\partial \theta_j}` is the Hessian of the KL divergence constraint (:math:`\boldsymbol{H}` is also called the Fisher information matrix. It is symmetric positive semi-definite), :math:`b\doteq J^{C}(\pi_k)-d` is the constraint violation of the policy :math:`\pi_{k}`, and :math:`\theta` is the parameter of the policy. PCPO linearizes the objective function at :math:`\pi_k` subject to second order approximation of the KL divergence constraint in order to obtain the following updates:

            .. math::

                \begin{eqnarray}
                &&\theta_{k+\frac{1}{2}} = \underset{\theta}{\arg\max}g^{T}(\theta-\theta_k)  \tag{8}\\
                \text{s.t.}\quad &&\frac{1}{2}(\theta-\theta_{k})^{T}\boldsymbol{H}(\theta-\theta_k)\le \delta . \label{eq:update1}
                \end{eqnarray}

            In fact, the above problem is essentially an optimization problem presented in TRPO, which can be completely solved using the method we introduced in the TRPO tutorial.
            +++
            The Omnisafe code of the :bdg-success-line:`Implementation of Stage I` can be seen in the :bdg-success:`Code with Omnisafe`, click on this :bdg-success-line:`card` to jump to view.

    .. tab-item:: Implementation of Stage 2

        .. card::
            :class-header: sd-bg-success  sd-text-white sd-font-weight-bold
            :class-card:  sd-outline-info sd-border-{3} sd-shadow-sm sd-rounded-3
            :class-footer: sd-font-weight-bold
            :link: pcpo-code-with-omnisafe
            :link-type: ref

            Projection Stage
            ^^^
            PCPO provides a selection reference for distance measures: if the projection is defined in the parameter space, :math:`L2` norm projection is selected, while if the projection is defined in the probability space, KL divergence projection is better.
            This can be approximated through the second order expansion.
            Again, PCPO linearizes the cost constraint at :math:`\pi_{k}`.
            This gives the following update for the projection step:

            .. math::

                \begin{eqnarray}
                &&\theta_{k+1} =\underset{\theta}{\arg\min}\frac{1}{2}(\theta-{\theta}_{k+\frac{1}{2}})^{T}\boldsymbol{L}(\theta-{\theta}_{k+\frac{1}{2}})\tag{9}\\
                \text{s.t.}\quad && a^{T}(\theta-\theta_{k})+b\leq 0
                \end{eqnarray}

            where :math:`\boldsymbol{L}=\boldsymbol{I}` for :math:`L2` norm projection, and :math:`\boldsymbol{L}=\boldsymbol{H}` for KL divergence projection.
            +++
            The Omnisafe code of the :bdg-success-line:`Implementation of Stage II` can be seen in the :bdg-success:`Code with Omnisafe`, click on this :bdg-success-line:`card` to jump to view.

PCPO solves Problem :ref:`(6) <performance bound>` and Problem :ref:`(7) <performance bound>` using :bdg-success-line:`convex programming`, see detailed in :bdg-ref-success:`Appendix<convex-programming>`.

For each policy update:

.. _pcpo-eq-10:

.. math::
    :nowrap:

    \begin{eqnarray}
    \theta_{k+1}=\theta_{k}+&\sqrt{\frac{2\delta}{g^T\boldsymbol{H}^{-1}g}}\boldsymbol{H}^{-1}g
    -\max\left(0,\frac{\sqrt{\frac{2\delta}{g^T\boldsymbol{H}^{-1}g}}a^{T}\boldsymbol{H}^{-1}g+b}{a^T\boldsymbol{L}^{-1}a}\right)\boldsymbol{L}^{-1}a\tag{10}
    \end{eqnarray}

.. hint::

    :math:`\boldsymbol{H}` is assumed invertible and PCPO requires to invert :math:`\boldsymbol{H}`, which is impractical for huge neural network policies.
    Hence it use the conjugate gradient method.
    (See appendix for a discussion of the tradeoff between the approximation error, and computational efficiency of the conjugate gradient method.)

.. grid:: 2

    .. grid-item::
        :columns: 12 6 6 5

        .. tab-set::

            .. tab-item:: Question I
                :sync: key1

                .. card::
                    :class-header: sd-bg-success  sd-text-white sd-font-weight-bold
                    :class-card:  sd-outline-info sd-border-{3} sd-shadow-sm sd-rounded-3 sd-font-weight-bold

                    Question
                    ^^^
                    Is using linear approximation to the constraint set enough to ensure constraint satisfaction since the real constraint set is maybe non-convex?

            .. tab-item:: Question II
                :sync: key2

                .. card::
                    :class-header: sd-bg-success  sd-text-white sd-font-weight-bold
                    :class-card:  sd-outline-info sd-border-{3} sd-shadow-sm sd-rounded-3 sd-font-weight-bold

                    Question
                    ^^^
                    Can PCPO solve the multi-constraint problem? And how PCPO actually do that?

    .. grid-item::
        :columns: 12 6 6 7

        .. tab-set::

            .. tab-item:: Answer I
                :sync: key1

                .. card::
                    :class-header: sd-bg-primary  sd-text-white sd-font-weight-bold
                    :class-card:  sd-outline-info sd-border-{3} sd-shadow-sm sd-rounded-3 sd-font-weight-bold

                    Answer
                    ^^^
                    In fact, if the step size :math:`\delta` is small, then the linearization of the constraint set is accurate enough to locally approximate it.

            .. tab-item:: Answer II
                :sync: key2

                .. card::
                    :class-header: sd-bg-primary  sd-text-white sd-font-weight-bold
                    :class-card:  sd-outline-info sd-border-{3} sd-shadow-sm sd-rounded-3 sd-font-weight-bold

                    Answer
                    ^^^
                    By sequentially projecting onto each of the sets,
                    the update in :ref:`(7) <performance bound>` can be extended by using alternating projections.

------

Analysis
~~~~~~~~

The update rule in :ref:`(7) <performance bound>` shows that the difference between PCPO with KL divergence and :math:`L2` norm projections is **the cost update direction**, leading to a difference in reward improvement.
These two projections converge to different stationary points with different convergence rates related to the smallest and largest singular values of the Fisher information matrix shown in :bdg-info-line:`Theorem 3`.
PCPO assumes that: PCPO minimizes the negative reward objective function :math:`f: R^n \rightarrow R` .
The function :math:`f` is :math:`L`-smooth and twice continuously differentiable over the closed and convex constraint set :math:`\mathcal{C}`.

.. _Theorem 3:

.. card::
    :class-header: sd-bg-info sd-text-white sd-font-weight-bold
    :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3
    :class-footer: sd-font-weight-bold
    :link: pcpo-theorem3-proof
    :link-type: ref

    Theorem 3
    ^^^
    Let :math:`\eta\doteq \sqrt{\frac{2\delta}{g^{T}\boldsymbol{H}^{-1}g}}` in :ref:`(7) <performance bound>`, where :math:`\delta` is the step size for reward improvement, :math:`g` is the gradient of :math:`f`, and :math:`\boldsymbol{H}` is the Fisher information matrix.
    Let :math:`\sigma_\mathrm{max}(\boldsymbol{H})` be the largest singular value of :math:`\boldsymbol{H}`, and :math:`a` be the gradient of cost advantage function in :ref:`(7) <performance bound>`.
    Then PCPO with KL divergence projection converges to a stationary point either inside the constraint set or in the boundary of the constraint set.
    In the latter case, the Lagrangian constraint :math:`g=-\alpha a, \alpha\geq0` holds.
    Moreover, at step :math:`k+1` the objective value satisfies

    .. math:: f(\theta_{k+1})\leq f(\theta_{k})+||\theta_{k+1}-\theta_{k}||^2_{-\frac{1}{\eta}\boldsymbol{H}+\frac{L}{2}\boldsymbol{I}}.

    PCPO with :math:`L2`  norm projection converges to a stationary point either inside the constraint set or in the boundary of the constraint set.
    In the latter case, the Lagrangian constraint :math:`\boldsymbol{H}^{-1}g=-\alpha a, \alpha\geq0` holds.
    If :math:`\sigma_\mathrm{max}(\boldsymbol{H})\leq1,` then a step :math:`k+1` objective value satisfies.

    .. math:: f(\theta_{k+1})\leq f(\theta_{k})+(\frac{L}{2}-\frac{1}{\eta})||\theta_{k+1}-\theta_{k}||^2_2.
    +++
    The proof of the :bdg-info-line:`Theorem 3` can be seen in the :bdg-info:`Appendix`, click on this :bdg-info-line:`card` to jump to view.

:bdg-info-line:`Theorem 3` shows that in the stationary point :math:`g` is a line that points to the opposite direction of :math:`a`.
Further, the improvement of the objective value is affected by the singular value of the Fisher information matrix.
Specifically, the objective of KL divergence projection decreases when :math:`\frac{L\eta}{2}\boldsymbol{I}\prec\boldsymbol{H},` implying that :math:`\sigma_\mathrm{min}(\boldsymbol{H})> \frac{L\eta}{2}`.
And the objective of :math:`L2` norm projection decreases when :math:`\eta<\frac{2}{L},` implying that condition number of :math:`\boldsymbol{H}` is upper bounded: :math:`\frac{\sigma_\mathrm{max}(\boldsymbol{H})}{\sigma_\mathrm{min}(\boldsymbol{H})}<\frac{2||g||^2_2}{L^2\delta}`.
Observing the singular values of the Fisher information matrix allows us to adaptively choose the appropriate projection and hence achieve objective improvement.
In the supplemental material, we further use an example to compare the optimization trajectories and stationary points of KL divergence and :math:`L2` norm projections.

------

.. _pcpo-code-with-omnisafe:

Code with OmniSafe
~~~~~~~~~~~~~~~~~~

Quick start
"""""""""""


.. card::
    :class-header: sd-bg-success sd-text-white sd-font-weight-bold
    :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3 sd-font-weight-bold
    :class-footer: sd-font-weight-bold

    Run PCPO in Omnisafe
    ^^^
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

                agent = omnisafe.Agent('PCPO', env)
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
                agent = omnisafe.Agent('PCPO', env, custom_cfgs=custom_dict)
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

                We use ``train_on_policy.py`` as the entrance file.
                You can train the agent with PCPO simply using ``train_on_policy.py``,
                with arguments about PCPO and enviroments does the training.
                For example, to run PCPO in SafetyPointGoal1-v0 , with 4 cpu cores and seed 0, you can use the following command:

                .. code-block:: guess
                    :linenos:

                    cd omnisafe/examples
                    ython train_on_policy.py --env-id SafetyPointGoal1-v0 --algo PCPO --parallel 5 --epochs 1


------

Architecture of functions
"""""""""""""""""""""""""

-  ``pcpo.learn()``

   -  ``env.roll_out()``
   -  ``pcpo.update()``

      -  ``pcpo.buf.get()``
      -  ``pcpo.update_policy_net()``

         -  ``Fvp()``
         -  ``conjugate_gradients()``
         -  ``search_step_size()``


      -  ``pcpo.update_cost_net()``
      -  ``pcpo.update_value_net()``

-  ``pcpo.log()``

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

        pcpo.update()
        ^^^
        Update actor, critic, running statistics

    .. card::
        :class-header: sd-bg-success sd-text-white sd-font-weight-bold
        :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3 sd-font-weight-bold
        :class-footer: sd-font-weight-bold

        pcpo.buf.get()
        ^^^
        Call this at the end of an epoch to get all of the data from the buffer

    .. card::
        :class-header: sd-bg-success sd-text-white sd-font-weight-bold
        :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3 sd-font-weight-bold
        :class-footer: sd-font-weight-bold

        pcpo.update_policy_net()
        ^^^
        Update policy network in 5 kinds of optimization case

    .. card::
        :class-header: sd-bg-success sd-text-white sd-font-weight-bold
        :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3 sd-font-weight-bold
        :class-footer: sd-font-weight-bold

        pcpo.update_value_net()
        ^^^
        Update Critic network for estimating reward.

    .. card::
        :class-header: sd-bg-success sd-text-white sd-font-weight-bold
        :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3 sd-font-weight-bold
        :class-footer: sd-font-weight-bold

        pcpo.update_cost_net()
        ^^^
        Update Critic network for estimating cost.

    .. card::
        :class-header: sd-bg-success sd-text-white sd-font-weight-bold
        :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3 sd-font-weight-bold
        :class-footer: sd-font-weight-bold

        pcpo.log()
        ^^^
        Get the trainning log and show the performance of the algorithm

------

Documentation of new functions
""""""""""""""""""""""""""""""

.. tab-set::

    .. tab-item:: pcpo.update_policy_net()

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            pcpo.update_policy_net()
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

            (4) Determine step direction and apply SGD step after grads where set (By ``adjust_cpo_step_direction()``)

            .. code-block:: python
                :linenos:

                final_step_dir, accept_step = self.adjust_cpo_step_direction(
                step_dir,
                g_flat,
                c=c,
                optim_case=2,
                p_dist=p_dist,
                data=data,
                total_steps=20,
                )

            (5) Update actor network parameters

            .. code-block:: python
                :linenos:

                new_theta = theta_old + final_step_dir
                set_param_values_to_model(self.ac.pi.net, new_theta)

    .. tab-item:: pcpo.adjust_cpo_step_direction()

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            pcpo.adjust_cpo_step_direction()
            ^^^
            PCPO algorithm performs line-search to ensure constraint satisfaction for rewards and costs, flowing the next steps:

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
            -  algo (string): The name of algorithm corresponding to current class, it does not actually affect any things which happen in the following.
            -  actor (string): The type of network in actor, discrete of continuous.
            -  model_cfgs (dictionary) : successrmation about actor and critic's net work configuration,it originates from ``algo.yaml`` file to describe ``hidden layers`` , ``activation function``, ``shared_weights`` and ``weight_initialization_mode``.

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

                    ======== ================  ====================================================================
                    Name        Type              Description
                    ======== ================  ====================================================================
                    ``v``    ``nn.Module``        Gives the current estimate of **V** for states in ``s``.
                    ``pi``   ``nn.Module``        Deterministically or continuously computes an action from the agent,

                                                    conditioned on states in ``s``.
                    ======== ================  ====================================================================

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
            -  adv_estimation_method (float): Roughly what KL divergence we think is appropriate between new and old policies after an update. This will get used for early stopping. (Usually small, 0.01 or 0.05.)
            -  standardized_reward (int):  Use standarized reward or not.
            -  standardized_cost (bool): Use standarized cost or not.

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

To prove the policy performance bound when the current policy is infeasible (constraint-violating), we first prove two lemma of the KL divergence between :math:`\pi_{k}` and :math:`\pi_{k+1}` for the KL divergence projection.
We then prove the main theorem for the worst-case performance degradation.

.. tab-set::

    .. tab-item:: Lemma 1
        :sync: key1

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3
            :class-footer: sd-font-weight-bold

            Lemma 1
            ^^^
            If the current policy :math:`\pi_{k}` satisfies the constraint, the constraint set is closed and convex, the KL divergence constraint for the first step is :math:`\mathbb{E}_{s\sim d^{\pi_{k}}}\big[\mathrm{KL}(\pi_{k+\frac{1}{2}} ||\pi_{k})[s]\big]\leq \delta`, where :math:`\delta` is the step size in the reward improvement step, then under KL divergence projection, we have

            .. math:: \mathbb{E}_{s\sim d^{\pi_{k}}}\big[\mathrm{KL}(\pi_{k+1} ||\pi_{k})[s]\big]\leq \delta.


    .. tab-item:: Lemma 2
        :sync: key2

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3
            :class-footer: sd-font-weight-bold

            Lemma 2
            ^^^
            If the current policy :math:`\pi_{k}` violates the constraint, the constraint set is closed and convex, the KL divergence constraint for the first step is :math:`\mathbb{E}_{s\sim d^{\pi_{k}}}\big[\mathrm{KL}(\pi_{k+\frac{1}{2}} ||\pi_{k})[s]\big]\leq \delta`.
            where :math:`\delta` is the step size in the reward improvement step, then under the KL divergence projection, we have

            .. math:: \mathbb{E}_{s\sim d^{\pi_{k}}}\big[\mathrm{KL}(\pi_{k+1} ||\pi_{k})[s]\big]\leq \delta+{b^+}^2\alpha_\mathrm{KL},

            where :math:`\alpha_\mathrm{KL} \doteq \frac{1}{2a^T\boldsymbol{H}^{-1}a}`, :math:`a` is the gradient of the cost advantage function, :math:`\boldsymbol{H}` is the Hessian of the KL divergence constraint, and :math:`b^+\doteq\max(0,J^{C}(\pi_k)-h)`.

.. _pcpo-eq-11:

.. tab-set::

    .. tab-item:: Proof of Lemma 1
        :sync: key1

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3
            :class-footer: sd-font-weight-bold

            Proof of Lemma 1
            ^^^
            By the Bregman divergence projection inequality, :math:`\pi_{k}` being in the constraint set, and :math:`\pi_{k+1}` being the projection of the :math:`\pi_{k+\frac{1}{2}}` onto the constraint set, we have

            .. math::

                \begin{aligned}
                &\mathbb{E}_{s\sim d^{\pi_{k}}}\big[\mathrm{KL}(\pi_{k} ||\pi_{k+\frac{1}{2}})[s]\big]\geq
                \mathbb{E}_{s\sim d^{\pi_{k}}}\big[\mathrm{KL}(\pi_{k}||\pi_{k+1})[s]\big] \\
                &+
                \mathbb{E}_{s\sim d^{\pi_{k}}}\big[\mathrm{KL}(\pi_{k+1} ||\pi_{k+\frac{1}{2}})[s]\big]\\
                &\Rightarrow\delta\geq
                \mathbb{E}_{s\sim d^{\pi_{k}}}\big[\mathrm{KL}(\pi_{k} ||\pi_{k+\frac{1}{2}})[s]\big]\geq
                \mathbb{E}_{s\sim d^{\pi_{k}}}\big[\mathrm{KL}(\pi_{k}||\pi_{k+1})[s]\big].
                \end{aligned}

            The derivation uses the fact that KL divergence is always greater than zero.
            We know that KL divergence is asymptotically symmetric when updating the policy within a local neighbourhood.
            Thus, we have

            .. math::

                \delta\geq
                \mathbb{E}_{s\sim d^{\pi_{k}}}\big[\mathrm{KL}(\pi_{k+\frac{1}{2}} ||\pi_{k})[s]\big]\geq
                \mathbb{E}_{s\sim d^{\pi_{k}}}\big[\mathrm{KL}(\pi_{k+1}||\pi_{k})[s]\big].

    .. tab-item:: Proof of Lemma 2
      :sync: key2

      .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3
            :class-footer: sd-font-weight-bold

            Proof of Lemma 2
            ^^^
            We define the sublevel set of cost constraint function for the current infeasible policy :math:`\pi_k`:

            .. math:: L^{\pi_k}=\{\pi~|~J^{C}(\pi_{k})+ \mathbb{E}_{\substack{s\sim d^{\pi_{k}}\\ a\sim \pi}}[A_{\pi_k}^{C}(s,a)]\leq J^{C}(\pi_{k})\}.

            This implies that the current policy :math:`\pi_k` lies in :math:`L^{\pi_k}`, and :math:`\pi_{k+\frac{1}{2}}` is projected onto the constraint set: :math:`\{\pi~|~J^{C}(\pi_{k})+ \mathbb{E}_{\substack{s\sim d^{\pi_{k}}\\ a\sim \pi}}[A_{\pi_k}^{C}(s,a)]\leq h\}`.
            Next, we define the policy :math:`\pi_{k+1}^l` as the projection of :math:`\pi_{k+\frac{1}{2}}` onto :math:`L^{\pi_k}`.

            For these three polices :math:`\pi_k, \pi_{k+1}` and :math:`\pi_{k+1}^l`, with :math:`\varphi(x)\doteq\sum_i x_i\log x_i`, we have

            .. math::
                :nowrap:

                \begin{eqnarray}
                \delta &&\geq  \mathbb{E}_{s\sim d^{\pi_{k}}}\big[\mathrm{KL}(\pi_{k+1}^l ||\pi_{k})[s]\big]
                \\&&=\mathbb{E}_{s\sim d^{\pi_{k}}}\big[\mathrm{KL}(\pi_{k+1} ||\pi_{k})[s]\big] -\mathbb{E}_{s\sim d^{\pi_{k}}}\big[\mathrm{KL} (\pi_{k+1} ||\pi_{k+1}^l)[s]\big]\\
                &&+\mathbb{E}_{s\sim d^{\pi_{k}}}\big[(\nabla\varphi(\pi_k)-\nabla\varphi(\pi_{k+1}^{l}))^T(\pi_{k+1}-\pi_{k+1}^l)[s]\big] \nonumber \\
                \end{eqnarray}

                \begin{eqnarray}
                \Rightarrow \mathbb{E}_{s\sim d^{\pi_{k}}}\big[\mathrm{KL} (\pi_{k+1} ||\pi_{k})[s]\big]&&\leq \delta + \mathbb{E}_{s\sim d^{\pi_{k}}}\big[\mathrm{KL} (\pi_{k+1} ||\pi_{k+1}^l)[s]\big]\\
                &&- \mathbb{E}_{s\sim d^{\pi_{k}}}\big[(\nabla\varphi(\pi_k)-\nabla\varphi(\pi_{k+1}^{l}))^T(\pi_{k+1}-\pi_{k+1}^l)[s]\big]. \tag{11}
                \end{eqnarray}

            The inequality :math:`\mathbb{E}_{s\sim d^{\pi_{k}}}\big[\mathrm{KL} (\pi_{k+1}^l ||\pi_{k})[s]\big]\leq\delta` comes from that :math:`\pi_{k}` and :math:`\pi_{k+1}^l` are in :math:`L^{\pi_k}`, and :bdg-info-line:`Lemma 1`.

            If the constraint violation of the current policy :math:`\pi_k` is small, :math:`b^+` is small, :math:`\mathbb{E}_{s\sim d^{\pi_{k}}}\big[\mathrm{KL} (\pi_{k+1} ||\pi_{k+1}^l)[s]\big]` can be approximated by the second order expansion.
            By the update rule in :ref:`(7) <performance bound>`, we have

            .. math::
                :nowrap:

                \begin{eqnarray}
                \mathbb{E}_{s\sim d^{\pi_{k}}}\big[\mathrm{KL}(\pi_{k+1} ||\pi_{k+1}^l)[s]\big] &&\approx \frac{1}{2}(\theta_{k+1}-\theta_{k+1}^l)^{T}\boldsymbol{H}(\theta_{k+1}-\theta_{k+1}^l)\\
                &&=\frac{1}{2} \Big(\frac{b^+}{a^T\boldsymbol{H}^{-1}a}\boldsymbol{H}^{-1}a\Big)^T\boldsymbol{H}\Big(\frac{b^+}{a^T\boldsymbol{H}^{-1}a}\boldsymbol{H}^{-1}a\Big)\\
                &&=\frac{{b^+}^2}{2a^T\boldsymbol{H}^{-1}a}\\
                &&={b^+}^2\alpha_\mathrm{KL}, \tag{12}
                \end{eqnarray}

            where :math:`\alpha_\mathrm{KL} \doteq \frac{1}{2a^T\boldsymbol{H}^{-1}a}.`

            And since :math:`\delta` is small, we have :math:`\nabla\varphi(\pi_k)-\nabla\varphi(\pi_{k+1}^{l})\approx \mathbf{0}` given :math:`s`.
            Thus, the third term in :ref:`(10) <pcpo-eq-10>` can be eliminated.

            Combining :ref:`(10) <pcpo-eq-10>` and :ref:`(11) <pcpo-eq-11>`, we have :math:`[
            \mathbb{E}_{s\sim d^{\pi_{k}}}\big[\mathrm{KL}(\pi_{k+1}||\pi_{k})[s]\big]\leq \delta+{b^+}^2\alpha_\mathrm{KL}.]`


Now we use :bdg-info-line:`Lemma 2` to prove the :bdg-info-line:`Theorem 2`.
Following the same proof in :bdg-ref-info-line:`Theorem 1<cards-clickable>`, we complete the proof.

.. _`appendix_proof_theorem_3`:

.. _`pcpo-theorem3-proof`:

Proof of Analytical Solution to PCPO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. card::
    :class-header: sd-bg-info sd-text-white sd-font-weight-bold
    :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3

    Analytical Solution to PCPO
    ^^^
    Consider the PCPO problem. In the first step, we optimize the reward:

    .. math::
        :nowrap:

        \begin{eqnarray}
            \theta_{k+\frac{1}{2}} = &&\underset{\theta}{\arg\,min}\quad g^{T}(\theta-\theta_{k}) \\
            \text{s.t.}\quad&&\frac{1}{2}(\theta-\theta_{k})^{T}\boldsymbol{H}(\theta-\theta_{k})\leq \delta,
        \end{eqnarray}

    and in the second step, we project the policy onto the constraint set:

    .. math::
        :nowrap:

        \begin{eqnarray}
            \theta_{k+1} = &&\underset{\theta}{\arg\,min}\quad \frac{1}{2}(\theta-{\theta}_{k+\frac{1}{2}})^{T}\boldsymbol{L}(\theta-{\theta}_{k+\frac{1}{2}}) \\
            \text{s.t.}\quad &&a^{T}(\theta-\theta_{k})+b\leq 0,
        \end{eqnarray}

    where :math:`g, a, \theta \in R^n, b, \delta\in R, \delta>0,` and :math:`\boldsymbol{H},\boldsymbol{L}\in R^{n\times n}, \boldsymbol{L}=\boldsymbol{H}`, if using the KL divergence projection, and :math:`\boldsymbol{L}=\boldsymbol{I}` if using the :math:`L2`  norm projection.
    When there is at least one strictly feasible point, the optimal solution satisfies

    .. math::
        :nowrap:

        \begin{eqnarray}
        \theta_{k+1}&&=\theta_{k}+\sqrt{\frac{2\delta}{g^T\boldsymbol{H}^{-1}g}}\boldsymbol{H}^{-1}g\nonumber\\
        &&-\max(0,\frac{\sqrt{\frac{2\delta}{g^T\boldsymbol{H}^{-1}g}}a^{T}\boldsymbol{H}^{-1}g+b}{a^T\boldsymbol{L}^{-1}a})\boldsymbol{L}^{-1}a
        \end{eqnarray}

    assuming that :math:`\boldsymbol{H}` is invertible to get a unique solution.

    .. dropdown:: Proof of Analytical Solution to PCPO (Click here)
        :color: info
        :class-body: sd-border-{3}

        For the first problem, since :math:`\boldsymbol{H}` is the Fisher Information matrix, which automatically guarantees it is positive semi-definite.
        Hence it is a convex program with quadratic inequality constraints.
        Hence if the primal problem has a feasible point, then Slater's condition is satisfied and strong duality holds.
        Let :math:`\theta^{*}` and :math:`\lambda^*` denote the solutions to the primal and dual problems, respectively.
        In addition, the primal objective function is continuously differentiable.
        Hence the Karush-Kuhn-Tucker (KKT) conditions are necessary and sufficient for the optimality of :math:`\theta^{*}` and :math:`\lambda^*.`
        We now form the Lagrangian:

        .. math:: \mathcal{L}(\theta,\lambda)=-g^{T}(\theta-\theta_{k})+\lambda\Big(\frac{1}{2}(\theta-\theta_{k})^{T}\boldsymbol{H}(\theta-\theta_{k})- \delta\Big).

        And we have the following KKT conditions:

        .. _`pcpo-eq-13`:

        .. math::
            :nowrap:

            \begin{eqnarray}
                -g + \lambda^*\boldsymbol{H}\theta^{*}-\lambda^*\boldsymbol{H}\theta_{k}=0~~~~&~~~\nabla_\theta\mathcal{L}(\theta^{*},\lambda^{*})=0 \tag{13}\\
                \frac{1}{2}(\theta^{*}-\theta_{k})^{T}\boldsymbol{H}(\theta^{*}-\theta_{k})- \delta=0~~~~&~~~\nabla_\lambda\mathcal{L}(\theta^{*},\lambda^{*})=0 \tag{14}\\
                \frac{1}{2}(\theta^{*}-\theta_{k})^{T}\boldsymbol{H}(\theta^{*}-\theta_{k})-\delta\leq0~~~~&~~~\text{primal constraints}\label{KKT_3}\tag{15}\\
                \lambda^*\geq0~~~~&~~~\text{dual constraints}\tag{16}\\
                \lambda^*\Big(\frac{1}{2}(\theta^{*}-\theta_{k})^{T}\boldsymbol{H}(\theta^{*}-\theta_{k})-\delta\Big)=0~~~~&~~~\text{complementary slackness}\tag{17}
            \end{eqnarray}

        By :ref:`(13) <pcpo-eq-13>`, we have :math:`\theta^{*}=\theta_{k}+\frac{1}{\lambda^*}\boldsymbol{H}^{-1}g`.
        And by plugging :ref:`(13) <pcpo-eq-13>` into :ref:`(14) <pcpo-eq-13>`, we have :math:`\lambda^*=\sqrt{\frac{g^T\boldsymbol{H}^{-1}g}{2\delta}}`.
        Hence we have our optimal solution:

        .. _`pcpo-eq-18`:

        .. math::
            :nowrap:

            \begin{eqnarray}
            \theta_{k+\frac{1}{2}}=\theta^{*}=\theta_{k}+\sqrt{\frac{2\delta}{g^T\boldsymbol{H}^{-1}g}}\boldsymbol{H}^{-1}g \tag{18}
            \end{eqnarray}

        which also satisfies :ref:`(15) <pcpo-eq-13>`, :ref:`(16) <pcpo-eq-13>`, and :ref:`(17) <pcpo-eq-13>`.

        Following the same reasoning, we now form the Lagrangian of the second problem:

        .. math::
            :nowrap:

            \begin{eqnarray}
            \mathcal{L}(\theta,\lambda)=\frac{1}{2}(\theta-{\theta}_{k+\frac{1}{2}})^{T}\boldsymbol{L}(\theta-{\theta}_{k+\frac{1}{2}})+\lambda(a^T(\theta-\theta_{k})+b)\tag{19}
            \end{eqnarray}

        And we have the following KKT conditions:

        .. _`pcpo-eq-20`:

        .. math::
            :nowrap:

            \begin{eqnarray}
            \boldsymbol{L}\theta^*-\boldsymbol{L}\theta_{k+\frac{1}{2}}+\lambda^*a=0~~~~&~~~\nabla_\theta\mathcal{L}(\theta^{*},\lambda^{*})=0 \tag{20}  \\
                a^T(\theta^*-\theta_{k})+b=0~~~~&~~~\nabla_\lambda\mathcal{L}(\theta^{*},\lambda^{*})=0 \tag{21}  \\
                a^T(\theta^*-\theta_{k})+b\leq0~~~~&~~~\text{primal constraints}\tag{22}  \\
                \lambda^*\geq0~~~~&~~~\text{dual constraints}\tag{23}  \\
                \lambda^*(a^T(\theta^*-\theta_{k})+b)=0~~~~&~~~\text{complementary slackness}\tag{24}
            \end{eqnarray}

        By :ref:`(20) <pcpo-eq-20>`, we have :math:`\theta^{*}=\theta_{k+1}+\lambda^*\boldsymbol{L}^{-1}a`.
        And by plugging :ref:`(20) <pcpo-eq-20>` into :ref:`(21) <pcpo-eq-20>` and :ref:`(23) <pcpo-eq-20>`, we have :math:`\lambda^*=\max(0,\\ \frac{a^T(\theta_{k+\frac{1}{2}}-\theta_{k})+b}{a\boldsymbol{L}^{-1}a})`.
        Hence we have our optimal solution:

        .. _`pcpo-eq-25`:

        .. math::
            :nowrap:

            \begin{eqnarray}
            \theta_{k+1}=\theta^{*}=\theta_{k+\frac{1}{2}}-\max(0,\frac{a^T(\theta_{k+\frac{1}{2}}-\theta_{k})+b}{a^T\boldsymbol{L}^{-1}a^T})\boldsymbol{L}^{-1}a\tag{25}
            \end{eqnarray}

        which also satisfies :ref:`(22) <pcpo-eq-20>` and :ref:`(24) <pcpo-eq-20>`.
        Hence by :ref:`(18) <pcpo-eq-18>` and :ref:`(25) <pcpo-eq-25>`, we have

        .. math::
            :nowrap:

            \begin{eqnarray}
            \theta_{k+1}&&=\theta_{k}+\sqrt{\frac{2\delta}{g^T\boldsymbol{H}^{-1}g}}\boldsymbol{H}^{-1}g\\
            &&-\max(0,\frac{\sqrt{\frac{2\delta}{g^T\boldsymbol{H}^{-1}g}}a^{T}\boldsymbol{H}^{-1}g+b}{a^T\boldsymbol{L}^{-1}a})\boldsymbol{L}^{-1}a
            \end{eqnarray}

Proof of Theorem 3
~~~~~~~~~~~~~~~~~~

For our analysis, we make the following assumptions: we minimize the negative reward objective function :math:`f: R^n \rightarrow R` (We follow the convention of the literature that authors typically minimize the objective function).
The function :math:`f` is :math:`L`-smooth and twice continuously differentiable over the closed and convex constraint set :math:`\mathcal{C}`.
We have the following :bdg-info-line:`Lemma 3` to characterize the projection and for the proof of :bdg-info-line:`Theorem 3`

.. card::
    :class-header: sd-bg-info sd-text-white sd-font-weight-bold
    :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3

    Lemma 3
    ^^^
    For any :math:`\theta`, :math:`\theta^{*}=\mathrm{Proj}^{\boldsymbol{L}}_{\mathcal{C}}(\theta)` if and only if :math:`(\theta-\theta^*)^T\boldsymbol{L}(\theta'-\theta^*)\leq0, \forall\theta'\in\mathcal{C}`,
    where :math:`\mathrm{Proj}^{\boldsymbol{L}}_{\mathcal{C}}(\theta)\doteq \underset{\theta' \in \mathrm{C}}{\arg\,min}||\theta-\theta'||^2_{\boldsymbol{L}}` and :math:`\boldsymbol{L}=\boldsymbol{H}` if using the KL divergence projection, and :math:`\boldsymbol{L}=\boldsymbol{I}` if using the :math:`L2` norm projection.

    +++
    .. dropdown:: Proof of Lemma 3 (Click here)
        :color: info
        :class-body: sd-border-{3}

        :math:`(\Rightarrow)` Let
        :math:`\theta^{*}=\mathrm{Proj}^{\boldsymbol{L}}_{\mathcal{C}}(\theta)`
        for a given :math:`\theta \not\in\mathcal{C},`
        :math:`\theta'\in\mathcal{C}` be such that
        :math:`\theta'\neq\theta^*,` and :math:`\alpha\in(0,1).` Then we have

        .. _`pcpo-eq-26`:

        .. math::
            :nowrap:

            \begin{eqnarray}\label{eq:appendix_lemmaD1_0}
                \left\|\theta-\theta^*\right\|_L^2
                && \leq\left\|\theta-\left(\theta^*+\alpha\left(\theta^{\prime}-\theta^*\right)\right)\right\|_L^2 \\
                &&=\left\|\theta-\theta^*\right\|_L^2+\alpha^2\left\|\theta^{\prime}-\theta^*\right\|_{\boldsymbol{L}}^2\\
                ~~~~ &&-2\alpha\left(\theta-\theta^*\right)^T \boldsymbol{L}\left(\theta^{\prime}-\theta^*\right) \\
                && \Rightarrow\left(\theta-\theta^*\right)^T \boldsymbol{L}\left(\theta^{\prime}-\theta^*\right) \leq \frac{\alpha}{2}\left\|\theta^{\prime}-\theta^*\right\|_{\boldsymbol{L}}^2\tag{26}
            \end{eqnarray}

        Since the right hand side of :ref:`(26) <pcpo-eq-26>` can be made arbitrarily small for a given :math:`\alpha`, and hence we have:

        .. math:: (\theta-\theta^*)^T\boldsymbol{L}(\theta'-\theta^*)\leq0, \forall\theta'\in\mathcal{C}.

        Let :math:`\theta^*\in\mathcal{C}` be such that :math:`(\theta-\theta^*)^T\boldsymbol{L}(\theta'-\theta^*)\leq0, \forall\theta'\in\mathcal{C}`.
        We show that :math:`\theta^*` must be the optimal solution.
        Let :math:`\theta'\in\mathcal{C}` and :math:`\theta'\neq\theta^*`.
        Then we have

        .. math::

            \begin{split}
            &\left\|\theta-\theta^{\prime}\right\|_L^2-\left\|\theta-\theta^*\right\|_L^2\\ &=\left\|\theta-\theta^*+\theta^*-\theta^{\prime}\right\|_L^2-\left\|\theta-\theta^*\right\|_L^2 \\
            &=\left\|\theta-\theta^*\right\|_L^2+\left\|\theta^{\prime}-\theta^*\right\|_L^2-2\left(\theta-\theta^*\right)^T \boldsymbol{L}\left(\theta^{\prime}-\theta^*\right)\\
            &~~~~-\left\|\theta-\theta^*\right\|_{\boldsymbol{L}}^2 \\
            &>0 \\
            &\Rightarrow\left\|\theta-\theta^{\prime}\right\|_L^2 >\left\|\theta-\theta^*\right\|_L^2 .
            \end{split}

        Hence, :math:`\theta^*` is the optimal solution to the optimization problem, and :math:`\theta^*=\mathrm{Proj}^{\boldsymbol{L}}_{\mathcal{C}}(\theta)`.

Based on :bdg-info-line:`Lemma 3` we have the proof of following :bdg-info-line:`Theorem 3`.

.. card::
    :class-header: sd-bg-info sd-text-white sd-font-weight-bold
    :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3

    Theorem 3 (Stationary Points of PCPO with the KL divergence and :math:`L2` Norm Projections)
    ^^^
    Let :math:`\eta\doteq \sqrt{\frac{2\delta}{g^{T}\boldsymbol{H}^{-1}g}}` in :ref:`(7) <performance bound>`, where :math:`\delta` is the step size for reward improvement, :math:`g` is the gradient of :math:`f`, :math:`\boldsymbol{H}` is the Fisher information matrix.
    Let :math:`\sigma_\mathrm{max}(\boldsymbol{H})` be the largest singular value of :math:`\boldsymbol{H}`, and :math:`a` be the gradient of cost advantage function in :ref:`(7) <performance bound>`.
    Then PCPO with the KL divergence projection converges to stationary points with :math:`g\in-a` (i.e., the gradient of :math:`f` belongs to the negative gradient of the cost advantage function).
    The objective value changes by

    .. math:: f(\theta_{k+1})\leq f(\theta_{k})+||\theta_{k+1}-\theta_{k}||^2_{-\frac{1}{\eta}\boldsymbol{H}+\frac{L}{2}\boldsymbol{I}}\tag{27}


    PCPO with the :math:`L2` norm projection converges to stationary points with :math:`\boldsymbol{H}^{-1}g\in-a` (i.e., the product of the inverse of :math:`\boldsymbol{H}` and gradient of :math:`f` belongs to the negative gradient of the cost advantage function).
    If :math:`\sigma_\mathrm{max}(\boldsymbol{H})\leq1`, then the objective value changes by

    .. math:: f(\theta_{k+1})\leq f(\theta_{k})+(\frac{L}{2}-\frac{1}{\eta})||\theta_{k+1}-\theta_{k}||^2_2\tag{28}

    .. dropdown:: Proof of Theorem 3 (Click here)
        :color: info
        :class-body: sd-border-{3}

        The proof of the theorem is based on working in a Hilbert space and the non-expansive property of the projection.
        We first prove stationary points for PCPO with the KL divergence and :math:`L2` norm projections, and then prove the change of the objective value.

        When in stationary points :math:`\theta^*`, we have

        .. _`pcpo-eq-29`:

        .. math::
            :nowrap:

            \begin{eqnarray}
            \theta^{*}&&=\theta^{*}-\sqrt{\frac{2\delta}{g^T\boldsymbol{H}^{-1}g}}\boldsymbol{H}^{-1}g
            -\max\left(0,\frac{\sqrt{\frac{2\delta}{g^T\boldsymbol{H}^{-1}g}}a^{T}\boldsymbol{H}^{-1}g+b}{a^T\boldsymbol{L}^{-1}a}\right)\boldsymbol{L}^{-1}a\\
            &&\Leftrightarrow \sqrt{\frac{2\delta}{g^T\boldsymbol{H}^{-1}g}}\boldsymbol{H}^{-1}g  = -\max(0,\frac{\sqrt{\frac{2\delta}{g^T\boldsymbol{H}^{-1}g}}a^{T}\boldsymbol{H}^{-1}g+b}{a^T\boldsymbol{L}^{-1}a})\boldsymbol{L}^{-1}a\\
            &&\Leftrightarrow  \boldsymbol{H}^{-1}g \in -\boldsymbol{L}^{-1}a.
            \label{eq:appendixStationary}\tag{29}
            \end{eqnarray}

        For the KL divergence projection (:math:`\boldsymbol{L}=\boldsymbol{H}`), :ref:`(29) <pcpo-eq-29>` boils down to :math:`g\in-a`, and for the :math:`L2` norm projection (:math:`\boldsymbol{L}=\boldsymbol{I}`), :ref:`(29) <pcpo-eq-29>` is equivalent to :math:`\boldsymbol{H}^{-1}g\in-a`.

        Now we prove the second part of the theorem. Based on :bdg-info-line:`Lemma 3`, for the KL divergence projection, we have

        .. _`pcpo-eq-30`:

        .. math::
            :nowrap:

            \begin{eqnarray}
            \label{eq:appendix_converge_0}
            \left(\theta_k-\theta_{k+1}\right)^T \boldsymbol{H}\left(\theta_k-\eta \boldsymbol{H}^{-1} \boldsymbol{g}-\theta_{k+1}\right) \leq 0 \\
            \Rightarrow \boldsymbol{g}^T\left(\theta_{k+1}-\theta_k\right) \leq-\frac{1}{\eta}\left\|\theta_{k+1}-\theta_k\right\|_{\boldsymbol{H}}^2\tag{30}
            \end{eqnarray}

        By :ref:`(30) <pcpo-eq-30>`, and :math:`L`-smooth continuous function :math:`f,` we have

        .. math::

            \begin{aligned}
            f\left(\theta_{k+1}\right) & \leq f\left(\theta_k\right)+\boldsymbol{g}^T\left(\theta_{k+1}-\theta_k\right)+\frac{L}{2}\left\|\theta_{k+1}-\theta_k\right\|_2^2 \\
            & \leq f\left(\theta_k\right)-\frac{1}{\eta}\left\|\theta_{k+1}-\theta_k\right\|_{\boldsymbol{H}}^2+\frac{L}{2}\left\|\theta_{k+1}-\theta_k\right\|_2^2 \\
            &=f\left(\theta_k\right)+\left(\theta_{k+1}-\theta_k\right)^T\left(-\frac{1}{\eta} \boldsymbol{H}+\frac{L}{2} \boldsymbol{I}\right)\left(\theta_{k+1}-\theta_k\right) \\
            &=f\left(\theta_k\right)+\left\|\theta_{k+1}-\theta_k\right\|_{-\frac{1}{\eta} \boldsymbol{H}+\frac{L}{2} \boldsymbol{I}}^2
            \end{aligned}

        For the :math:`L2` norm projection, we have

        .. _`pcpo-eq-31`:

        .. math::
            :nowrap:

                \begin{eqnarray}
                    (\theta_{k}-\theta_{k+1})^T(\theta_{k}-\eta\boldsymbol{H}^{-1}g-\theta_{k+1})\leq0\\
                    \Rightarrow g^T\boldsymbol{H}^{-1}(\theta_{k+1}-\theta_{k})\leq -\frac{1}{\eta}||\theta_{k+1}-\theta_{k}||^2_2\tag{31}
                \end{eqnarray}

        By :ref:`(31) <pcpo-eq-31>`, :math:`L`-smooth continuous function :math:`f`, and if :math:`\sigma_\mathrm{max}(\boldsymbol{H})\leq1`, we have

        .. math::

            \begin{aligned}
                f(\theta_{k+1})&\leq f(\theta_{k})+g^T(\theta_{k+1}-\theta_{k})+\frac{L}{2}||\theta_{k+1}-\theta_{k}||^2_2 \nonumber\\
                &\leq f(\theta_{k})+(\frac{L}{2}-\frac{1}{\eta})||\theta_{k+1}-\theta_{k}||^2_2.\nonumber
            \end{aligned}

        To see why we need the assumption of :math:`\sigma_\mathrm{max}(\boldsymbol{H})\leq1`, we define :math:`\boldsymbol{H}=\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{U}^T` as the singular value decomposition of :math:`\boldsymbol{H}` with :math:`u_i` being the column vector of :math:`\boldsymbol{U}`.
        Then we have

        .. math::

            \begin{aligned}
                g^T\boldsymbol{H}^{-1}(\theta_{k+1}-\theta_{k})
                &=g^T\boldsymbol{U}\boldsymbol{\Sigma}^{-1}\boldsymbol{U}^T(\theta_{k+1}-\theta_{k}) \nonumber\\
                &=g^T(\sum_{i}\frac{1}{\sigma_i(\boldsymbol{H})}u_iu_i^T)(\theta_{k+1}-\theta_{k})\nonumber\\
                &=\sum_{i}\frac{1}{\sigma_i(\boldsymbol{H})}g^T(\theta_{k+1}-\theta_{k}).\nonumber
            \end{aligned}

        If we want to have

        .. math:: g^T(\theta_{k+1}-\theta_{k})\leq g^T\boldsymbol{H}^{-1}(\theta_{k+1}-\theta_{k})\leq -\frac{1}{\eta}||\theta_{k+1}-\theta_{k}||^2_2,

        then every singular value :math:`\sigma_i(\boldsymbol{H})` of :math:`\boldsymbol{H}` needs to be smaller than :math:`1`, and hence :math:`\sigma_\mathrm{max}(\boldsymbol{H})\leq1`, which justifies the assumption we use to prove the bound.

        .. note::

            To make the objective value for PCPO with the KL divergence projection improves, the right hand side of :ref:`(25) <pcpo-eq-25>` needs to be negative.
            Hence we have :math:`\frac{L\eta}{2}\boldsymbol{I}\prec\boldsymbol{H}`, implying that :math:`\sigma_\mathrm{min}(\boldsymbol{H})>\frac{L\eta}{2}`.
            And to make the objective value for PCPO with the :math:`L2` norm projection improves, the right hand side of :ref:`(26) <pcpo-eq-26>` needs to be negative.
            Hence we have :math:`\eta<\frac{2}{L}`, implying that

            .. math::

                \begin{eqnarray}
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
                    \tag{32}
                \end{eqnarray}

            By the definition of the condition number and :ref:`(29) <pcpo-eq-29>`, we have
