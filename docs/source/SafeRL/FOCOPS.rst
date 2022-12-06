First Order Constrained Optimization in Policy Space
====================================================

Quick Facts
-----------

.. card::
    :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3
    :class-body: sd-font-weight-bold

    #. FOCOPS is an :bdg-success-line:`on-policy` algorithm.
    #. FOCOPS can be used for environments with both :bdg-success-line:`discrete` and :bdg-success-line:`continuous` action spaces.
    #. FOCOPS is an algorithm using :bdg-success-line:`first-order method`.
    #. The OmniSafe implementation of FOCOPS support :bdg-success-line:`parallelization`.

------

.. contents:: Table of Contents
    :depth: 3

FOCOPS Theorem
--------------

Background
~~~~~~~~~~

**First Order Constrained Optimization (FOCOPS)** in Policy Space is a new CPO-based method which maximizes an agent's overall reward while ensuring the agent satisfies a set of cost constraints. FOCOPS purposes that CPO has disadvantages below:

.. grid:: 2

    .. grid-item::
        :columns: 12 6 6 6

        .. card::
            :class-header: sd-bg-warning sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3 sd-font-weight-bold

            Problems of CPO
            ^^^
            -  Sampling error resulting from taking sample trajectories from the current Policy.

            -  Approximation errors resulting from Taylor approximations.

            -  Approximation errors result from using the conjugate method to calculate the inverse of the Fisher information matrix.
            
    .. grid-item::
        :columns: 12 6 6 6

        .. card::
            :class-header: sd-bg-primary sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3 sd-font-weight-bold

            Advantage of FOCOPS
            ^^^
            -  Extremely simple to implement since it only utilizes first order approximations.

            -  Simple first-order method avoids the last two sources of error caused by Taylor method and the conjugate method.

            -  Outperform than CPO in experiment.

            -  No recovery steps required


FOCOPS mainly includes the following contributions:

It provides a **two-stage policy update** to optimize the current Policy.
Next, it provides the practical implementation for solving the two-stage policy update.
Finally, FOCOPS provides rigorous derivative proofs for the above theories, as detailed in the :bdg-ref-info:`Appendix<focops-appendix>` to this tutorial.
One suggested reading order is CPO( :doc:`../SafeRL/CPO`), PCPO( :doc:`../SafeRL/PCPO`), then FOCOPS.
If you have not read the PCPO, it does not matter.
It will not affect your reading experience much.
Nevertheless, be sure to read this article after reading the CPO tutorial we have written so that you can fully understand the following passage.

------

Optimization Objective
~~~~~~~~~~~~~~~~~~~~~~

In the previous chapters, you learned that CPO solves the following optimization problems:

.. _`focops-eq-1`:

.. math::
    :nowrap:

    \begin{eqnarray}
        &&\pi_{k+1}=\arg \max _{\pi \in \Pi_{\boldsymbol{\theta}}} \mathbb{E}_{\substack{s \sim d_{\pi_k}\\a \sim \pi}}[A^R_{\pi_k}(s, a)]\tag{1}\\
        \text{s.t.} \quad &&J^{C_i}\left(\pi_k\right) \leq d_i-\frac{1}{1-\gamma} \mathbb{E}_{\substack{s \sim d_{\pi_k} \\ a \sim \pi}}\left[A^{C_i}_{\pi_k}(s, a)\right] \quad \forall i \tag{2} \\
        &&\bar{D}_{K L}\left(\pi \| \pi_k\right) \leq \delta\tag{3}
    \end{eqnarray}


where :math:`\prod_{\theta}\subseteq\prod` denotes the set of parametrized policies with parameters :math:`\theta`, and :math:`\bar{D}_{K L}` is the KL divergence of two policy.
In local policy search for CMDPs, we additionally require policy iterates to be feasible for the CMDP, so instead of optimizing over :math:`\prod_{\theta}`, PCPO optimizes over :math:`\prod_{\theta}\cap\prod_{C}`.
Next, we will introduce you to how FOCOPS solves the above optimization problems.
For you to have a clearer understanding, we hope that you will read the next section with the following questions:

.. card::
    :class-header: sd-bg-primary sd-text-white sd-font-weight-bold
    :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3 sd-font-weight-bold

    Questions
    ^^^
    -  What is a two-stage policy update, and how?

    -  How to practically implement FOCOPS?

    -  How do parameters impact the performance of the algorithm?

------

Two-stage Policy Update
~~~~~~~~~~~~~~~~~~~~~~~

Instead of solving the Problem :ref:`(1) <focops-eq-1>` ~ :ref:`(3) <focops-eq-1>` directly, FOCOPS uses a **two-stage** approach summarized below:

.. card::
    :class-header: sd-bg-primary sd-text-white sd-font-weight-bold
    :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3 sd-font-weight-bold

    Two-stage Policy Update
    ^^^
    -  Given policy :math:`\pi_{\theta_k}`, find an optimal update policy :math:`\pi^*` by solving the optimization problem from Problem :ref:`(1) <focops-eq-1>` in the non-parameterized policy space.

    -  Project the Policy found in the previous step back into the parameterized policy space :math:`\Pi_{\theta}` by solving for the closest policy :math:`\pi_{\theta}\in\Pi_{\theta}` to :math:`\pi^*`, in order to obtain :math:`\pi_{\theta_{k+1}}`.

------

Finding the Optimal Update Policy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the first stage, FOCOPS rewrites Problem :ref:`(1) <focops-eq-1>` ~ :ref:`(3) <focops-eq-1>` as below:

.. _`focops-eq-4`:

.. math::
    :nowrap:

    \begin{eqnarray}
        &&\pi^*=\arg \max _{\pi \in \Pi} \mathbb{E}_{\substack{s \sim d_{\pi_k}\\a \sim \pi}}[A^R_{\pi_k}(s, a)]\tag{4}\\
        \text{s.t.} \quad && J^{C}\left(\pi_k\right) \leq d-\frac{1}{1-\gamma} \mathbb{E}{\substack{s \sim d_{\pi_k} \\ a \sim \pi}}\left[A^{C}_{\pi_k}(s, a)\right] \quad \tag{5} \\
        && \bar{D}_{K L}\left(\pi \| \pi_k\right) \leq \delta\tag{6}
    \end{eqnarray}

These problems are only slightly different from Problem :ref:`(1) <focops-eq-1>` ~ :ref:`(3) <focops-eq-1>`, that is, the parameter of interest is now the nonparameterized Policy :math:`\pi` and not the policy parameter :math:`\theta`.
Then FOCOPS provides a solution as follows:

.. _focops-theorem-1:

.. card::
    :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
    :class-card: sd-outline-info sd-border-{3} sd-shadow-sm sd-rounded-3
    :class-footer: sd-font-weight-bold
    :link: focops-appendix
    :link-type: ref

    Theorem 1
    ^^^
    Let :math:`\tilde{b}=(1-\gamma)\left(b-\tilde{J}^C\left(\pi_{\theta_k}\right)\right)`.
    If :math:`\pi_{\theta_k}` is a feasible solution, the optimal policy for Problem :ref:`(4) <focops-eq-4>` ~ :ref:`(6) <focops-eq-4>` takes the form

    .. _`focops-eq-7`:

    .. math:: \pi^*(a \mid s)=\frac{\pi_{\theta_k}(a \mid s)}{Z_{\lambda, \nu}(s)} \exp \left(\frac{1}{\lambda}\left(A_{\pi_{\theta_k}}(s, a)-\nu A^C_{\pi_{\theta_k}}(s, a)\right)\right)\tag{7}

    where :math:`Z_{\lambda,\nu}(s)` is the partition function which ensures Problem :ref:`(7) <focops-eq-7>` is a valid probability distribution, :math:`\lambda` and :math:`\nu` are solutions to the optimization problem:

    .. _`focops-eq-8`:

    .. math::
        :nowrap:

        \begin{eqnarray}
            \min _{\lambda, \nu \geq 0} \lambda \delta+\nu \tilde{b}+\lambda \underset{\substack{s \sim d^{\pi_{\theta_k}} \\ a \sim \pi^*}}{\mathbb{E}}[\log Z_{\lambda, \nu}(s)]\tag{8}
        \end{eqnarray}
    +++
    The proof of the :bdg-info-line:`Theorem 1` can be seen in the :bdg-info:`Appendix`, click on this :bdg-info-line:`card` to jump to view.

The form of the optimal Policy is intuitive.
It gives high probability mass to areas of the state-action space with high return, offset by a penalty term times the cost advantage.
We will refer to the optimal solution to Problem :ref:`(4) <focops-eq-4>` ~ :ref:`(6) <focops-eq-4>` as the *optimal update policy*.
Suppose you need help understanding the meaning of the above Equation.
In that case, you can first think that FOCOPS finally solves Problem :ref:`(4) <focops-eq-4>` ~ :ref:`(6) <focops-eq-4>` by solving Problem :ref:`(7) <focops-eq-7>`and Problem :ref:`(8) <focops-eq-8>`.
That is, the :bdg-info-line:`Theorem 1` is a viable solution.

.. grid:: 2

    .. grid-item::
        :columns: 12 6 6 4

        .. tab-set::

            .. tab-item:: Question I
                :sync: key1

                .. card::
                    :class-header: sd-bg-success  sd-text-white sd-font-weight-bold
                    :class-card:  sd-outline-info sd-border-{3} sd-shadow-sm sd-rounded-3 sd-font-weight-bold

                    Question
                    ^^^
                    What is the bound for FOCOPS worst-case guarantee for cost constraint?

            .. tab-item:: Question II
                :sync: key2

                .. card::
                    :class-header: sd-bg-success  sd-text-white sd-font-weight-bold
                    :class-card:  sd-outline-info sd-border-{3} sd-shadow-sm sd-rounded-3 sd-font-weight-bold

                    Question
                    ^^^
                    Can FOCOPS solve the multi-constraint problem and how?

    .. grid-item::
      :columns: 12 6 6 8

      .. tab-set::

            .. tab-item:: Answer I
                :sync: key1

                .. card::
                    :class-header: sd-bg-primary  sd-text-white sd-font-weight-bold
                    :class-card:  sd-outline-info sd-border-{3} sd-shadow-sm sd-rounded-3 sd-font-weight-bold

                    Answer
                    ^^^
                    FOCOPS purpose that the optimal update policy :math:`\pi^*` satisfies the following bound for the worst-case guarantee for cost constraint in CPO:

                    .. math:: J^C\left(\pi^*\right) \leq d+\frac{\sqrt{2 \delta} \gamma \epsilon_C^{\pi^*}}{(1-\gamma)^2}

                    where :math:`\epsilon^C_{\pi^*}=\max _s\left|\mathbb{E}_{a \sim \pi}\left[A^C_{\pi_{\theta_k}}(s, a)\right]\right|`.


            .. tab-item:: Answer II
                :sync: key2

                .. card::
                    :class-header: sd-bg-primary  sd-text-white sd-font-weight-bold
                    :class-card:  sd-outline-info sd-border-{3} sd-shadow-sm sd-rounded-3 sd-font-weight-bold

                    Answer
                    ^^^
                    By introducing Lagrange multipliers :math:`\nu_1,\nu_2,...,\nu_m\ge0`, one for each cost constraint and applying a similar duality argument, FOCOPS extends its results to accommodate for multiple constraints.

------

Approximating the Optimal Update Policy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The optimal update policy :math:`\pi^*` is obtained in the previous section.
However, it is not a parameterized policy.
In this section, we will show you how FOCOPS projects the optimal update policy back into the parameterized policy space by minimizing the loss function:

.. math:: \mathcal{L}(\theta)=\underset{s \sim d^{\pi_{\theta_k}}}{\mathbb{E}}\left[D_{\mathrm{KL}}\left(\pi_\theta \| \pi^*\right)[s]\right]\tag{9}

Here :math:`\pi_{\theta}\in \Pi_{\theta}` is some projected policy that FOCOPS will use to approximate the optimal update policy.
The first-order methods are also used to minimize this loss function:

.. card::
    :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
    :class-card: sd-outline-info sd-border-{3} sd-shadow-sm sd-rounded-3
    :class-footer: sd-font-weight-bold
    :link: focops-appendix
    :link-type: ref

    Corollary 1
    ^^^
    The gradient of :math:`\mathcal{L}(\theta)` takes the form

    .. _`focops-eq-10`:

    .. math:: \nabla_\theta \mathcal{L}(\theta)=\underset{s \sim d^{\pi_\theta}}{\mathbb{E}}\left[\nabla_\theta D_{K L}\left(\pi_\theta \| \pi^*\right)[s]\right]\tag{10}

    where

    .. math::
        :nowrap:

        \begin{eqnarray}
        \nabla_\theta D_{K L}\left(\pi_\theta \| \pi^*\right)[s]=\nabla_\theta D_{K L}\left(\pi_\theta \| \pi_{\theta_k}\right)[s]
        -\frac{1}{\lambda} \underset{a \sim \pi_{\theta_k}}{\mathbb{E}}\left[\frac{\nabla_\theta \pi_\theta(a \mid s)}{\pi_{\theta_k}(a \mid s)}\left(A_{\pi_{\theta_k}}(s, a)-\nu A^C_{\pi_{\theta_k}}(s, a)\right)\right]\tag{11}
        \end{eqnarray}
    +++
    The proof of the :bdg-info-line:`Corollary 1` can be seen in the :bdg-info:`Appendix`, click on this :bdg-info-line:`card` to jump to view.

Note that Equation :ref:`(10) <focops-eq-10>` can be estimated by sampling from the trajectories generated by Policy :math:`\pi_{\theta_k}` so Policy can be trained using stochastic gradients.

:bdg-info-line:`Corollary 1` outlines the FOCOPS algorithm:

At every iteration, we begin with a policy :math:`\pi_{\theta_k}`, which we use to run trajectories and gather data.
We use that data and Equation :ref:`(8) <focops-eq-8>` first to estimate :math:`\lambda` and :math:`\nu`.
We then draw a minibatch from the data to estimate :math:`\nabla_\theta \mathcal{L}(\theta)` given in :bdg-info-line:`Corollary 1`.
After taking a gradient step using Equation:ref:`(10) <focops-eq-10>`, we draw another minibatch and repeat the process.

------

Practical Implementation
------------------------

.. hint::

    Solving Problem :ref:`(8) <focops-eq-8>` is computationally impractical for large state or action spaces as it requires calculating the partition function :math:`Z_{\lambda,\nu}(s)`, which often involves evaluating a high-dimensional integral or sum.
    Furthermore, :math:`\lambda` and :math:`\nu` depend on :math:`k` and should be adapted at every iteration.

So in this section, we will introduce you to how FOCOPS practically implements its algorithm purpose.
In practice, through hyperparameter sweeps, FOCOPS found that a fixed :math:`\lambda` provides good results, which means the value of :math:`\lambda` does not have to be updated.
However, :math:`\nu` needs to be continuously adapted during training so as to ensure cost-constraint satisfaction.
FOCOPS appeals to an intuitive heuristic for determining :math:`\nu` based on primal-dual gradient methods.
With strong duality, the optimal :math:`\lambda^*` and :math:`\nu^*` minimizes the dual function :ref:`(8) <focops-eq-8>` which then be denoted as :math:`L(\pi^*,\lambda,\nu)`.
By applying gradient descent w.r.t :math:`\nu` to minimize :math:`L(\pi^*,\lambda,\nu)`, we obtain:

.. card::
    :class-header: sd-bg-success  sd-text-white sd-font-weight-bold
    :class-card: sd-outline-info sd-border-{3} sd-shadow-sm sd-rounded-3
    :class-footer: sd-font-weight-bold
    :link: focops-appendix
    :link-type: ref

    Corollary 2
    ^^^
    The derivative of :math:`L(\pi^*,\lambda,\nu)` w.r.t :math:`\nu` is

    .. _`focops-eq-12`:
    
    .. math::
        :nowrap:

        \begin{eqnarray}
        \frac{\partial L\left(\pi^*, \lambda, \nu\right)}{\partial \nu}=\tilde{b}-\underset{\substack{s \sim d^{\pi^*} \\ a \sim \pi^*}}{\mathbb{E}}\left[A_{\pi_{\theta_k}}(s, a)\right]\tag{12}
        \end{eqnarray}
    +++
    The proof of the :bdg-success-line:`Corollary 2` can be seen in the :bdg-success:`Appendix`, click on this :bdg-success-line:`card` to jump to view.

The last term in the gradient expression in Equation :ref:`(12) <focops-eq-12>` cannot be evaluated since we do not have access to :math:`\pi^*`.
Since :math:`\pi_{\theta_k}` and :math:`\pi^*` are 'close', it is reasonable to assume that :math:`E_{s \sim d^{\pi_k}, a \sim \pi^*}\left[A_{\pi_{\theta_k}}(s, a)\right] \approx E_{s \sim d^{\pi_k}, a \sim \pi_{\theta_k}}\left[A_{\pi_{\theta_k}}(s, a)\right]=0`.
In practice, this term can be set to zero, which gives the updated term:

.. _`focops-eq-13`:

.. math::
    :nowrap:

    \begin{eqnarray}
    \nu \leftarrow \underset{\nu}{\operatorname{proj}}\left[\nu-\alpha\left(d-J^C\left(\pi_{\theta_k}\right)\right)\right]\tag{13}
    \end{eqnarray}

where :math:`\alpha` is the step size.
Note that we have incorporated the discount term :math:`(1-\gamma)` into :math:`\tilde{b}` into the step size.
The projection operator :math:`proj_{\nu}` projects :math:`\nu` back into the interval :math:`[0,\nu_{max}]`, where :math:`\nu_{max}` is chosen so that :math:`\nu` does not become too large.
In fact. FOCOPS purposed that even setting :math:`\nu_{max}=+\infty` does not appear to reduce performance greatly.
Practically, :math:`J^C(\pi_{\theta_k})` can be estimated via Monte Carlo methods using trajectories collected from :math:`\pi_{\theta_k}`.
Using the update rule :ref:`(13) <focops-eq-13>`, FOCOPS performs one update step on :math:`\nu` before updating the Policy parameters :math:`\theta`.
A per-state acceptance indicator function :math:`I\left(s_j\right)^n:=\mathbf{1}_{D_{\mathrm{KL}}\left(\pi_\theta \| \pi_{\theta_k}\right)\left[s_j\right] \leq \delta}` is added to :ref:`(10) <focops-eq-10>`, in order better to enforce the accuracy for the first-order purposed method.

.. hint::

    Here :math:`N` is the number of samples collected by Policy :math:`\pi_{\theta_k}`, :math:`\hat A` and :math:`\hat A^C` are estimates of the advantage functions (for the return and cost) obtained from critic networks.
    The advantage functions are obtained using the Generalized Advantage Estimator (GAE).
    Note that FOCOPS only requires first-order methods (gradient descent) and is thus extremely simple to implement.

------

Variables Analysis
~~~~~~~~~~~~~~~~~~

In this section, we will explain the meaning of parameters :math:`\lambda` and :math:`\mu` of FOCOPS and their impact on the algorithm's performance in the experiment.

.. tab-set::

    .. tab-item:: Analysis of :math:`\lambda`

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info sd-border-{3} sd-shadow-sm sd-rounded-3
            :class-footer: sd-font-weight-bold

            Analysis of :math:`\lambda`
            ^^^
            In Equation :ref:`(7) <focops-eq-7>`, note that as :math:`\lambda \rightarrow 0`, :math:`\pi^*` approaches a greedy policy;
            as :math:`\lambda` increases, the Policy becomes more exploratory.
            Therefore :math:`\lambda` is similar to the temperature term used in maximum entropy reinforcement learning,
            which has been shown to produce good results when fixed during training.
            In practice, FOCOPS finds that its algorithm reaches the best performance when the :math:`\lambda` is fixed.

    .. tab-item:: Analysis of :math:`\nu`

        .. card::
            :class-header: sd-bg-success  sd-text-white sd-font-weight-bold
            :class-card:  sd-outline-info sd-border-{3} sd-shadow-sm sd-rounded-3
            :class-footer: sd-font-weight-bold

            Analysis of :math:`\nu`
            ^^^
            We recall that in Equation :ref:`(7) <focops-eq-7>`,
            :math:`\nu` acts as a cost penalty term where increasing :math:`\nu` makes it less likely for state-action pairs with higher costs to be sampled by :math:`\pi^*`.
            Hence in this regard, the update rule in :ref:`(13) <focops-eq-13>` is intuitive,
            because it increases :math:`\nu` if :math:`J^C(\pi_{\theta_k})>d`
            (which means the agent violate the cost constraints) and decreases :math:`\nu` otherwise.

------

.. _focops_code_with_omniSafe:

Code with OmniSafe
~~~~~~~~~~~~~~~~~~

Quick start
"""""""""""

.. card::
    :class-header: sd-bg-success sd-text-white sd-font-weight-bold
    :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3 sd-font-weight-bold
    :class-footer: sd-font-weight-bold

    Run FOCOPS in Omnisafe
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

                env = omnisafe.Env('SafetyPointGoal1-v0')

                agent = omnisafe.Agent('FOCOPS', env)
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
                agent = omnisafe.Agent('FOCOPS', env, custom_cfgs=custom_dict)
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

                We use ``train_on_policy.py`` as the entrance file. You can train the agent with FOCOPS simply using ``train_on_policy.py``, with arguments about FOCOPS and enviroments does the training.
                For example, to run FOCOPS in SafetyPointGoal1-v0, with 4 cpu cores and seed 0, you can use the following command:

                .. code-block:: guess
                    :linenos:

                    cd omnisafe/examples
                    python train_on_policy.py --env-id SafetyPointGoal1-v0 --algo FOCOPS --parallel 5 --epochs 1


------

Architecture of functions
"""""""""""""""""""""""""

-  ``focops.learn()``

  -  ``env.roll_out()``
  -  ``focops.update()``

    -  ``focops.buf.get()``
    -  ``focops.pre_process_data(raw_data)``
    -  ``focops.update_policy_net()``
    -  ``focops.update_cost_net()``
    -  ``focops.update_value_net()``

-  ``focops.log()``

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

        focops.update()
        ^^^
        Update actor, critic, running statistics

    .. card::
        :class-header: sd-bg-success sd-text-white sd-font-weight-bold
        :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3 sd-font-weight-bold
        :class-footer: sd-font-weight-bold

        focops.buf.get()
        ^^^
        Call this at the end of an epoch to get all of the data from the buffer

    .. card::
        :class-header: sd-bg-success sd-text-white sd-font-weight-bold
        :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3 sd-font-weight-bold
        :class-footer: sd-font-weight-bold

        focops.update_policy_net()
        ^^^
        Update policy network in 5 kinds of optimization case

    .. card::
        :class-header: sd-bg-success sd-text-white sd-font-weight-bold
        :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3 sd-font-weight-bold
        :class-footer: sd-font-weight-bold

        focops.update_value_net()
        ^^^
        Update Critic network for estimating reward.

    .. card::
        :class-header: sd-bg-success sd-text-white sd-font-weight-bold
        :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3 sd-font-weight-bold
        :class-footer: sd-font-weight-bold

        focops.update_cost_net()
        ^^^
        Update Critic network for estimating cost.

    .. card::
        :class-header: sd-bg-success sd-text-white sd-font-weight-bold
        :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3 sd-font-weight-bold
        :class-footer: sd-font-weight-bold

        focops.log()
        ^^^
        Get the trainning log and show the performance of the algorithm

------

Documentation of new functions
""""""""""""""""""""""""""""""

.. tab-set::

    .. tab-item:: focops.compute_loss_pi(data: dict)

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            focops.compute_loss_pi(data: dict)
            ^^^
            Compute the loss of policy network, flowing the next steps:

            (1) Calculate the KL divergence between the new policy and the old policy

            .. code-block:: python
                :linenos:

                dist, _log_p = self.ac.pi(data['obs'], data['act'])
                ratio = torch.exp(_log_p - data['log_p'])
                kl_new_old = torch.distributions.kl.kl_divergence(dist, self.p_dist).sum(-1, keepdim=True)


            (2) Compute the loss of policy network based on FOCOPS method, where ``self.lagrangian_multiplier`` is :math:`\nu``
                and ``self.lam`` is :math:`\lambda` in FOCOPS paper.

            .. code-block:: python
                :linenos:

                loss_pi = (
                    kl_new_old
                    - (1 / self.lam) * ratio * (data['adv'] - self.lagrangian_multiplier * data['cost_adv'])
                ) * (kl_new_old.detach() <= self.eta).type(torch.float32)
                loss_pi = loss_pi.mean()
                loss_pi -= self.entropy_coef * dist.entropy().mean()

    .. tab-item:: focops.update_lagrange_multiplier(ep_costs: float)

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            focops.update_lagrange_multiplier(ep_costs: float)
            ^^^
            FOCOPS algorithm update ``self.lagrangian_multiplier`` which is :math:`\nu` in FOCOPS paper by projection.

            .. code-block:: python
                :linenos:

                self.lagrangian_multiplier += self.lambda_lr * (ep_costs - self.cost_limit)
                if self.lagrangian_multiplier < 0.0:
                    self.lagrangian_multiplier = 0.0
                elif self.lagrangian_multiplier > 2.0:
                    self.lagrangian_multiplier = 2.0

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
            -  algo (string): The name of algorithm corresponding to current
                class, it does not actually affect any things which happen in the
                following.
            -  actor (string): The type of network in actor, discrete of
                continuous.
            -  model_cfgs (dictionary) : successrmation about actor and critic's net
                work configuration,it originates from ``algo.yaml`` file to describe
                ``hidden layers`` , ``activation function``, ``shared_weights`` and ``weight_initialization_mode``.

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
            -  adv_estimation_method (float):Roughly what KL divergence we think is
                appropriate between new and old policies after an update. This will
                get used for early stopping. (Usually small, 0.01 or 0.05.)
            -  standardized_reward (int):  Use standarized reward or not.
            -  standardized_cost (bool): Use standarized cost or not.

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
   :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3

   Lemma 1
   ^^^
   Problem
   :ref:`(4) <focops-eq-4>` ~ :ref:`(6) <focops-eq-4>`
   is convex w.r.t
   :math:`\pi={\pi(a|s):s\in \mathrm{S},a\in\mathrm{A}}`.

.. card::
    :class-header: sd-bg-info sd-text-white sd-font-weight-bold
    :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3

    Proof of Lemma 1
    ^^^
    First, note that the objective function is linear w.r.t :math:`\pi`.
    Since :math:`J^{C}(\pi_{\theta_k})` is a constant w.r.t :math:`\pi`, constraint :ref:`(5) <focops-eq-4>` is linear.
    Constraint :ref:`(6) <focops-eq-4>` can be rewritten as :math:`\sum_s d^{\pi_{\theta_k}}(s) D_{\mathrm{KL}}\left(\pi \| \pi_{\theta_k}\right)[s] \leq \delta`.
    The KL divergence is convex w.r.t its first argument.
    Hence Constraint :ref:`(5) <focops-eq-4>`, a linear combination of convex functions, is also convex.
    Since :math:`\pi_{\theta_k}` satisfies Constraint :ref:`(6) <focops-eq-4>` also satisfies Constraint :ref:`(5) <focops-eq-4>`, therefore Slater's constraint qualification holds, and strong duality holds.

.. dropdown:: Proof of Theorem 1 (Click here)
    :color: info
    :class-body: sd-border-{3}

    Based on :bdg-info-line:`Lemma 1` the optimal value of the Problem :ref:`(4) <focops-eq-4>` ~ :ref:`(6) <focops-eq-4>` :math:`p^*` can be solved by solving the corresponding dual problem.
    Let

    .. math:: L(\pi, \lambda, \nu)=\lambda \delta+\nu \tilde{b}+\underset{s \sim d^{\pi_{\theta_k}}}{\mathbb{E}}\left[A^{lag}-\lambda D_{\mathrm{KL}}\left(\pi \| \pi_{\theta_k}\right)[s]\right]\nonumber

    where :math:`A^{lag}=\underset{a \sim \pi(\cdot \mid s)}{\mathbb{E}}\left[A_{\pi_{\theta_k}}(s, a)-\nu A^C_{\pi_{\theta_k}}(s, a)\right]`.
    Therefore.

    .. _`focops-eq-15`:

    .. math:: p^*=\max _{\pi \in \Pi} \min _{\lambda, \nu \geq 0} L(\pi, \lambda, \nu)=\min _{\lambda, \nu \geq 0} \max _{\pi \in \Pi} L(\pi, \lambda, \nu)\tag{15}

    Note that if :math:`\pi^*`, :math:`\lambda^*`, :math:`\nu^*` are optimal for Problem\ :ref:`(15) <focops-eq-15>`, :math:`\pi^*` is also optimal for Problem :ref:`(4) <focops-eq-4>` ~ :ref:`(6) <focops-eq-4>` because of the strong duality.

    Consider the inner maximization problem in Problem :ref:`(15) <focops-eq-15>`.
    We separate it from the original problem and try to solve it first:

    .. _`focops-eq-16`:

    .. math::
        :nowrap:

        \begin{eqnarray}
        &&\underset{\pi}{\operatorname{max}}  A^{lag}-\underset{a \sim \pi(\cdot \mid s)}{\mathbb{E}}\left[\lambda\left(\log \pi(a \mid s)+\log \pi_{\theta_k}(a \mid s)\right)\right]\tag{16} \\
        \text { s.t. } && \sum_a \pi(a \mid s)=1 \\
        && \pi(a \mid s) \geq 0 \quad \forall a \in \mathcal{A}
        \end{eqnarray}

    Which is equivalent to the inner maximization problem in :ref:`(15) <focops-eq-15>`.
    We can solve this convex optimization problem using a simple Lagrangian argument.
    We can write the Lagrangian of it as:

    .. math::
        :nowrap:

        \begin{eqnarray}
        G(\pi)=\sum_a \pi(a \mid s)[A_{\pi_{\theta_k}}(s, a)-\nu A^C_{\pi_{\theta_k}}(s, a)
        -\lambda(\log \pi(a \mid s)-\log \pi_{\theta_k}(a \mid s))+\zeta]-1\tag{17}
        \end{eqnarray}

    where :math:`\zeta > 0` is the Lagrange multiplier associated with the constraint :math:`\sum_a \pi(a \mid s)=1`.
    Different :math:`G(\pi)` w.r.t. :math:`\pi(a \mid s)` for some :math:`a`:

    .. _`focops-eq-18`:

    .. math::
        :nowrap:

        \begin{eqnarray}
        \frac{\partial G}{\partial \pi(a \mid s)}=A_{\pi_{\theta_k}}(s, a)-\nu A^C_{\pi_{\theta_k}}(s, a)-\lambda\left(\log \pi(a \mid s)+1-\log \pi_{\theta_k}(a \mid s)\right)+\zeta\tag{18}
        \end{eqnarray}

    Setting Equation :ref:`(18) <focops-eq-18>` to zero and rearranging the term, we obtain:

    .. math:: \pi(a \mid s)=\pi_{\theta_k}(a \mid s) \exp \left(\frac{1}{\lambda}\left(A_{\pi_{\theta_k}}(s, a)-\nu A^C_{\pi_{\theta_k}}(s, a)\right)+\frac{\zeta}{\lambda}+1\right)\tag{19}

    We chose :math:`\zeta` so that :math:`\sum_a \pi(a \mid s)=1` and rewrite :math:`\zeta / \lambda+1` as :math:`Z_{\lambda, \nu}(s)`.
    We find that the optimal solution :math:`\pi^*` to Equation :ref:`(16) <focops-eq-16>` takes the form

    .. math:: \pi^*(a \mid s)=\frac{\pi_{\theta_k}(a \mid s)}{Z_{\lambda, \nu}(s)} \exp \left(\frac{1}{\lambda}\left(A_{\pi_{\theta_k}}(s, a)-\nu A^C_{\pi_{\theta_k}}(s, a)\right)\right)

    Then we obtain:

    .. math::
        :nowrap:

        \begin{eqnarray}
        &&\underset{\substack{s \sim d^{\theta_{\theta_k}} \\
        a \sim \pi^*}}{\mathbb{E}}\left[A_{\pi_{\theta_k}}(s, a)-\nu A^C_{\pi_{\theta_k}}(s, a)-\lambda\left(\log \pi^*(a \mid s)-\log \pi_{\theta_k}(a \mid s)\right)\right] \\
        = &&\underset{\substack{s \sim d^{\pi_{\theta_k}} \\
        a \sim \pi^*}}{\mathbb{E}}\left[A_{\pi_{\theta_k}}(s, a)-\nu A^C_{\pi_{\theta_k}}(s, a)-\lambda\left(\log \pi_{\theta_k}(a \mid s)-\log Z_{\lambda, \nu}(s)\right.\right. \\
        &&\left.\left. + \frac{1}{\lambda}\left(A_{\pi_{\theta_k}}(s, a)-\nu A^C_{\pi_{\theta_k}}(s, a)\right)-\log \pi_{\theta_k}(a \mid s)\right)\right]\\
        = &&\lambda\underset{\substack{s \sim d^{\theta_{\theta_k}} \\
        a \sim \pi^*}}{\mathbb{E}}[logZ_{\lambda,\nu}(s)]\nonumber
        \end{eqnarray}

    Plugging the result back to Equation :ref:`(15) <focops-eq-15>`, we obtain:

    .. math::

        p^*=\underset{\lambda,\nu\ge0}{min}\lambda\delta+\nu\tilde{b}+\lambda\underset{\substack{s \sim d^{\theta_{\theta_k}} \\
        a \sim \pi^*}}{\mathbb{E}}[logZ_{\lambda,\nu}(s)]

------

.. _focops-proof-corollary:

Proof of Corollary
~~~~~~~~~~~~~~~~~~

.. tab-set::

   .. tab-item:: Proof of Corollary 1

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card:  sd-outline-info sd-border-{3} sd-shadow-sm sd-rounded-3

            Proof of Corollary 1
            ^^^
            We only need to calculate the gradient of the loss function for a single sampled s. We first note that,

            .. math::
                :nowrap:

                \begin{eqnarray}
                &&D_{\mathrm{KL}}\left(\pi_\theta \| \pi^*\right)[s]\\
                =&&-\sum_a \pi_\theta(a \mid s) \log \pi^*(a \mid s)+\sum_a \pi_\theta(a \mid s) \log \pi_\theta(a \mid s)\tag{20} \\
                =&&H\left(\pi_\theta, \pi^*\right)[s]-H\left(\pi_\theta\right)[s]
                \end{eqnarray}

            where :math:`H\left(\pi_\theta\right)[s]` is the entropy and :math:`H\left(\pi_\theta, \pi^*\right)[s]` is the cross-entropy under state :math: 's`.
            The above Equation is the basic mathematical knowledge in information theory, which you can get in any information theory textbook.
            We expand the cross entropy term, which gives us the following:

            .. math::
                :nowrap:

                \begin{eqnarray}
                H\left(\pi_\theta, \pi^*\right)[s] &=&-\sum_a \pi_\theta(a \mid s) \log \pi^*(a \mid s) \\
                &=&-\sum_a \pi_\theta(a \mid s) \log \left(\frac{\pi_{\theta_k}(a \mid s)}{Z_{\lambda, \nu}(s)} \exp \left[\frac{1}{\lambda}\left(A_{\pi_{\theta_k}}(s, a)-\nu A^C_{\pi_{\theta_k}}(s, a)\right)\right]\right) \\
                &=&-\sum_a \pi_\theta(a \mid s) \log \pi_{\theta_k}(a \mid s)+\log Z_{\lambda, \nu}(s)-\frac{1}{\lambda} \sum_a \pi_\theta(a \mid s)\left(A_{\pi_{\theta_k}}(s, a)-\nu A^C_{\pi_{\theta_k}}(s, a)\right)
                \end{eqnarray}

            We then subtract the entropy term to recover the KL divergence:

            .. math::
                :nowrap:

                \begin{eqnarray}
                &D_{\mathrm{KL}}\left(\pi_\theta \| \pi^*\right)[s]=D_{\mathrm{KL}}\left(\pi_\theta \| \pi_{\theta_k}\right)[s]+\log Z_{\lambda, \nu}(s)-\\&\frac{1}{\lambda} \underset{a \sim \pi_{\theta_k}(\cdot \mid s)}{\mathbb{E}}\left[\frac{\pi_\theta(a \mid s)}{\pi_{\theta_k}(a \mid s)}\left(A_{\pi_{\theta_k}}(s, a)-\nu A^C_{\pi_{\theta_k}}(s, a)\right)\right]\nonumber
                \end{eqnarray}

            In the last equality, we applied importance sampling to rewrite the expectation w.r.t. :math:`\pi_{\theta_k}`.
            Finally, taking the gradient on both sides gives us the following:

            .. math::
                :nowrap:

                \begin{eqnarray}
                &\nabla_\theta D_{\mathrm{KL}}\left(\pi_\theta \| \pi^*\right)[s]=\nabla_\theta D_{\mathrm{KL}}\left(\pi_\theta \| \pi_{\theta_k}\right)[s]\\&-\frac{1}{\lambda} \underset{a \sim \pi_{\theta_k}(\cdot \mid s)}{\mathbb{E}}\left[\frac{\nabla_\theta \pi_\theta(a \mid s)}{\pi_{\theta_k}(a \mid s)}\left(A_{\pi_{\theta_k}}(s, a)-\nu A^C_{\pi_{\theta_k}}(s, a)\right)\right]\nonumber
                \end{eqnarray}

   .. tab-item:: Proof of Corollary 2

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card:  sd-outline-info sd-border-{3} sd-shadow-sm sd-rounded-3

            Proof of Corollary 2
            ^^^
            From :bdg-ref-info-line:`Theorem 1<focops-theorem-1>`, we have:

            .. math::
                :nowrap:

                \begin{eqnarray}
                L\left(\pi^*, \lambda, \nu\right)=\lambda \delta+\nu \tilde{b}+\lambda \underset{\substack{s \sim d^{\pi^*} \\ a \sim \pi^*}}{\mathbb{E}}\left[\log Z_{\lambda, \nu}(s)\right]\tag{21}
                \end{eqnarray}

            The first two terms are an affine function w.r.t. :math:`\nu`.
            Therefore, its derivative is :math:`\tilde{b}`. We will then focus on the expectation in the last term.
            To simplify our derivation, we will first calculate the derivative of :math:`\pi^*` w.r.t. :math:`\nu`,

            .. math::
                :nowrap:

                \begin{eqnarray}
                \frac{\partial \pi^*(a \mid s)}{\partial \nu} &=&\frac{\pi_{\theta_k}(a \mid s)}{Z_{\lambda, \nu}^2(s)}\left[Z_{\lambda, \nu}(s) \frac{\partial}{\partial \nu} \exp \left(\frac{1}{\lambda}\left(A_{\pi_{\theta_k}}(s, a)-\nu A^C_{\pi_{\theta_k}}(s, a)\right)\right)\right.\\
                &&\left.-\exp \left(\frac{1}{\lambda}\left(A_{\pi_{\theta_k}}(s, a)-\nu A^C_{\pi_{\theta_k}}(s, a)\right)\right) \frac{\partial Z_{\lambda, \nu}(s)}{\partial \nu}\right] \\
                &=&-\frac{A^C_{\pi_{\theta_k}}(s, a)}{\lambda} \pi^*(a \mid s)-\pi^*(a \mid s) \frac{\partial \log Z_{\lambda, \nu}(s)}{\partial \nu}\nonumber
                \end{eqnarray}

            Therefore the derivative of the expectation in the last term of :math:`L(\pi^*,\lambda,\nu)` can be written as:
            
            .. _`focops-eq-22`:
            
            .. math::
                :nowrap:

                \begin{eqnarray}\label{FOCOPS:proof_C2_1}
                \frac{\partial}{\partial \nu} \underset{\substack{s \sim d^\pi \theta_k \\
                a \sim \pi^*}}{\mathbb{E}}\left[\log Z_{\lambda, \nu}(s)\right]
                &=& \underset{\substack{s \sim d^{\pi_\theta} \\
                a \sim \pi_{\theta_k}}}{\mathbb{E}}\left[\frac{\partial}{\partial \nu}\left(\frac{\pi^*(a \mid s)}{\pi_{\theta_k}(a \mid s)} \log Z_{\lambda, \nu}(s)\right)\right] \\
                &=& \underset{\substack{s \sim d^{\pi_\theta} \\
                a \sim \pi_{\theta_k}}}{\mathbb{E}}\left[\frac{1}{\pi_{\theta_k}(a \mid s)}\left(\frac{\partial \pi^*(a \mid s)}{\partial \nu} \log Z_{\lambda, \nu}(s)+\pi^*(a \mid s) \frac{\partial \log Z_{\lambda, \nu}(s)}{\partial \nu}\right)\right] \\
                &=& \underset{\substack{s \sim d^{\pi_\theta} \\
                a \sim \pi^*}}{\mathbb{E}}\left[-(\frac{A^C_{\pi_{\theta_k}}(s, a)}{\lambda}+\frac{\partial \log Z_{\lambda, \nu}(s)}{\partial \nu}) \log Z_{\lambda, \nu}(s)+\frac{\partial \log Z_{\lambda, \nu}(s)}{\partial \nu}\right]\tag{22}
                \end{eqnarray}

            Also:

            .. math::
                :nowrap:

                \begin{eqnarray}\label{FOCOPS:proof_C2_2}
                \frac{\partial Z_{\lambda, \nu}(s)}{\partial \nu} &=&\frac{\partial}{\partial \nu} \sum_a \pi_{\theta_k}(a \mid s) \exp \left(\frac{1}{\lambda}\left(A_{\pi_{\theta_k}}(s, a)-\nu A^C_{\pi_{\theta_k}}(s, a)\right)\right) \\
                &=&\sum_a-\pi_{\theta_k}(a \mid s) \frac{A^C_{\pi_{\theta_k}}(s, a)}{\lambda} \exp \left(\frac{1}{\lambda}\left(A_{\pi_{\theta_k}}(s, a)-\nu A^C_{\pi_{\theta_k}}(s, a)\right)\right) \\
                &=&\sum_a-\frac{A^C_{\pi_{\theta_k}}(s, a)}{\lambda} \frac{\pi_{\theta_k}(a \mid s)}{Z_{\lambda, \nu}(s)} \exp \left(\frac{1}{\lambda}\left(A_{\pi_{\theta_k}}(s, a)-\nu A^C_{\pi_{\theta_k}}(s, a)\right)\right) Z_{\lambda, \nu}(s) \\
                &=&-\frac{Z_{\lambda, \nu}(s)}{\lambda} \underset{a \sim \pi^*(\cdot \mid s)}{\mathbb{E}}\left[A^C_{\pi_{\theta_k}}(s, a)\right]\tag{23}
                \end{eqnarray}

            Therefore:

            .. _`focops-eq-24`:

            .. math:: \frac{\partial \log Z_{\lambda, \nu}(s)}{\partial \nu}=\frac{\partial Z_{\lambda, \nu}(s)}{\partial \nu} \frac{1}{Z_{\lambda, \nu}(s)}=-\frac{1}{\lambda} \underset{a \sim \pi^*(\cdot \mid s)}{\mathbb{E}}\left[A^C_{\pi_{\theta_k}}(s, a)\right]\tag{24}

            Plugging :ref:`(24) <focops-eq-24>`  into the last equality in :ref:`(22) <focops-eq-22>`  gives us:
           
            .. _`focops-eq-25`:
           
            .. math::
                :nowrap:

                \begin{eqnarray}
                \frac{\partial}{\partial \nu} \underset{\substack{s \sim d^{\pi_\theta} \\
                a \sim \pi^*}}{\mathbb{E}}\left[\log Z_{\lambda, \nu}(s)\right]
                &=&\underset{\substack{s \sim d^{\pi^*} \\
                a \sim \pi^*}}{\mathbb{E}}\left[-\frac{A^C_{\pi_{\theta_k}}(s, a)}{\lambda} \log Z_{\lambda, \nu}(s)+\frac{A^C_{\pi_{\theta_k}}(s, a)}{\lambda} \log Z_{\lambda, \nu}(s)-\frac{1}{\lambda} A^C_{\pi_{\theta_k}}(s, a)\right] \\
                &=&-\frac{1}{\lambda} \underset{\substack{s \sim d^{\pi_{\theta_k}} \\
                a \sim \pi^*}}{\mathbb{E}}\left[A^C_{\pi_{\theta_k}}(s, a)\right]\tag{25}
                \end{eqnarray}

            Combining :ref:`(25) <focops-eq-25>`  with the derivatives of the affine term give us the final desired result.
