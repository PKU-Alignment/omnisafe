Trust Region Policy Optimization
================================

Quick Facts
-----------

.. card::
    :class-card: sd-outline-info  sd-rounded-1
    :class-body: sd-font-weight-bold

    #. TRPO is an :bdg-info-line:`on-policy` algorithm.
    #. TRPO is an improvement work done based on :bdg-info-line:`NPG` .
    #. TRPO is an important theoretical basis for :bdg-ref-info-line:`CPO <../saferl/cpo>` .
    #. An :bdg-ref-info-line:`API Documentation <trpoapi>`  is available for TRPO.

------

TRPO Theorem
------------

Background
~~~~~~~~~~

**Trust region policy optimization (TRPO)** is an iterative method for
optimizing policies in reinforcement learning that ensures monotonic
improvements. It works by iteratively finding a local approximation of the
objective return and maximizing the approximated function. TRPO guarantees that
the new policy is constrained within a trust region relative to the current
policy, which is achieved by using KL divergence to measure the distance
between the two policies.

TRPO is well-suited for optimizing comprehensive nonlinear policies such as
neural networks. It is based on the **Natural Policy Gradient (NPG)** method,
which uses conjugate gradient to avoid expensive computational
costs. Furthermore, TRPO incorporates a line search mechanism to ensure
that updated policy adhere to the predetermined KL divergence constraint.

.. grid:: 2

    .. grid-item::
        :columns: 12 6 6 5

        .. card::
            :class-header: sd-bg-warning sd-text-white sd-font-weight-bold
            :class-card: sd-outline-warning  sd-rounded-1 sd-font-weight-bold

            Problems of NPG
            ^^^
            -  It is very difficult to calculate the Hessian matrix directly.

            -  Error introduced by Taylor expansion because of the fixed step length.

            -  Low utilization of sampled data.

    .. grid-item::
        :columns: 12 6 6 6

        .. card::
            :class-header: sd-bg-primary sd-text-white sd-font-weight-bold
            :class-card: sd-outline-primary  sd-rounded-1 sd-font-weight-bold

            Advantage of TRPO
            ^^^
            -  Using conjugate gradient algorithm to compute the Fisher-Vector product.

            -  Using line search algorithm to eliminate the error introduced by Taylor expansion.

            -  Using importance sampling to reuse data.

------

Performance difference over policies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In policy optimization, our objective is to ensure that every update leads to a
consistent improvement in the expected return. To accomplish this, we usually
formulate the equation for expected return in a specific format that is both
intuitive and straightforward to manipulate.

.. math::
    :label: trpo-eq-1

    J^R(\pi') = J^R(\pi) + \{J^R(\pi') - J^R(\pi)\}


To achieve monotonic improvements, we only need to consider
:math:`\Delta = J^R(\pi') - J^R(\pi)` to be non-negative.

As shown in **NPG**, the difference in performance between two policies
:math:`\pi'` and :math:`\pi` can be expressed as:

.. _trpo-Theorem 1:

.. card::
    :class-header: sd-bg-info sd-text-white sd-font-weight-bold
    :class-card: sd-outline-info  sd-rounded-1
    :class-footer: sd-font-weight-bold
    :link: appendix-theorem1
    :link-type: ref

    Theorem 1 (Performance Difference Bound)
    ^^^

    .. _`trpo-eq-2`:

    .. math::
        :label: trpo-eq-2

            J^R(\pi') = J^R(\pi) + \mathbb{E}_{\tau \sim \pi'}[\sum_{t=0}^{\infty} \gamma^t A^{R}_{\pi}(s_t,a_t)]

    where this expectation is taken over trajectories :math:`\tau=(s_0, a_0, s_1,\\ a_1, \cdots)`,
    and the notation :math:`\mathbb{E}_{\tau \sim \pi'}[\cdots]` indicates that actions are sampled from :math:`\pi'` to generate :math:`\tau`.
    +++
    The proof of the :bdg-info-line:`Theorem 1` can be seen in the :bdg-ref-info:`Appendix`, click on this :bdg-info-line:`card` to jump to view.

:bdg-info-line:`Theorem 1` is intuitive as the expected discounted reward of
:math:`\pi'` can be viewed as the expected discounted reward of :math:`\pi`,
and an extra advantage of :math:`\pi'` over :math:`\pi`.
The latter term accounts for how much :math:`\pi'`
can improve over :math:`\pi`, which is of our interest.

.. note::

    We can rewrite :bdg-info-line:`Theorem 1` with a sum over states instead of timesteps:

    .. _`trpo-eq-3`:

    .. math::
        :label: trpo-eq-3

        \label{equation: performance in discount visit density}
        J^R(\pi') &=J^R(\pi)+\sum_{t=0}^{\infty} \sum_s P\left(s_t=s \mid \pi'\right) \sum_a \pi' (a \mid s) \gamma^t A^{R}_{\pi}(s, a) \\
        &=J^R(\pi)+\sum_s \sum_{t=0}^{\infty} \gamma^t P\left(s_t=s \mid \pi' \right) \sum_a \pi'(a \mid s) A^{R}_{\pi}(s, a) \\
        &=J^R(\pi)+\sum_s d_{\pi'}(s) \sum_a \pi'(a \mid s) A^{R}_{\pi}(s, a)


This equation implies for any policy :math:`\pi'`, if it has a nonnegative
expected advantage at every state :math:`s`, i.e.,
:math:`\sum_a \pi'(a \mid s) A^{R}_{\pi}(s, a) \geq 0`,
it is guaranteed to increase the policy performance :math:`J^R`,
or leave it constant in the case
that the expected advantage is zero everywhere.
However, in the approximate setting, it will typically be unavoidable,
due to estimation and approximation errors,
that there will be some states :math:`s` in which the expected advantage is
negative, that is,
:math:`\sum_a \pi'(a \mid s) A^{R}_{\pi}(s, a)<0`.

------

Surrogate function for the objective
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:eq:`trpo-eq-3` requires information about future state distribution under
:math:`\pi'`,
which is usually unknown and difficult to estimate.
The complex dependency of :math:`d_{\pi'}(s)` on :math:`\pi'` makes
:eq:`trpo-eq-3` difficult to optimize directly.
Instead, we introduce the following local approximation to :math:`J^R`:

.. _`trpo-eq-4`:

.. math::
    :label: trpo-eq-4

    L_\pi(\pi')=J^R(\pi)+\sum_s d_\pi(s) \sum_a \pi'(a \mid s) A^{R}_{\pi}(s, a)


Here we only replace :math:`d_{\pi'}` with :math:`d_\pi`.
It has been proved that if the two policy :math:`\pi'` and :math:`\pi` are
close enough,
:math:`L_\pi(\pi')` can be considered as equivalent to :math:`J^R(\pi')`.

.. _trpo-Corollary 1:

.. card::
    :class-header: sd-bg-info sd-text-white sd-font-weight-bold
    :class-card: sd-outline-info  sd-rounded-1
    :class-footer: sd-font-weight-bold
    :link: appendix-corollary1
    :link-type: ref

    Corollary 1 (Performance Difference Bound)
    ^^^
    Formally, suppose a parameterized policy :math:`\pi_{\boldsymbol{\theta}}`,
    where :math:`\pi_{\boldsymbol{\theta}}(a \mid s)` is a differentiable function of the parameter vector :math:`{\boldsymbol{\theta}}`,
    then :math:`L_\pi` matches :math:`J^R` to first order (see **NPG**).
    That is, for any parameter value :math:`{\boldsymbol{\theta}}_0`, we have:

    .. math::
        :label: trpo-eq-5

        L_{\pi_{{\boldsymbol{\theta}}_0}}\left(\pi_{{\boldsymbol{\theta}}_0}\right)=J^R\left(\pi_{{\boldsymbol{\theta}}_0}\right)


    .. _`trpo-eq-6`:

    .. math::
        :label: trpo-eq-6

        \nabla_{\boldsymbol{\theta}} L_{\pi_{{\boldsymbol{\theta}}_0}}\left(\pi_{\boldsymbol{\theta}}\right)|_{{\boldsymbol{\theta}}={\boldsymbol{\theta}}_0}=\left.\nabla_{\boldsymbol{\theta}} J^R\left(\pi_{\boldsymbol{\theta}}\right)\right|_{{\boldsymbol{\theta}}={\boldsymbol{\theta}}_0}

    +++
    The proof of the :bdg-info-line:`Corollary 1` can be seen in the :bdg-ref-info:`Appendix`, click on this :bdg-info-line:`card` to jump to view.

:eq:`trpo-eq-6` implies that a sufficiently small step
:math:`\pi_{{\boldsymbol{\theta}}_0} \rightarrow \pi'` improving
:math:`L_{\pi_{{\boldsymbol{\theta}}_{\text {old }}}}` will also improve :math:`J^R`,
but does not provide explicit guidance on determining the appropriate step size
for policy updates.

To address this issue, **NPG** proposed a policy updating scheme called
**conservative policy iteration(CPI)**,
which could provide explicit lower bounds on the improvement of :math:`J^R`.
To define the conservative policy iteration update,
let :math:`\pi_{\mathrm{old}}` denote the current policy,
and let
:math:`\pi^{*}=\arg \underset{\pi^{*}}{\max} L_{\pi_{\text {old }}}\left(\pi^{*}\right)`.
The new policy :math:`\pi_{\text {new }}`
was defined to be the following mixture:

.. math::
    :label: trpo-eq-7

    \pi_{\text {new }}(a \mid s)=(1-\alpha) \pi_{\text {old }}(a \mid s)+\alpha \pi^{*}(a \mid s)


Kakade and Langford derived the following lower bound:

.. _`trpo-eq-8`:

.. math::
    :label: trpo-eq-8

    J^R\left(\pi_{\text {new }}\right)  &\geq L_{\pi_{\text {old }}}\left(\pi_{\text {new }}\right)-\frac{2 \epsilon \gamma}{(1-\gamma)^2} \alpha^2  \\
    \text { where } \epsilon &=\max _s\left|\mathbb{E}_{a \sim \pi^{*}(a \mid s)}\left[A^{R}_{\pi}(s, a)\right]\right|


However, the lower bound in :eq:`trpo-eq-8` only applies to mixture policies,
so it needs to be extended to general policy cases.

------

Monotonic Improvement Guarantee for General Stochastic Policies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Based on the theoretical guarantee :eq:`trpo-eq-16` in mixture policies case,
TRPO extends the lower bound to general policies by replacing :math:`\alpha`
with a distance measure between :math:`\pi` and :math:`\pi'`,
and changing the constant :math:`\epsilon` appropriately.
The chosen distance measurement is the total variation divergence
(TV divergence),
which is defined by
:math:`D_{TV}(p \| q)=\frac{1}{2} \sum_i \left|p_i-q_i\right|`
for discrete probability distributions :math:`p, q`.
Define :math:`D_{\mathrm{TV}}^{\max }(\pi, \pi')` as

.. math::
    :label: trpo-eq-9

    D_{\mathrm{TV}}^{\max}(\pi, \pi')=\max_s D_{\mathrm{TV}}\left(\pi\left(\cdot \mid s\right) \| \pi'\left(\cdot \mid s\right)\right)


And the new bound is derived by introducing the :math:`\alpha`-coupling method.

.. _trpo-Theorem 2:

.. card::
    :class-header: sd-bg-info sd-text-white sd-font-weight-bold
    :class-card: sd-outline-info  sd-rounded-1
    :class-footer: sd-font-weight-bold
    :link: appendix-theorem2
    :link-type: ref

    Theorem 2 (Performance Difference Bound derived by :math:`\alpha`-coupling method)
    ^^^
    Let
    :math:`\alpha=D_{\mathrm{TV}}^{\max }\left(\pi_{\mathrm{old}}, \pi_{\text {new }}\right)`.
    Then the following bound holds:

    .. math::
        :label: trpo-eq-10

        J^{R}\left(\pi_{\text {new }}\right)  &\geq L_{\pi_{\text {old }}}\left(\pi_{\text {new }}\right)-\frac{4 \epsilon \gamma}{(1-\gamma)^2} \alpha^2 \\
        \text { where } \epsilon &=\max _{s, a}\left|A^{R}_{\pi}(s, a)\right|

    +++
    The proof of the :bdg-info-line:`Theorem 2` can be seen in the :bdg-ref-info:`Appendix`, click on this :bdg-info-line:`card` to jump to view.

The proof extends Kakade and Langford's result. Given the fact that the random
variables from two distributions with total variation
divergence less than :math:`\alpha` can be coupled,
we easily obtain that they are equal with probability :math:`1-\alpha`.

Next, we note the following relationship between the total variation divergence
and the :math:`\mathrm{KL}` divergence:
:math:`[D_{\mathrm{TV}}(p \| q)]^2 \leq D_{\mathrm{KL}}(p \| q)`.
Let
:math:`D_{\mathrm{KL}}^{\max }(\pi, \pi')=\underset{s}{\max} D_{\mathrm{KL}}(\pi(\cdot|s) \| \pi'(\cdot|s))`.
The following bound then follows directly from :bdg-info-line:`Theorem 2` :

.. _`trpo-eq-11`:

.. math::
    :label: trpo-eq-11

    J^R(\pi') & \geq L_\pi(\pi')-C D_{\mathrm{KL}}^{\max }(\pi, \pi') \\
    \quad \text { where } C &=\frac{4 \epsilon \gamma}{(1-\gamma)^2}


TRPO describes an approximate policy iteration scheme based on the policy
improvement bound in :eq:`trpo-eq-11`.
Note that for now, we assume exact evaluation of the advantage values :math:`A^{R}_{\pi}`.

It follows from :eq:`trpo-eq-11` that TRPO is guaranteed to generate a
monotonically improving sequence of policies
:math:`J^R\left(\pi_0\right) \leq J^R\left(\pi_1\right) \leq J^R\left(\pi_2\right) \leq \cdots \leq J^R\left(\pi_n\right)`.
To see this, let
:math:`M_i(\pi)=L_{\pi_i}(\pi)-C D_{\mathrm{KL}}^{\max }\left(\pi_i, \pi\right)`.
Then

.. math::
    :label: trpo-eq-12

    J^{R}\left(\pi_{i+1}\right) &\geq M_i\left(\pi_{i+1}\right) \\
    J^{R}\left(\pi_i\right)&=M_i\left(\pi_i\right), \text { therefore, } \\
    J^{R}\left(\pi_{i+1}\right)-\eta\left(\pi_i\right)&\geq M_i\left(\pi_{i+1}\right)-M\left(\pi_i\right)


Thus, by maximizing :math:`M_i` at each iteration, we guarantee that the true
objective :math:`J^R` is non-decreasing.

.. _trust-region-policy-optimization-1:

------

Practical Implementation
------------------------

Approximately Solving the TRPO Update
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Until now, we present the iteration algorithm with theoretically guaranteed
monotonic improvement for new policy over the current policy.
However, in practice, when we consider policies in parameterized space
:math:`\pi_{{\boldsymbol{\theta}}}(a \mid s)`,
the algorithm cannot work well. By plugging in the notation :math:`{\boldsymbol{\theta}}`, our
update step becomes

.. math::
    :label: trpo-eq-13

    & L_{{\boldsymbol{\theta}}_{old}}({\boldsymbol{\theta}})-C D_{\mathrm{KL}}^{\max }({\boldsymbol{\theta}}_{old}, {\boldsymbol{\theta}}) \\


where :math:`C=\frac{4 \epsilon \gamma}{(1-\gamma)^2}`,
and :math:`{\boldsymbol{\theta}}_{old}, {\boldsymbol{\theta}}`
are short for :math:`\pi_{{\boldsymbol{\theta}}_{old}}, \pi_{{\boldsymbol{\theta}}}`.
In practice, the penalty coefficient :math:`C` for KL divergence would produce
a very small step size and the improvement would be too conservative.
To allow larger step size, instead of penalty term on KL divergence,
TRPO uses fixed KL divergence constraint to bound the distance between
:math:`\pi_{{\boldsymbol{\theta}}_{old}}` and :math:`\pi_{{\boldsymbol{\theta}}}`:

.. math::
    :label: trpo-eq-14

    \underset{{\boldsymbol{\theta}}}{\max}\quad  &L_{{\boldsymbol{\theta}}_{old}}({\boldsymbol{\theta}}) \\
    \text{s.t. } \quad &D_{\mathrm{KL}}^{\max }({\boldsymbol{\theta}}_{old}, {\boldsymbol{\theta}}) \le \delta


This problem imposes a constraint that the KL divergence is bounded at every
point in the state space.
While it is motivated by the theory,
this problem is impractical to solve due to a large number of constraints.
Instead, TRPO uses a heuristic approximation that considers the average KL
divergence:

.. math::
    :label: trpo-eq-15

    \underset{{\boldsymbol{\theta}}}{\max}\quad  &L_{{\boldsymbol{\theta}}_{old}}({\boldsymbol{\theta}}) \label{eq:maxklconst} \\
    \text{s.t. } \quad &\bar{D}_{\mathrm{KL}}({\boldsymbol{\theta}}_{old}, {\boldsymbol{\theta}}) \le \delta


where
:math:`\bar{D}_{\mathrm{KL}}:=\mathbb{E}_{s \sim \rho}\left[D_{\mathrm{KL}}\left(\pi_{{\boldsymbol{\theta}}_1}(\cdot \mid s) \| \pi_{{\boldsymbol{\theta}}_2}(\cdot \mid s)\right)\right]`
.The method TRPO describes involves two steps:

.. card::
    :class-header: sd-bg-success sd-text-white sd-font-weight-bold
    :class-card: sd-outline-success  sd-rounded-1 sd-font-weight-bold

    Two Steps For TRPO Update
    ^^^
    -  Compute a search direction, using a linear approximation to the objective and quadratic approximation to the constraint.

    -  Perform a line search in the specified direction, ensuring both improvement of the nonlinear objective and satisfaction of the nonlinear constraint.

.. grid:: 2

    .. grid-item::
      :columns: 12 6 6 5

      .. card::
         :class-header: sd-bg-warning sd-text-white sd-font-weight-bold
         :class-card: sd-outline-warning  sd-rounded-1 sd-font-weight-bold

         Problems
         ^^^
         -  It is prohibitively costly to form the full Hessian matrix.

         -  How to compute the maximal step length such that the KL divergence is satisfied ?

         -  How to ensure improvement of the surrogate objective and satisfaction of the KL divergence ?
    .. grid-item::
      :columns: 12 6 6 6

      .. card::
         :class-header: sd-bg-primary sd-text-white sd-font-weight-bold
         :class-card: sd-outline-primary  sd-rounded-1 sd-font-weight-bold

         Solutions
         ^^^
         -  :bdg-ref-success-line:`Conjugate gradient algorithm<conjugate>` can approximately search the update direction without forming this full Hessian matrix.

         -  The max step size can be formed by an intermediate result produced by the conjugate gradient algorithm.

         -  A :bdg-ref-success-line:`line search algorithm<conjugate>` can be used to meet the goal.

.. tab-set::

    .. tab-item:: Computing the Fisher-Vector Product

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success  sd-rounded-1
            :class-footer: sd-font-weight-bold
            :link: conjugate
            :link-type: ref

            Computing the Fisher-Vector Product
            ^^^
            TRPO approximately computes the search direction by solving the equation :math:`Hx=g`,
            where :math:`H` is the Fisher information matrix, i.e.,
            the quadratic approximation to the KL divergence constraint :math:`\bar{D}_{\mathrm{KL}}\left({\boldsymbol{\theta}}_{\text {old }}, {\boldsymbol{\theta}}\right) \approx \frac{1}{2}\left({\boldsymbol{\theta}}-{\boldsymbol{\theta}}_{\text {old }}\right)^T H\left({\boldsymbol{\theta}}-{\boldsymbol{\theta}}_{\text {old }}\right)`,
            where :math:`H_{i j}=\frac{\partial}{\partial {\boldsymbol{\theta}}_i} \frac{\partial}{\partial {\boldsymbol{\theta}}_j} \bar{D}_{\mathrm{KL}}\left({\boldsymbol{\theta}}_{\text {old }}, {\boldsymbol{\theta}}\right)` (according to the definition of matrix :math:`H`).
            It is very difficult to calculate the entire :math:`H` or :math:`H^{-1}` directly,
            so TRPO uses the conjugate gradient algorithm to approximately solve the equation :math:`Hx=g` without forming this full matrix.
            +++
            The implementation of :bdg-success-line:`Computing the Fisher-Vector Product` can be seen in the :bdg-success:`Code with OmniSafe`, click on this :bdg-success-line:`card` to jump to view.


    .. tab-item:: Computing The Final Update Step

        .. card::
            :class-header: sd-bg-success  sd-text-white sd-font-weight-bold
            :class-card:  sd-outline-success  sd-rounded-1
            :class-footer: sd-font-weight-bold
            :link: conjugate
            :link-type: ref

            Computing The Final Update Step
            ^^^
            Having computed the search direction :math:`s\approx H^{-1}g`,
            TRPO next needs to compute the appropriate step to ensure improvement of the surrogate objective and satisfaction of the KL divergence constraint.
            First, TRPO computes the maximal step length :math:`\beta` such that :math:`{\boldsymbol{\theta}} + \beta s` will satisfy the KL divergence constraint.
            To do this, let :math:`\delta=\bar{D}_{\mathrm{KL}} \approx \frac{1}{2}(\beta s)^T H(\beta s)=\frac{1}{2} \beta^2 s^T A s`.
            Finally, we obtain :math:`\beta=\sqrt{2 \delta / s^T H s}`.

            .. hint::
                The term :math:`s^T H s` is an intermediate result produced by the conjugate gradient algorithm.

            To meet the constraints, TRPO uses line search algorithm to compute the final step length.
            Detailedly, TRPO performs the line search on the objective :math:`L_{{\boldsymbol{\theta}}_{\text {old }}}({\boldsymbol{\theta}})-\mathcal{X}\left[\bar{D}_{\text {KL }}\left({\boldsymbol{\theta}}_{\text {old }}, {\boldsymbol{\theta}}\right) \leq \delta\right]`, where :math:`\mathcal{X}[\ldots]` equals to :math:`0`,
            when its argument is true, and :math:`+\infty` when it is false.
            Starting with the maximal value of the step length :math:`\beta` computed in the previous paragraph,
            TRPO shrinks :math:`\beta` exponentially until the objective improves. Without this line search,
            the algorithm occasionally computes large steps that cause a catastrophic degradation of performance.
            +++
            The implementation of :bdg-success-line:`Computing The Final Update Step` can be seen in the :bdg-success:`Code with OmniSafe`, click on this :bdg-success-line:`card` to jump to view.

.. _trpo-Code_with_OmniSafe:

Code with OmniSafe
~~~~~~~~~~~~~~~~~~

Quick start
"""""""""""

.. card::
    :class-header: sd-bg-success sd-text-white sd-font-weight-bold
    :class-card: sd-outline-success  sd-rounded-1 sd-font-weight-bold
    :class-footer: sd-font-weight-bold

    Run TRPO in OmniSafe
    ^^^

    Here are 3 ways to run TRPO in OmniSafe:

    * Run Agent from preset yaml file
    * Run Agent from custom config dict
    * Run Agent from custom terminal config

    .. tab-set::

        .. tab-item:: Yaml file style

            .. code-block:: python
                :linenos:

                import omnisafe


                env_id = 'SafetyPointGoal1-v0'

                agent = omnisafe.Agent('TRPO', env_id)
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

                agent = omnisafe.Agent('TRPO', env_id, custom_cfgs=custom_cfgs)
                agent.learn()


        .. tab-item:: Terminal config style

            We use ``train_policy.py`` as the entrance file. You can train the agent with TRPO simply using ``train_policy.py``, with arguments about TRPO and environments does the training.
            For example, to run TRPO in SafetyPointGoal1-v0 , with 1 torch thread, seed 0 and single environment, you can use the following command:

            .. code-block:: bash
                :linenos:

                cd examples
                python train_policy.py --algo TRPO --env-id SafetyPointGoal1-v0 --parallel 1 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 1

------

Architecture of functions
"""""""""""""""""""""""""

- ``TRPO.learn()``

  - ``TRPO._env.rollout()``
  - ``TRPO._update()``

    - ``TRPO._buf.get()``
    - ``TRPO._update_actor()``

      - ``TRPO._fvp()``
      - ``conjugate_gradients()``
      - ``TRPO._cpo_search_step()``

    - ``TRPO._update_reward_critic()``

------

.. _conjugate:

Documentation of algorithm specific functions
"""""""""""""""""""""""""""""""""""""""""""""

.. tab-set::

    .. tab-item:: trpo._fvp()

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success  sd-rounded-1 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            trpo._fvp()
            ^^^
            TRPO algorithm builds the Hessian-vector product instead of the full Hessian matrix based on an approximation of the KL-divergence,
            flowing the next steps:

            (1) Calculate the KL divergence between two policy.
                Note that ``self._actor_critic.actor`` denotes the actor :math:`\pi` and ``kl`` denotes the KL divergence.

            .. code-block:: python
                :linenos:

                self._actor_critic.actor.zero_grad()
                q_dist = self._actor_critic.actor(self._fvp_obs)
                with torch.no_grad():
                    p_dist = self._actor_critic.actor(self._fvp_obs)
                kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).mean()

            (2) Use ``torch.autograd.grad()`` to compute the Hessian-vector product.
                Please note that in we compute the gradient of ``kl_p`` (The product of the Jacobian of KL divergence and :math:`g`) instead of ``grads`` (The Jacobian of KL divergence)

            .. code-block:: python
                :linenos:

                grads = torch.autograd.grad(
                    kl,
                    tuple(self._actor_critic.actor.parameters()),
                    create_graph=True,
                )
                flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

                kl_p = (flat_grad_kl * params).sum()
                grads = torch.autograd.grad(
                    kl_p,
                    tuple(self._actor_critic.actor.parameters()),
                    retain_graph=False,
                )

            (3) return the Hessian-vector product.

            .. code-block:: python
                :linenos:

                flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads])
                distributed.avg_tensor(flat_grad_grad_kl)

                return flat_grad_grad_kl + params * self._cfgs.algo_cfgs.cg_damping

    .. tab-item:: conjugate_gradients()

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success  sd-rounded-1 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            conjugate_gradients()
            ^^^
            TRPO algorithm uses conjugate gradients algorithm to search the update direction with Hessian-vector product,
            The conjugate gradient descent method attempts to solve problem :math:`Hx=g`
            flowing the next steps:

            (1) Set the initial solution ``x`` and calculate the error ``r`` between the ``x`` and the target ``b_vector`` (:math:`g` in above equation). Note that ``Fvp`` is the Hessian-vector product, which denotes :math:`H`.

            .. code-block:: python
                :linenos:

                vector_x = torch.zeros_like(vector_b)
                vector_r = vector_b - fisher_product(vector_x)
                vector_p = vector_r.clone()
                rdotr = torch.dot(vector_r, vector_r)

            (2) Performs ``n_step`` conjugate gradient.

            .. code-block:: python
                :linenos:

                for _ in range(num_steps):
                    vector_z = fisher_product(vector_p)
                    alpha = rdotr / (torch.dot(vector_p, vector_z) + eps)
                    vector_x += alpha * vector_p
                    vector_r -= alpha * vector_z
                    new_rdotr = torch.dot(vector_r, vector_r)
                    if torch.sqrt(new_rdotr) < residual_tol:
                        break
                    vector_mu = new_rdotr / (rdotr + eps)
                    vector_p = vector_r + vector_mu * vector_p
                    rdotr = new_rdotr
                return vector_x

            (3) Return the solution of :math:`x` without computing :math:`x=H^{-1}g`.


    .. tab-item:: trpo._search_step_size()

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success  sd-rounded-1 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            trpo._search_step_size()
            ^^^
            TRPO algorithm performs line-search to ensure constraint satisfaction for rewards and costs,
            and search around for a satisfied step of policy update to improve loss and reward performance,
            flowing the next steps:

            (1) Get the current policy parameters and initialize the step size.

            .. code-block:: python
                :linenos:

                # How far to go in a single update
                step_frac = 1.0
                # Get old parameterized policy expression
                theta_old = get_flat_params_from(self._actor_critic.actor)

            (2) Calculate the expected reward improvement.

            .. code-block:: python
                :linenos:

                expected_improve = g_flat.dot(step_dir)

            (3) Performs line-search to find a step improve the surrogate while not violating trust region.

            - Search acceptance step ranging from 0 to total step.

            .. code-block:: python
                :linenos:

                # While not within_trust_region and not out of total_steps:
                for step in range(total_steps):
                    # update theta params
                    new_theta = theta_old + step_frac * step_direction
                    # set new params as params of net
                    set_param_values_to_model(self._actor_critic.actor, new_theta)

            - In each step of for loop, calculate the policy performance and KL divergence.

            .. code-block:: python
                :linenos:

                with torch.no_grad():
                    loss, _ = self._loss_pi(obs, act, logp, adv)
                    # compute KL distance between new and old policy
                    q_dist = self._actor_critic.actor(obs)
                    # KL-distance of old p-dist and new q-dist, applied in KLEarlyStopping
                    kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).mean().item()
                    kl = distributed.dist_avg(kl)

            - Step only if surrogate is improved and within the trust region.

            .. code-block:: python
                :linenos:

                # real loss improve: old policy loss - new policy loss
                loss_improve = loss_before - loss.item()
                # average processes.... multi-processing style like: mpi_tools.mpi_avg(xxx)
                loss_improve = distributed.dist_avg(loss_improve)
                self._logger.log(f'Expected Improvement: {expected_improve} Actual: {loss_improve}')
                if not torch.isfinite(loss):
                    self._logger.log('WARNING: loss_pi not finite')
                elif loss_improve < 0:
                    self._logger.log('INFO: did not improve improve <0')
                elif kl > self._cfgs.algo_cfgs.target_kl:
                    self._logger.log('INFO: violated KL constraint.')
                else:
                    # step only if surrogate is improved and when within trust reg.
                    acceptance_step = step + 1
                    self._logger.log(f'Accept step at i={acceptance_step}')
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

                The following configs are specific to TRPO algorithm.

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

Reference
---------

-  `A Natural Policy
   Gradient <https://proceedings.neurips.cc/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf>`__
-  `Trust Region Policy
   Optimization <https://arxiv.org/abs/1502.05477>`__

Appendix
--------

:bdg-ref-info-line:`Click here to jump to TRPO Theorem<trpo-Theorem 1>`

:bdg-ref-success-line:`Click here to jump to Code withOmniSafe<trpo-Code_with_OmniSafe>`

.. _appendix-theorem1:

Proof of Theorem 1 (Difference between two arbitrary policies)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. card::
    :class-header: sd-bg-info sd-text-white sd-font-weight-bold
    :class-card: sd-outline-info  sd-rounded-1

    Proof of Theorem 1
    ^^^
    First note that :math:`A^{R}_{\pi}(s, a)=\mathbb{E}_{s' \sim \mathbb{P}\left(s^{\prime} \mid s, a\right)}\left[r(s)+\gamma V^R_{\pi}\left(s^{\prime}\right)-V^R_{\pi}(s)\right]`.
    Therefore,

    .. _`trpo-eq-15`:

    .. math::
        :label: trpo-eq-16

        \mathbb{E}_{\tau \sim \pi'}\left[\sum_{t=0}^{\infty} \gamma^t A^{R}_{\pi}\left(s_t, a_t\right)\right] &=\mathbb{E}_{\tau \sim \pi'}\left[\sum _ { t = 0 } ^ { \infty } \gamma ^ { t } \left(r\left(s_t\right)+\gamma V^{R}_{\pi}\left(s_{t+1}\right)-V^{R}_{\pi}\left(s_{t} \right)\right) \right] \\
        &=\mathbb{E}_{\tau \sim \pi'}\left[-V^R_{\pi}\left(s_0\right)+\sum_{t=0}^{\infty} \gamma^t r\left(s_t\right)\right] \\
        &=-\mathbb{E}_{s_0}\left[V^R_{\pi}\left(s_0\right)\right]+\mathbb{E}_{\tau \sim \pi'}\left[\sum_{t=0}^{\infty} \gamma^t r\left(s_t\right)\right] \\
        &=-J^R(\pi)+J^R(\pi')

.. _appendix-corollary1:

Proof of Corollary 1
~~~~~~~~~~~~~~~~~~~~

.. card::
    :class-header: sd-bg-info sd-text-white sd-font-weight-bold
    :class-card: sd-outline-info  sd-rounded-1

    Proof of Corollary 1
    ^^^
    From :eq:`trpo-eq-2` and :eq:`trpo-eq-4` , we can easily know that

    .. math::
        :label: trpo-eq-17

        & L_{\pi_{{\boldsymbol{\theta}}_0}}\left(\pi_{{\boldsymbol{\theta}}_0}\right)=J^{R}\left(\pi_{{\boldsymbol{\theta}}_0}\right)\quad \\
        \text{since}~~ &\sum_s \rho_\pi(s) \sum_a \pi'(a \mid s) A^{R}_{\pi}(s, a)=0.

    Now :eq:`trpo-eq-4` can be written as follows:

    .. math::
        :label: trpo-eq-18

        J^{R}\left(\pi^{'}_{{\boldsymbol{\theta}}}\right) = J^{R}(\pi_{{\boldsymbol{\theta}}_0}) + \sum_s d_{\pi^{'}_{{\boldsymbol{\theta}}}}(s) \sum_a \pi^{'}_{{\boldsymbol{\theta}}}(a|s) A^{R}_{\pi_{{\boldsymbol{\theta}}_0}}(s,a)

    So,

    .. _`trpo-eq-18`:

    .. math::
        :label: trpo-eq-19

        \nabla_{{\boldsymbol{\theta}}} J^{R}(\pi_{{\boldsymbol{\theta}}})|_{{\boldsymbol{\theta}} = {\boldsymbol{\theta}}_0} &= J^{R}(\pi_{{\boldsymbol{\theta}}_0}) + \sum_s \nabla d_{\pi_{{\boldsymbol{\theta}}}}(s) \sum_a \pi_{{\boldsymbol{\theta}}}(a|s) A^{R}_{\pi_{{\boldsymbol{\theta}}_0}}(s,a)+\sum_s d_{\pi_{{\boldsymbol{\theta}}}}(s) \sum_a \nabla \pi_{{\boldsymbol{\theta}}}(a|s) A^{R}_{\pi_{{\boldsymbol{\theta}}_0}}(s,a) \\
        &= J^{R}(\pi_{{\boldsymbol{\theta}}_0}) + \sum_s d_{\pi_{{\boldsymbol{\theta}}}}(s) \sum_a \nabla \pi_{{\boldsymbol{\theta}}}(a|s) A^{R}_{\pi_{{\boldsymbol{\theta}}_0}}(s,a)

    .. note::
        :math:`\sum_s \nabla d_{\pi_{{\boldsymbol{\theta}}}}(s) \sum_a \pi_{{\boldsymbol{\theta}}}(a|s) A^{R}_{\pi_{{\boldsymbol{\theta}}}}(s,a)=0`

    Meanwhile,

    .. _`trpo-eq-19`:

    .. math::
        :label: trpo-eq-20

        L_{\pi_{{\boldsymbol{\theta}}_0}}(\pi_{{\boldsymbol{\theta}}})=J^{R}(\pi_{{\boldsymbol{\theta}}_0})+\sum_s d_{\pi_{{\boldsymbol{\theta}}_0}}(s) \sum_a \pi_{{\boldsymbol{\theta}}}(a \mid s) A^{R}_{\pi_{{\boldsymbol{\theta}}_0}}(s, a)

    So,

    .. math::
        :label: trpo-eq-21

        \nabla L_{\pi_{{\boldsymbol{\theta}}_0}}(\pi_{{\boldsymbol{\theta}}}) | _{{\boldsymbol{\theta}} = {\boldsymbol{\theta}}_0}=J^{R}(\pi_{{\boldsymbol{\theta}}_0})+\sum_s d_{\pi_{{\boldsymbol{\theta}}_0}}(s) \sum_a \nabla \pi_{{\boldsymbol{\theta}}}(a \mid s) A^{R}_{\pi_{{\boldsymbol{\theta}}_0}}(s, a)


    Combine :eq:`trpo-eq-19`  and
    :eq:`trpo-eq-20`, we have

    .. math::
        :label: trpo-eq-22

        \left.\nabla_{\boldsymbol{\theta}} L_{\pi_{{\boldsymbol{\theta}}_0}}\left(\pi_{\boldsymbol{\theta}}\right)\right|_{{\boldsymbol{\theta}}={\boldsymbol{\theta}}_0}=\left.\nabla_{\boldsymbol{\theta}} J^{R}\left(\pi_{\boldsymbol{\theta}}\right)\right|_{{\boldsymbol{\theta}}={\boldsymbol{\theta}}_0}

.. _appendix-theorem2:

Proof of Theorem 2 (Difference between two arbitrary policies)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define :math:`\bar{A}^R(s)` as the expected advantage of :math:`\pi'` over
:math:`\pi` at :math:`s`,

.. math::
    :label: trpo-eq-23

    \bar{A}^R(s)=\mathbb{E}_{a \sim \pi^{'}(\cdot \mid s)}\left[A^{R}_{\pi}(s, a)\right]


:bdg-info-line:`Theorem 1` can be written as follows:

.. math::
    :label: trpo-eq-24

    J^R(\pi')=J^R(\pi)+\mathbb{E}_{\tau \sim \pi'}\left[\sum_{t=0}^{\infty} \gamma^t \bar{A}^R\left(s_t\right)\right]


Note that :math:`L_\pi` can be written as

.. math::
    :label: trpo-eq-25

    L_\pi(\pi')=J^R(\pi)+\mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^{\infty} \gamma^t \bar{A}^R\left(s_t\right)\right]


To bound the difference between :math:`J^R(\pi')` and :math:`L_\pi(\pi')`,
we need to bound the difference arising from each timestep.
To do this, we first need to introduce a measure of how much :math:`\pi` and
:math:`\pi'` agree.
Specifically, we'll couple the policies,
so that define a joint distribution over pairs of actions.

.. card::
    :class-header: sd-bg-info sd-text-white sd-font-weight-bold
    :class-card: sd-outline-info  sd-rounded-1

    Definition 1
    ^^^
    :math:`(\pi, \pi')` is an :math:`\alpha`-coupled policy pair if it
    defines a joint distribution :math:`(a, a')|s`, such that
    :math:`P(a \neq a'|s) \leq \alpha` for all s.
    :math:`\pi` and :math:`\pi'` will denote the marginal distributions of a and :math:`a'`, respectively.

Computationally, :math:`\alpha`-coupling means that if we randomly choose a
seed for our random number generator,
and then we sample from each of :math:`\pi` and :math:`\pi'` after setting that
seed,
the results will agree for at least fraction :math:`1-\alpha` of seeds.

.. tab-set::

    .. tab-item:: Lemma 1
        :sync: key1

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Lemma 1
            ^^^
            Given that :math:`\pi, \pi'` are :math:`\alpha`-coupled policies,
            for all s,

            .. _`trpo-eq-25`:

            .. math::
                :label: trpo-eq-26

                |\bar{A}^R(s)| \leq 2 \alpha \max _{s, a}\left|A^{R}_{\pi}(s, a)\right|



    .. tab-item:: Lemma 2
        :sync: key2

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Lemma 2
            ^^^
            Let :math:`(\pi, \pi')` be an :math:`\alpha`-coupled policy pair.
            Then

            .. math::
                :label: trpo-eq-27

                \left|\mathbb{E}_{s_t \sim \pi'}\left[\bar{A}^R\left(s_t\right)\right]-\mathbb{E}_{s_t \sim \pi}\left[\bar{A}^R\left(s_t\right)\right]\right|&\leq 2 \alpha \max _s \bar{A}^R(s) \\
                &\leq 4 \alpha\left(1-(1-\alpha)^t\right) \max _s\left|A^{R}_{\pi}(s, a)\right|


.. tab-set::

    .. tab-item:: Proof of Lemma 1
        :sync: key1

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Proof of Lemma 1
            ^^^

            .. _`trpo-eq-26`:

            .. math::
                :label: trpo-eq-28

                \bar{A}^R(s) &= \mathbb{E}_{\tilde{a} \sim \tilde{\pi}}\left[A^{R}_{\pi}(s, \tilde{a})\right] - \mathbb{E}_{a \sim \pi}\left[A^{R}_{\pi}(s, a)\right] \\
                &=\mathbb{E}_{(a, \tilde{a}) \sim(\pi, \tilde{\pi})}\left[A^{R}_{\pi}(s, \tilde{a})-A^{R}_{\pi}(s, a)\right]\\
                &= P(a \neq \tilde{a} \mid s) \mathbb{E}_{(a, \tilde{a}) \sim(\pi, \tilde{\pi}) \mid a \neq \tilde{a}}\left[A^{R}_{\pi}(s, \tilde{a})-A^{R}_{\pi}(s, a)\right]


            So,

            .. math::
                :label: trpo-eq-29

                |\bar{A}^R(s)|  \leq \alpha \cdot 2 \max _{s, a}\left|A^{R}_{\pi}(s, a)\right|


    .. tab-item:: Proof of Lemma 2
        :sync: key2

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Proof of Lemma 2
            ^^^
            Given the coupled policy pair :math:`(\pi, \pi')`,
            we can also obtain a coupling over the trajectory distributions produced by :math:`\pi` and :math:`\pi'`,
            respectively. Namely, we have pairs of trajectories :math:`\tau, \tau'`,
            where :math:`\tau` is obtained by taking actions from :math:`\pi`,
            and :math:`\tau'` is obtained by taking actions from :math:`\pi'`,
            where the same random seed is used to generate both trajectories.
            We will consider the advantage of :math:`\pi'` over :math:`\pi` at timestep :math:`t`,
            and decompose this expectation based on whether :math:`\pi` agrees with :math:`\pi'` at all timesteps :math:`i<t`

            Let :math:`n_t` denote the number of times that :math:`a_i \neq a^{'}_i` for :math:`i<t`,
            i.e., the number of times that :math:`\pi` and :math:`\pi'` disagree before timestep :math:`t`.

            .. math::
                :label: trpo-eq-30

                \mathbb{E}_{s_t \sim \pi'}\left[\bar{A}^R\left(s_t\right)\right]&=P\left(n_t=0\right) \mathbb{E}_{s_t \sim \pi' \mid n_t=0}\left[\bar{A}^R\left(s_t\right)\right]\\
                &+P\left(n_t>0\right) \mathbb{E}_{s_t \sim \pi' \mid n_t>0}\left[\bar{A}^R\left(s_t\right)\right]


            The expectation decomposes similarly for actions are sampled using
            :math:`\pi` :

            .. math::
                :label: trpo-eq-31

                \mathbb{E}_{s_t \sim \pi}\left[\bar{A}^R\left(s_t\right)\right]&=P\left(n_t=0\right) \mathbb{E}_{s_t \sim \pi \mid n_t=0}\left[\bar{A}^R\left(s_t\right)\right]\\
                &+P\left(n_t>0\right) \mathbb{E}_{s_t \sim \pi \mid n_t>0}\left[\bar{A}^R\left(s_t\right)\right]


            Note that the :math:`n_t=0` terms are equal:

            .. math::
                :label: trpo-eq-32

                \mathbb{E}_{s_t \sim \pi' \mid n_t=0}\left[\bar{A}^R\left(s_t\right)\right]=\mathbb{E}_{s_t \sim \pi \mid n_t=0}\left[\bar{A}^R\left(s_t\right)\right]


            because :math:`n_t=0` indicates that :math:`\pi` and :math:`\pi'` agreed on all timesteps less than :math:`t`.
            Subtracting Equations :eq:`trpo-eq-26` and :eq:`trpo-eq-27`, we get

            .. _`trpo-eq-32`:

            .. math::
                :label: trpo-eq-33

                &\mathbb{E}_{s_t \sim \pi'}\left[\bar{A}^R\left(s_t\right)\right]-\mathbb{E}_{s_t \sim \pi}\left[\bar{A}^R\left(s_t\right)\right]
                \\
                =&P\left(n_t>0\right)\left(\mathbb{E}_{s_t \sim \pi' \mid n_t>0}\left[\bar{A}^R\left(s_t\right)\right]-\mathbb{E}_{s_t \sim \pi \mid n_t>0}\left[\bar{A}^R\left(s_t\right)\right]\right)
                \label{equation: sub for unfold}


            By definition of :math:`\alpha, P(\pi, \pi'` agree at timestep :math:`i) \geq 1-\alpha`,
            so :math:`P\left(n_t=0\right) \geq(1-\alpha)^t`, and

            .. _`trpo-eq-33`:

            .. math::
                :label: trpo-eq-34

                P\left(n_t>0\right) \leq 1-(1-\alpha)^t
                \label{equation: probability with a couple policy}


            Next, note that

            .. _`trpo-eq-34`:

            .. math::
                :label: trpo-eq-35

                &\left|\mathbb{E}_{s_t \sim \pi' \mid n_t>0}\left[\bar{A}^R\left(s_t\right)\right]-\mathbb{E}_{s_t \sim \pi \mid n_t>0}\left[\bar{A}^R\left(s_t\right)\right]\right| \\
                & \leq\left|\mathbb{E}_{s_t \sim \pi' \mid n_t>0}\left[\bar{A}^R\left(s_t\right)\right]\right|+\left|\mathbb{E}_{s_t \sim \pi \mid n_t>0}\left[\bar{A}^R\left(s_t\right)\right]\right| \\
                & \leq 4 \alpha \max _{s, a}\left|A^{R}_{\pi}(s, a)\right|
                \label{equation: abs performance bound nt geq 0}


            Where the second inequality follows from Lemma 2.
            Plugging :eq:`trpo-eq-34` and :eq:`trpo-eq-35` into :eq:`trpo-eq-33`, we get

            .. math::
                :label: trpo-eq-36

                \left|\mathbb{E}_{s_t \sim \pi'}\left[\bar{A}^R\left(s_t\right)\right]-\mathbb{E}_{s_t \sim \pi}\left[\bar{A}^R\left(s_t\right)\right]\right| \leq 4 \alpha\left(1-(1-\alpha)^t\right) \max _{s, a}\left|A^{R}_{\pi}(s, a)\right|


The preceding Lemma bounds the difference in expected advantage at each
timestep :math:`t`.
We can sum over time to bound the difference between :math:`J^R(\pi')` and
:math:`L_\pi(\pi')`. Subtracting :eq:`trpo-eq-24` and :eq:`trpo-eq-25`,
and defining :math:`\epsilon=\max _{s, a}\left|A^{R}_{\pi}(s, a)\right|`, we have

.. _`trpo-eq-36`:

.. math::
    :label: trpo-eq-37

    \left|J^R(\pi')-L_\pi(\pi')\right| &=\sum_{t=0}^{\infty} \gamma^t\left|\mathbb{E}_{\tau \sim \pi'}\left[\bar{A}^R\left(s_t\right)\right]-\mathbb{E}_{\tau \sim \pi}\left[\bar{A}^R\left(s_t\right)\right]\right|  \\
    & \leq \sum_{t=0}^{\infty} \gamma^t \cdot 4 \epsilon \alpha\left(1-(1-\alpha)^t\right)  \\
    &=4 \epsilon \alpha\left(\frac{1}{1-\gamma}-\frac{1}{1-\gamma(1-\alpha)}\right)  \\
    &=\frac{4 \alpha^2 \gamma \epsilon}{(1-\gamma)(1-\gamma(1-\alpha))}  \\
    & \leq \frac{4 \alpha^2 \gamma \epsilon}{(1-\gamma)^2} \label{TRPO: difference between L and J}


Last, to replace :math:`\alpha` by the total variation divergence,
we need to use the correspondence between TV divergence and coupled random
variables:

.. note::

    Suppose :math:`p_X` and :math:`p_Y` are distributions with
    :math:`D_{T V}\left(p_X \| p_Y\right)=\alpha`. Then there exists a
    joint distribution :math:`(X, Y)` whose marginals are
    :math:`p_X, p_Y`, for which :math:`X=Y` with probability
    :math:`1-\alpha`. More details in See (Levin et al., 2009),
    Proposition 4.7.

It follows that if we have two policies :math:`\pi` and :math:`\pi'`
such that

.. math::
    :label: trpo-eq-38

    \max_s D_{\mathrm{TV}}(\pi(\cdot|s) \| \pi'(\cdot|s)) \leq \alpha

then we can define an :math:`\alpha`-coupled policy pair :math:`(\pi, \pi')`
with appropriate marginals.
Taking
:math:`\alpha=\underset{s}{\max} D_{T V}\left(\pi(\cdot \mid s) \| \pi'(\cdot \mid s)\right)`
in :eq:`trpo-eq-37`, :bdg-info-line:`Theorem 2` follows.
