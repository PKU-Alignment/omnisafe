Mathematical Notations
======================

Introduction
------------

In this section, we will provide an introduction to the notations used
throughout this tutorial. In reinforcement learning, two fundamental notations
are commonly used: **linear algebra** and **constrained Markov decision
processes**. It is essential to familiarize yourself with these notations
before proceeding. When reading formulas in the following chapters, if you come
across a mathematical symbol whose meaning you're unsure of, refer to the
notations introduced in this chapter.

Linear Algebra
--------------

.. grid:: 2

    .. grid-item::
        :columns: 12 6 6 7

        .. card::
            :class-header: sd-bg-info sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-1

            Vector
            ^^^
            A vector is a mathematical object representing a quantity that has
            both magnitude and direction. It is an ordered finite list of
            numbers that can be written as a vertical array surrounded by
            square brackets.

            .. math::

               \boldsymbol{a} =
               \left[\begin{array}{r}
               a_1 \\
               a_2 \\
               \cdots \\
               a_n
               \end{array}\right]
               \in \mathbb{R}^n

            Usually, we use a bold lowercase letter to denote a vector (e.g.
            :math:`\boldsymbol{a}=(a_1,a_2,\cdots,a_n)\in\mathbb{R}^{n}`), and its
            :math:`i^{th}` element written as
            :math:`\boldsymbol{a}[i]=:\boldsymbol{a}_{i},~~1\leq i\leq n.`

    .. grid-item::
        :columns: 12 6 6 5

        .. card::
            :class-header: sd-bg-info sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-1

            Matrix
            ^^^
            *Matrix* is a mathematical term that refers to a collection of
            numbers, whether real or complex, arranged in a rectangular array.
            In this tutorial, we will use bold capital letters to denote
            matrices, such as the following example:
            :math:`\mathbf{A}=(a_{i,j})\in\mathbb{R}^{m\times n}`, and its :math:`(i,j)^{\text{th}}` element denoted as

            .. math:: \mathbf{A}[i,j]=:a_{i,j},

            where :math:`1\leq i\leq m,1\leq j\leq n`.

Constrained Markov Decision Processes
-------------------------------------

A **Reinforcement Learning (RL)** problem is typically formulated as
Infinite-horizon Discounted **Markov Decision Process (MDP)**.

It is a tuple
:math:`\mathcal{M}=\{\mathcal{S}, \mathcal{A}, \mathbb{P}, r, \mu, \gamma\}`,
where:

.. card::
   :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
   :class-card: sd-outline-info  sd-rounded-1
   :class-footer: sd-font-weight-bold

   Key Notations
   ^^^
   -  :math:`\mathcal{S}` is a finite set of states;

   -  :math:`\mathcal{A}` is a finite set of actions;

   -  :math:`\mathbb{P}(\cdot|\cdot,\cdot)` are the transition
      probability distribution,
      :math:`\mathcal{S}\times\mathcal{A}\times\mathcal{S}\rightarrow[0,1]`;

   -  :math:`\mu` are the distribution of the initial state :math:`s_0`,
      :math:`\mathcal{S} \rightarrow \mathbb{R}` ;

   -  :math:`r` are the reward function,
      :math:`\mathcal{S} \rightarrow \mathbb{R}`;

   -  :math:`\gamma\in(0,1)` are the discount factor.

A stationary parameterized policy :math:`\pi_{{\boldsymbol{\theta}}}` is a probability
distribution defined on :math:`\mathcal{S}\times\mathcal{A}`,
:math:`\pi_{{\boldsymbol{\theta}}}(a|s)` denotes the probability of
playing :math:`a` in state :math:`s`.
With explicit notation dropped to reduce clutter,
we use :math:`\pi` to represent :math:`\pi_{{\boldsymbol{\theta}}}`.

.. tab-set::

    .. tab-item:: From MDP

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Markov Decision Processes
            ^^^
            Let :math:`J^R(\pi)` denote its expected discounted reward,

            .. math:: J^R(\pi) \doteq \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^{\infty} \gamma^t r\left(s_t\right)\right]

            Here :math:`\tau` denotes a trajectory :math:`(s_0, a_0, s_1, ...)`,
            and :math:`\tau \sim \pi` is shorthand for indicating that the distribution over trajectories depends on a stationary parameterized policy
            :math:`\pi_{{\boldsymbol{\theta}}}`: :math:`s_0 \sim \mu`,
            :math:`a_t \sim \pi(\cdot|s_t)`,
            :math:`s_{t+1} \sim \mathbb{P}(\cdot | s_t, a_t)`.
            Meanwhile, let :math:`R(\tau)` denote the discounted return of a trajectory. :math:`R(\tau) = \sum_{t=0}^{\infty} \gamma^t r(s_t)`

            The state action value function

            .. math:: Q^R_{\pi} \left(s, a\right) \doteq \mathbb{E}_{\tau \sim \pi}\left[ R(\tau) | s_0 = s, a_0 = a \right]

            The value function

            .. math:: V^R_{\pi}\left(s\right) \doteq \mathbb{E}_{\tau \sim \pi}\left[R(\tau) | s_0 = s\right]

            And the advantage function

            .. math:: A^R_{\pi}(s, a) \doteq Q^R_{\pi}(s, a)-V^R_{\pi}(s)

            Let :math:`\mathbb{P}_{\pi}\left(s'\mid s\right)` denote one-step state transition probability from :math:`s` to :math:`s'` by executing :math:`\pi`,

            .. math:: \mathbb{P}_{\pi}\left(s'\mid s\right)=\sum_{a\in\mathcal{A}}\pi\left(a\mid s\right) \mathbb{P}_{\pi}\left(s'\mid s,a\right)

            Then for any initial state :math:`s_0 \sim \mu`, we have

            .. math:: \mathbb{P}_{\pi}\left(s_t=s\mid s_0\right)=\sum_{s'\in\mathcal{S}} \mathbb{P}_{\pi}\left(s_t=s\mid s_{t-1}=s'\right)\mathbb{P}_{\pi}\left(s_{t-1}=s'\mid s_0\right)

            where :math:`s_0 \sim \mu` and the actions are chosen according to :math:`\pi`.

            Let :math:`d_{\boldsymbol{\pi}}` be the (unnormalized) discounted visitation frequencies here need to explain :math:`\mathbb{P}`.

            .. math::

               \begin{aligned}
                  d_{\boldsymbol{\pi}}(s)&=\sum_{t=0}^{\infty} \gamma^t \mathbb{P}_{\pi}\left(s_t=s \mid s_0\right)\\
                  &=\mathbb{P}\left(s_0=s\right)+\gamma \mathbb{P}\left(s_1=s\mid s_0\right)+\gamma^2 \mathbb{P}\left(s_2=s\mid s_0\right)+\cdots
               \end{aligned}

    .. tab-item:: To CMDP

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card:  sd-outline-info  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Constrained Markov Decision Processes
            ^^^
            A **Constrained Markov Decision Process(CMDP)** extends the MDP framework by augmenting with constraints restricting the set of feasible policies. Specifically,
            we introduce a set :math:`C` of auxiliary cost functions:
            :math:`C_1, \cdots, C_m` and cost limits:
            :math:`d_1, \cdots, d_m`, that each of them :math:`C_i`:
            :math:`\mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow \mathbb{R}`
            mapping transition tuples to costs.

            Let :math:`J^{C_i}(\pi)` denote the expected discounted return of policy :math:`\pi` in terms of cost function,

            .. math::

               \begin{aligned}
                  J^{C_i}(\pi) = \mathbb{E}_{\tau \sim \pi}[\sum_{t=0}^{\infty} \gamma^t C_i(s_t, a_t, s_{t+1})]
               \end{aligned}

            So, the feasible set of stationary parameterized policies for CMDP is

            .. math::

               \begin{aligned}
                  \Pi_{C} \doteq \{ \pi_{{\boldsymbol{\theta}}} \in \Pi~:~\forall~i, ~ J^{C_i}(\pi) \leq d_i \}
               \end{aligned}

            The goal of CMDP is to find the optimal policy :math:`\pi^{*}`:

            .. math::

               \begin{aligned}
                  \label{def:problem-setting}
                  \pi^{*}=\arg\max_{\pi_{\boldsymbol{\theta}} \in\Pi_{C}} J^R(\pi_{{\boldsymbol{\theta}}})
               \end{aligned}

            Respectively we have:

            The state action value function

            .. math:: Q^{C}_{\pi} \left(s, a\right) \doteq \mathbb{E}_{\tau \sim \pi}\left[ C(\tau) | s_0 = s, a_0 = a \right]

            The value function

            .. math:: V^{C}_{\pi}\left(s\right) \doteq \mathbb{E}_{\tau \sim \pi}\left[C(\tau) | s_0 = s\right]

            And the advantage function

            .. math:: A^{C}_{\pi}(s, a) \doteq Q^{C}_{\pi}(s, a)-V^{C}_{\pi}(s)


To summarize all of the above notation, we show the following table,

- :math:`\tau` is a trajectory that consist of
  :math:`\left(s_0, a_0, s_1, a_1, \cdots\right)`
- :math:`\pi_{{\boldsymbol{\theta}}}` or :math:`{\boldsymbol{\theta}}` is a stationary parameterized policy
  which is a probability distribution defined on
  :math:`\mathcal{S}\times\mathcal{A}`, :math:`\pi_{{\boldsymbol{\theta}}}(a|s)`
  denotes the probability of playing :math:`a` in state :math:`s`.
- :math:`J^R(\pi_{{\boldsymbol{\theta}}}),~ J^R({\boldsymbol{\theta}})` are the expected discounted reward
  over trajectories, depending on a stationary parameterized policy
  :math:`\pi_{{\boldsymbol{\theta}}}` or a stationary parameterized policy
  :math:`\pi_{{\boldsymbol{\theta}}}`.
- :math:`J^{C}(\pi_{{\boldsymbol{\theta}}}),~ J^{C}({\boldsymbol{\theta}})` are the
  expected discounted cost over trajectories, depending on a stationary
  parameterized policy :math:`\pi_{{\boldsymbol{\theta}}}` or a stationary parameterized
  policy :math:`\pi_{{\boldsymbol{\theta}}}`.
- :math:`Q_{\pi_{{\boldsymbol{\theta}}}}^{R},~ Q_{{\boldsymbol{\theta}}}^{R}` are the state action value
  function for reward.
- :math:`Q_{\pi_{{\boldsymbol{\theta}}}}^{C_i},~  Q_{{\boldsymbol{\theta}}}^{C_i}` are the
  state action value function for cost.
- :math:`V_{\pi_{{\boldsymbol{\theta}}}}^{R},~  V_{{\boldsymbol{\theta}}}^{R}`
  are the value function for reward.
- :math:`V_{\pi_{{\boldsymbol{\theta}}}}^{C_i},~  V_{{\boldsymbol{\theta}}}^{C_i}`
  are the value function for cost.
- :math:`A_{\pi_{{\boldsymbol{\theta}}}}^{R},~  A_{{\boldsymbol{\theta}}}^{R}` are the advantage function for
  reward.
- :math:`A_{\pi_{{\boldsymbol{\theta}}}}^{C_i},~  A_{{\boldsymbol{\theta}}}^{C_i}`
  are the advantage function for cost.


References
----------

-  `Constrained Markov Decision Processes <https://www.semanticscholar.org/paper/Constrained-Markov-Decision-Processes-Altman/3cc2608fd77b9b65f5bd378e8797b2ab1b8acde7>`__
-  `Markov Decision Processes <https://dl.acm.org/doi/book/10.5555/551283>`__
-  `Convex Optimization <https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf>`__
