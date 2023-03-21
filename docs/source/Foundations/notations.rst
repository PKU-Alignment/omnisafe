Notations
=========

Introduction
------------
In this section, we will introduce the notations used in this tutorial.
In Reinforcement Learning, we often use the following two basic notations:
**Linear Algebra** and **Constrained Markov Decision Processes**.
Make sure you are familiar with this section before you start.
You can return to this section any time you are puzzled by some notations in the following chapters.

Linear Algebra
--------------

.. grid:: 2

    .. grid-item::
        :columns: 12 6 6 6

        .. card::
            :class-header: sd-bg-info sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-3

            Vector
            ^^^
            A vector is an ordered finite list of numbers.
            Typically, vectors are written as vertical arrays surrounded by square brackets,
            as in:

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
            :math:`\mathbf{a}=(a_1,a_2,\cdots,a_n)\in\mathbb{R}^{n}`), and its
            :math:`i^{th}` element written as
            :math:`\mathbf{a}[i]=:a_{i},~~1\leq i\leq n.`

    .. grid-item::
        :columns: 12 6 6 5

        .. card::
            :class-header: sd-bg-info sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-3

            Matrix
            ^^^
            Matrix, mathematical term.
            In mathematics, a matrix is a collection of complex or real numbers arranged in a rectangular array.

            Similarly, we use a bold capital letter to denote matrix, e.g.,
            :math:`\mathbf{A}=(a_{i,j})\in\mathbb{R}^{m\times n}`, and its :math:`(i,j)^{\text{th}}` element denoted as

            .. math:: \mathbf{A}[i,j]=:a_{i,j},

            where :math:`1\leq i\leq m,1\leq j\leq n`.

Constrained Markov Decision Processes
-------------------------------------

For the convenience of reference, we list key notations that have be used.


A **Reinforcement Learning (RL)** problem is often formulated as Infinite-horizon Discounted **Markov Decision Process (MDP)**.
It is a tuple :math:`\mathcal{M}=\{\mathcal{S}, \mathcal{A}, \mathbb{P}, r, \mu, \gamma\}`, where

.. card::
   :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
   :class-card: sd-outline-info  sd-rounded-3
   :class-footer: sd-font-weight-bold

   Key Notations
   ^^^
   -  :math:`\mathcal{S}` is a finite set of states;

   -  :math:`\mathcal{A}` is a finite set of actions;

   -  :math:`\mathbb{P}(\cdot|\cdot,\cdot)` is the transition
      probability distribution,
      :math:`\mathcal{S}\times\mathcal{A}\times\mathcal{S}\rightarrow[0,1]`;

   -  :math:`\mu` is the distribution of the initial state :math:`s_0`,
      :math:`\mathcal{S} \rightarrow \mathbb{R}` ;

   -  :math:`r` is the reward function,
      :math:`\mathcal{S} \rightarrow \mathbb{R}`;

   -  :math:`\gamma\in(0,1)` is the discount factor.

A stationary parameterized policy :math:`\pi_{\theta}` is a probability distribution defined on :math:`\mathcal{S}\times\mathcal{A}`,
:math:`\pi_{\theta}(a|s)` denotes the probability of playing :math:`a` in state :math:`s`. With explicit notation dropped to reduce clutter,
we use :math:`\boldsymbol{\theta}` to represent :math:`\pi_{\theta}`.

.. tab-set::

    .. tab-item:: From MDP

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-3
            :class-footer: sd-font-weight-bold

            Markov Decision Processes
            ^^^
            Let :math:`J(\boldsymbol{\theta})` denote its expected discounted reward,

            .. math:: J(\boldsymbol{\theta}) \doteq \mathbb{E}_{\tau \sim \boldsymbol{\theta}}\left[\sum_{t=0}^{\infty} \gamma^t r\left(s_t\right)\right],

            Here :math:`\tau` denotes a trajectory :math:`(s_0, a_0, s_1, ...)`,
            and :math:`\tau \sim \pi` is shorthand for indicating that the distribution over trajectories depends on a stationary parameterized policy
            :math:`\pi_{\theta}`: :math:`s_0 \sim \mu`,
            :math:`a_t \sim \boldsymbol{\theta}(\cdot|s_t)`,
            :math:`s_{t+1} \sim \mathbb{P}(\cdot | s_t, a_t)`.
            Meanwhile, let :math:`R(\tau)` denote the discounted return of a trajectory.

            The state action value function

            .. math:: Q^R_{\boldsymbol{\theta}} \left(s, a\right) \doteq \mathbb{E}_{\tau \sim \boldsymbol{\theta}}\left[ R(\tau) | s_0 = s, a_0 = a \right].

            The value function

            .. math:: V^R_{\boldsymbol{\theta}}\left(s\right) \doteq \mathbb{E}_{\tau \sim \boldsymbol{\theta}}\left[R(\tau) | s_0 = s\right].

            And the advantage function

            .. math:: A^R_{\boldsymbol{\theta}}(s, a) \doteq Q^R_{\boldsymbol{\theta}}(s, a)-V^R_{\boldsymbol{\theta}}(s).

            Let :math:`\mathbb{P}_{\pi}\left(s'\mid s\right)` denote one-step state transition probability from :math:`s` to :math:`s'` by executing :math:`\pi`,

            .. math:: \mathbb{P}_{\pi}\left(s'\mid s\right)=\sum_{a\in\mathbb{A}}\pi\left(a\mid s\right) \mathbb{P}_{\pi}\left(s'\mid s,a\right).

            Then for any initial state :math:`s_0 \sim \mu`, we have

            .. math:: \mathbb{P}_{\pi}\left(s_t=s\mid s_0\right)=\sum_{s'\in\mathbb{S}} \mathbb{P}_{\pi}\left(s_t=s\mid s_{t-1}=s'\right)\mathbb{P}_{\pi}\left(s_{t-1}=s'\mid s_0\right),

            where :math:`s_0 \sim \mu` and the actions are chosen according to :math:`\pi`.

            Let :math:`d_{\boldsymbol{\pi}}` be the (unnormalized) discounted visitation frequencies here need to explain :math:`\mathbb{P}` and P.

            .. math::

               \begin{aligned}
                  d_{\boldsymbol{\pi}}(s)&=\sum_{t=0}^{\infty} \gamma^t \mathbb{P}_{\pi}\left(s_t=s \mid s_0\right)\\
                  &=\mathbb{P}\left(s_0=s\right)+\gamma \mathbb{P}\left(s_1=s\mid s_0\right)+\gamma^2 \mathbb{P}\left(s_2=s\mid s_0\right)+\cdots.
               \end{aligned}

    .. tab-item:: To CMDP

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card:  sd-outline-info  sd-rounded-3
            :class-footer: sd-font-weight-bold

            Constrained Markov Decision Processes
            ^^^
            A **Constrained Markov Decision Process(CMDP)** extends the MDP framework by augmenting with constraints restricting the set of feasible policies. Specifically,
            we introduce a set :math:`C` of auxiliary cost functions:
            :math:`C_1, \cdots, C_m` and cost limits:
            :math:`d_1, \cdots, d_m`, that each of them :math:`C_i`:
            :math:`\mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow \mathbb{R}`
            mapping transition tuples to costs.

            Let :math:`J^{C_i}(\boldsymbol{\theta})` denote the expected discounted return of policy :math:`\boldsymbol{\theta}` in terms of cost function,

            .. math::

               \begin{aligned}
                  J^{C_i}(\boldsymbol{\theta}) = \mathbb{E}_{\tau \sim \boldsymbol{\theta}}[\sum_{t=0}^{\infty} \gamma^t C_i(s_t, a_t, s_{t+1})].
               \end{aligned}

            So, the feasible set of stationary parameterized policies for CMDP is

            .. math::

               \begin{aligned}
                  \Pi_{C} \doteq \{ \pi_{\theta} \in \Pi~:~\forall~i, ~ J^{C_i}(\boldsymbol{\theta}) \leq d_i \}
               \end{aligned}

            The goal of CMDP is to find the optimal policy :math:`\pi^{*}`:

            .. math::

               \begin{aligned}
                  \label{def:problem-setting}
                  \pi^{*}=\arg\max_{\pi_{\theta}\in\Pi_{C}} J(\pi_{\theta}).
               \end{aligned}

            Respectively we have:

            The state action value function

            .. math:: Q^{C}_{\boldsymbol{\theta}} \left(s, a\right) \doteq \mathbb{E}_{\tau \sim \boldsymbol{\theta}}\left[ C(\tau) | s_0 = s, a_0 = a \right].

            The value function

            .. math:: V^{C}_{\boldsymbol{\theta}}\left(s\right) \doteq \mathbb{E}_{\tau \sim \boldsymbol{\theta}}\left[C(\tau) | s_0 = s\right].

            And the advantage function

            .. math:: A^{C}_{\boldsymbol{\theta}}(s, a) \doteq Q^{C}_{\boldsymbol{\theta}}(s, a)-V^{C}_{\boldsymbol{\theta}}(s).

            To summarize all of the above notation, we show the following table,

References
----------

-  `Constrained Markov Decision Processes <https://www.semanticscholar.org/paper/Constrained-Markov-Decision-Processes-Altman/3cc2608fd77b9b65f5bd378e8797b2ab1b8acde7>`__
-  `Markov Decision Processes <https://dl.acm.org/doi/book/10.5555/551283>`__
