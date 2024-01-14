Lagrange Duality
================

.. _`lagrange_theorem`:

Primal Problem
--------------

Consider a general optimization problem (called as the primal problem):

.. _preknow-eq-1:

.. math::
    :label: preknow-eq-1

    \underset{x}{\min} & f(x) \\
    \text { s.t. } & h_i(x) \leq 0, i=1, \cdots, m \\
    & \ell_j(x)=0, j=1, \cdots, r


We define its Lagrangian dual version as:

.. math:: L(x, u, v)=f(x)+\sum_{i=1}^m u_i h_i(x)+\sum_{j=1}^r v_j \ell_j(x)

Lagrange multipliers :math:`u \in \mathbb{R}^m, v \in \mathbb{R}^r`.

.. note::
    This expression may appear complex and difficult to understand at first
    glance. However, we will provide a detailed explanation of how it can be
    utilized to solve the constrained optimization problem presented in Problem :eq:`preknow-eq-1`.

.. tab-set::

    .. tab-item:: Lemma 1
        :sync: key1

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Lemma 1
            ^^^
            At each feasible :math:`x, f(x)=\underset{u \geq 0, v}{\max} L(x, u, v)`,
            and the supremum is taken iff :math:`u \geq 0` satisfying :math:`u_i h_i(x)=0, i=1, \cdots, m`.


    .. tab-item:: Lemma 2
        :sync: key2

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Lemma 2
            ^^^
            The optimal value of the primal problem, named as :math:`f^*`,
            satisfies:

            .. math::



                f^*=\underset{x}{\min} \theta_p(x)=\underset{x}{\min}\underset{u \geq 0, v}{\max}  L(x, u, v)


.. tab-set::

    .. tab-item:: Proof of Lemma 1
        :sync: key1

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Proof of Lemma 1
            ^^^
            Define :math:`\theta_p(x)=\underset{u \geq 0, v}{\max} L(x, u, v)`.
            If :math:`x` is feasible, that means the conditions in Problem
            :eq:`preknow-eq-1` are satisfied. Then we have
            :math:`h_i(x)\le0` and :math:`\ell_j(x)=0`, thus
            :math:`L(x, u, v)=f(x)+\sum_{i=1}^m u_i h_i(x)+\sum_{j=1}^r v_j \ell_j(x)\le f(x)`.
            The last inequality becomes equality iff :math:`u_ih_i(x)=0, i=1,...,m`.
            So, if :math:`x` is feasible, we obtain :math:`f(x)=\theta_p(x)`, where the subscript :math:`p` denotes *primal problem*.

    .. tab-item:: Proof of Lemma 2
      :sync: key2

      .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Proof of Lemma 2
            ^^^
            If :math:`x` is infeasible, we have :math:`h_i(x)>0` or
            :math:`\ell_j(x)\neq0`. Then a quick fact is that
            :math:`\theta_p(x)\rightarrow +\infty` as :math:`u_i\rightarrow +\infty`
            or :math:`v_jh_j(x)\rightarrow +\infty`. So in total, if :math:`f^*`
            violates the constraints, it will not be the optimal value of the primal
            problem. Thus we obtain :math:`f^*=\underset{x}{\min}\theta_p(x)`
            if :math:`f^*` is the optimal value of the primal problem.

Dual Problem
------------

Given a Lagrangian multiplier, we define its Lagrange dual function as:

.. math:: \theta_d(u,v)=\underset{x}{\min} L(x,u,v)

where the subscription :math:`d` denotes the dual problem. It is worth
mentioning that the infimum here does not require :math:`x` to be taken
in the feasible set.

Given the primal problem :eq:`preknow-eq-1`, we
define its Lagrange dual problem as:

.. math::

   \begin{array}{rl}
   \underset{u,v}{\max}& \theta_d(u, v) \\
   \text {s.t.} & u \geq 0
   \end{array}

From the definitions we easily obtain that the optimal value of the dual
problem, named as :math:`g^*`, satisfies:
:math:`g^*=\underset{u\ge0,v}{\max}\underset{x}{\min}L(x,u,v)`.

.. grid:: 2

    .. grid-item::
        :columns: 12 6 6 3

        .. card::
            :class-header: sd-bg-info sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-1

            Lemma3
            ^^^
            The dual problem is a convex optimization problem.

    .. grid-item::
        :columns: 12 6 6 9

        .. card::
            :class-header: sd-bg-info sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-1

            Proof of Lemma 3
            ^^^
            By definition,
            :math:`\theta_d(u,v)=\underset{x}{\min} L(x,u,v)` can be viewed as
            point-wise infimum of affine functions of :math:`u` and :math:`v`, thus
            is concave. :math:`u \geq 0` is affine constraints. Hence dual problem
            is a concave maximization problem, which is a convex optimization
            problem.

Strong and Week Duality
-----------------------

In the previous section, we learned about the definition of primal and dual
problems. You may have noticed that the dual problem has a useful property, it
is convex.

.. note::

    The natural question that arises is whether the solution to the primal
    problem can be obtained by solving the dual problem, since the latter is
    easier to solve.

To answer this question, we need to understand the concepts of weak and strong
duality. These concepts will allow us to establish a connection between the
primal and dual problems.

.. tab-set::

    .. tab-item:: Weak Duality

        .. card::
            :class-header: sd-bg-primary  sd-text-white sd-font-weight-bold
            :class-card:  sd-outline-info  sd-rounded-1

            Introduction to Weak Duality
            ^^^
            The Lagrangian dual problem yields a lower bound for the primal problem.
            It always holds true that :math:`f^*\ge g^*`. We define that as weak
            duality.

            We have the definitions that:

            .. math:: f^*=\underset{x}{\min}\underset{u \geq 0, v}{\max}  L(x, u, v) \quad g^*=\underset{u\ge0,v}{\max}\underset{x}{\min} L(x,u,v)

            Then:

            .. math::

                \begin{aligned}
                    g^*&=\underset{u\ge0,v}{\max}\underset{x}{\min} L(x,u,v)=\underset{x}{\min} L(x,u^*,v^*)\nonumber\\
                    &\le L(x^*,u^*,v^*)\le \underset{u\ge 0,v}{\max} L(x^*,u,v)\nonumber\\
                    &=\underset{x}{\min}\underset{u \geq 0, v}{\max}  L(x, u, v)=f^*\nonumber
                \end{aligned}

            The weak duality is intuitive because it simply takes a small step based
            on the definition. However, it makes little sense for us to solve Problem
            :eq:`preknow-eq-1`, because :math:`f^*\neq g^*`.
            So we will introduce strong duality and luckily, with that we can obtain
            :math:`f^*=g^*`.


    .. tab-item:: Strong Duality

        .. card::
            :class-header: sd-bg-primary  sd-text-white sd-font-weight-bold
            :class-card:  sd-outline-info  sd-rounded-1

            Introduction to Strong Duality
            ^^^
            In some problems, we actually have :math:`f^*=g^*`, which is called
            strong duality. In fact, for convex optimization problems, we nearly
            always have strong duality, only in addition to some slight conditions.
            A most common condition is the Slater's condition.

            If the primal is a convex problem, and there exists at least one
            strictly feasible :math:`\tilde{x}\in \mathbb{R}^n`, satisfying the
            Slater's condition, meaning that:

            .. math:: \exists \tilde{x}, h_i(\tilde{x})<0, i=1, \ldots, m, \ell_j(\tilde{x})=0, j=1, \ldots r

            Then strong duality holds.

Summary
-------

This section introduces the Lagrange method, a powerful tool that allows us to
convert a constrained optimization problem into an unconstrained optimization
problem. In addition, under certain conditions, the solution of a complex
primal problem can be converted to a relatively simple solution of a dual
problem. Safe RL algorithms are essentially solutions to constrained problems,
so understanding the Lagrange method is crucial to understanding many of these
algorithms.

References
----------

- `Convex Optimization <https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf>`__
