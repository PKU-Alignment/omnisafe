Basic Mathematical Theory
=========================



Introduction
------------
`Reinforcement Learning <https://static.hlt.bme.hu/semantics/external/pages/deep_learning/en.wikipedia.org/wiki/Reinforcement_learning.html#:~:text=Reinforcement%20learning%20%28RL%29%20is%20an%20area%20of%20machine,as%20to%20maximize%20some%20notion%20of%20cumulative%20reward.>`__
is one of the disciplines in machine learning that is more closely related to mathematics.
**Safe Reinforcement Learning** is particularly close to mathematical theory,
especially **Optimization Theory**.

This section introduces the basic mathematical theory of Safe Reinforcement Learning.
We will briefly introduce the following topics: Linear Algebra and Optimization Theory.

If you are new to these mathematical theories in subsequent chapters, please refer back to this article.
If this still does not solve your confusion, please refer to the more detailed introduction to mathematical theory.

Knowledge of Vector and Matrix
------------------------------

Vector Projection
~~~~~~~~~~~~~~~~~

The projection of a vector :math:`\boldsymbol{y} \in \mathbb{R}^m` onto the span
of :math:`\left\{\boldsymbol{x}_1, \ldots, \boldsymbol{x}_n\right\}` (here we assume
:math:`\boldsymbol{x}_i \in \mathbb{R}^m` )is the vector
:math:`\boldsymbol{v} \in \operatorname{span}\left(\left\{\boldsymbol{x}_1, \ldots, \boldsymbol{x}_n\right\}\right)`,
such that :math:`\boldsymbol{v}` is as close as possible to :math:`\boldsymbol{y}`, as
measured by the Euclidean norm :math:`\|\boldsymbol{v}-\boldsymbol{y}\|_2`. We denote
the projection as
:math:`\operatorname{Proj}\left(\boldsymbol{y} ;\left\{\boldsymbol{x}_1, \ldots, \boldsymbol{x}_n\right\}\right)`
and can define it formally as

.. math:: \operatorname{Proj}\left(\boldsymbol{y} ;\left\{\boldsymbol{x}_1, \ldots, \boldsymbol{x}_n\right\}\right)=\mathop{\arg\min}\limits_{\boldsymbol{v} \in \operatorname{span}\left(\left\{\boldsymbol{x}_1, \ldots, \boldsymbol{x}_n\right\}\right)}\|\boldsymbol{y}-\boldsymbol{v}\|_2 .

Given a full rank matrix :math:`\mathbf{A} \in \mathbb{R}^{m \times n}`
with :math:`m \geq n`, we can define the projection of a vector
:math:`\boldsymbol{y} \in \mathbb{R}^m` onto the range of :math:`\mathbf{A}` as
follows:

.. math:: \operatorname{Proj}(\boldsymbol{y} ; \mathbf{A})=\mathop{\arg\min}\limits_{\boldsymbol{v} \in \mathcal{R}(\mathbf{A})}\|\boldsymbol{v}-\boldsymbol{y}\|_2=\mathbf{A}\left(\mathbf{A}^{\top} \mathbf{A}\right)^{-1} \mathbf{A}^{\top} \boldsymbol{y} .


Norms
~~~~~

.. tab-set::

    .. tab-item:: Vector Norm

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-3
            :class-footer: sd-font-weight-bold

            Introduction of Vector Norm
            ^^^
            A norm of a vector :math:`\Vert\boldsymbol{x}\Vert` is a measure of the "length"
            of the vector. More formally, a norm is any function
            :math:`f: \mathbb{R}^n \rightarrow \mathbb{R}` that satisfies four
            properties:

            #. For all :math:`\boldsymbol{x} \in \mathbb{R}^n, f(\boldsymbol{x}) \geq 0`
               (non-negativity).

            #. :math:`f(\boldsymbol{x})=0` if and only if :math:`\boldsymbol{x}=\mathbf{0}`
               (definiteness).

            #. For all
               :math:`\boldsymbol{x} \in \mathbb{R}^n, t \in \mathbb{R}, f(t \boldsymbol{x})=|t| f(x)`
               (absolute value homogeneity).

            #. For all
               :math:`\boldsymbol{x}, \boldsymbol{y} \in \mathbb{R}^n, f(\boldsymbol{x}+\boldsymbol{y}) \leq f(\boldsymbol{x})+f(\boldsymbol{y})`
               (triangle inequality).

            Consider the following common examples:

            -  **p-norm:**
               :math:`\|\boldsymbol{x}\|_p=\left(\sum_{i=1}^n\left|x_i\right|^p\right)^{1 / p}`,
               for :math:`p \geq 1`.

            -  **2-norm:** :math:`\|\boldsymbol{x}\|_2=\sqrt{\sum_{i=1}^n x_i^2}`, also
               called Euclidean norm. Note that
               :math:`\|\boldsymbol{x}\|_2^2=\boldsymbol{x}^{\top} \boldsymbol{x}`.

            -  **1-norm:** :math:`\|\boldsymbol{x}\|_1=\sum_{i=1}^n\left|x_i\right|`.

            -  **Max-norm:** :math:`\|x\|_{\infty}=\max _i\left|x_i\right|`.

            -  **0-norm:**
               :math:`\|\boldsymbol{x}\|_0=\sum_{i=1}^n \mathbb{I}\left(\left|x_i\right|>0\right)`.
               This is a pseudo-norm, since it does not satisfy homogeneity. It
               counts the number of non-zero elements in :math:`\boldsymbol{x}`. If we
               define :math:`0^0=0`, we can write this as
               :math:`\|\boldsymbol{x}\|_0=\sum_{i=1}^n x_i^0`

    .. tab-item:: Matrix Norm

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card:  sd-outline-info  sd-rounded-3
            :class-footer: sd-font-weight-bold

            Introduction of Matrix Norm
            ^^^
            Suppose we think of a matrix
            :math:`\mathbf{A} \in \mathbb{R}^{m \times n}` as defining a linear
            function :math:`f(\boldsymbol{x})=\mathbf{A} \boldsymbol{x}`. We define the induced norm
            of :math:`\mathbf{A}` as the maximum amount by which :math:`f` can
            lengthen any unit-norm input:

            .. math:: \|\mathbf{A}\|_p=\max _{\boldsymbol{x} \neq 0} \frac{\|\mathbf{A} \boldsymbol{x}\|_p}{\|\boldsymbol{x}\|_p}=\max _{\|\boldsymbol{x}\|=1}\|\mathbf{A} \boldsymbol{x}\|_p

            Typically :math:`p=2`, in which case

            .. math:: \|\mathbf{A}\|_2=\sqrt{\lambda_{\max }\left(\mathbf{A}^{\top} \mathbf{A}\right)}=\max _i \sigma_i

            where :math:`\sigma_i` is the :math:`i^{th}`  singular value. The nuclear
            norm, also called the trace norm, is defined as

            .. math:: \|\mathbf{A}\|_*=\operatorname{tr}\left(\sqrt{\mathbf{A}^{\top} \mathbf{A}}\right)=\sum_i \sigma_i

            where :math:`\sqrt{\mathbf{A}^{\top} \mathbf{A}}` is the matrix square
            root. Since the singular values are always non-negative, we have

            .. math:: \|\mathbf{A}\|_*=\sum_i\left|\sigma_i\right|=\|\boldsymbol{\sigma}\|_1

            Using this as a regularizer encourages many singular values to become
            zero, resulting in a low rank matrix. More generally, we can define the
            Schatten :math:`p`-norm as

            .. math:: \|\mathbf{A}\|_p=\left(\sum_i \sigma_i^p(\mathbf{A})\right)^{1 / p}

            If we think of a matrix as a vector, we can define the matrix norm in
            terms of a vector norm,
            :math:`\|\mathbf{A}\|=\|\operatorname{vec}(\mathbf{A})\|`. If the vector
            norm is the 2-norm, the corresponding matrix norm is the Frobenius norm:

            .. math:: \|\mathbf{A}\|_F=\sqrt{\sum_{i=1}^m \sum_{j=1}^n a_{i j}^2}=\sqrt{\operatorname{tr}\left(\mathbf{A}^{\top} \mathbf{A}\right)}=\|\operatorname{vec}(\mathbf{A})\|_2

            If :math:`\mathbf{A}` is expensive to evaluate, but
            :math:`\mathbf{A} \boldsymbol{v}` is cheap (for a random vector :math:`\boldsymbol{v}`
            ), we can create a stochastic approximation to the Frobenius norm by
            using the Hutchinson trace estimator as follows:

            .. math:: \|\mathbf{A}\|_F^2=\operatorname{tr}\left(\mathbf{A}^{\top} \mathbf{A}\right)=\mathbb{E}\left[\boldsymbol{v}^{\top} \mathbf{A}^{\top} \mathbf{A} \boldsymbol{v}\right]=\mathbb{E}\left[\|\mathbf{A} \boldsymbol{v}\|_2^2\right]

            where :math:`\boldsymbol{v} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})`.

Lagrange Duality
----------------

.. _`lagrange_theorem`:

Primal Problem
~~~~~~~~~~~~~~

Consider a general optimization problem (called as the primal problem):

.. _preknow-eq-1:

.. math::
    :nowrap:

    \begin{eqnarray}
        \underset{x}{\text{min}} && f(x)\tag{1} \\
        \text { s.t. } && h_i(x) \leq 0, i=1, \cdots, m \\
        && \ell_j(x)=0, j=1, \cdots, r
    \end{eqnarray}

We define its Lagrangian as:

.. math:: L(x, u, v)=f(x)+\sum_{i=1}^m u_i h_i(x)+\sum_{j=1}^r v_j \ell_j(x)\tag{2}

Lagrange multipliers :math:`u \in \mathbb{R}^m, v \in \mathbb{R}^r`.

.. note::

    This expression may be so complex that you won't immediately understand
    what it means. Don't worry; we'll explain how it can be used to solve the constrained optimization problem in Problem :ref:`(1) <preknow-eq-1>`.

.. tab-set::

    .. tab-item:: Lemma 1
        :sync: key1

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-3
            :class-footer: sd-font-weight-bold

            Lemma 1
            ^^^
            At each feasible :math:`x, f(x)=\underset{u \geq 0, v}{\max} L(x, u, v)`,
            and the supremum is taken iff :math:`u \geq 0` satisfying :math:`u_i h_i(x)=0, i=1, \cdots, m`.


    .. tab-item:: Lemma 2
        :sync: key2

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-3
            :class-footer: sd-font-weight-bold

            Lemma 2
            ^^^
            The optimal value of the primal problem, named as :math:`f^*`,
            satisfies:

            .. math::
                :nowrap:

                \begin{eqnarray}
                f^*=\underset{x}{\text{min}}\quad \theta_p(x)=\underset{x}{\text{min}}\underset{u \geq 0, v}{\max} \quad L(x, u, v)
                \end{eqnarray}

.. tab-set::

    .. tab-item:: Proof of Lemma 1
        :sync: key1

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-3
            :class-footer: sd-font-weight-bold

            Proof of Lemma 1
            ^^^
            Define :math:`\theta_p(x)=\underset{u \geq 0, v}{\max} L(x, u, v)`.
            If :math:`x` is feasible, that means the conditions in Problem
            :ref:`(1) <preknow-eq-1>` are satisfied. Then we have
            :math:`h_i(x)\le0` and :math:`\ell_j(x)=0`, thus
            :math:`L(x, u, v)=f(x)+\sum_{i=1}^m u_i h_i(x)+\sum_{j=1}^r v_j \ell_j(x)\le f(x)`.
            The last inequality becomes equality iff :math:`u_ih_i(x)=0, i=1,...,m`.
            So, if :math:`x` is feasible, we obtain :math:`f(x)=\theta_p(x)`, where
            the subscript :math:`p` denotes *primal problem*.

    .. tab-item:: Proof of Lemma 2
      :sync: key2

      .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-3
            :class-footer: sd-font-weight-bold

            Proof of Lemma 2
            ^^^
            If :math:`x` is infeasible, we have :math:`h_i(x)>0` or
            :math:`\ell_j(x)\neq0`. Then a quick fact is that
            :math:`\theta_p(x)\rightarrow +\infty` as :math:`u_i\rightarrow +\infty`
            or :math:`v_jh_j(x)\rightarrow +\infty`. So in total, if :math:`f^*`
            violates the constraints, it will not be the optimal value of the primal
            problem. Thus we obtain :math:`f^*=\underset{x}{\text{min}}\quad \theta_p(x)`
            if :math:`f^*` is the optimal value of the primal problem.

Dual Problem
~~~~~~~~~~~~

Given a Lagrangian, we define its Lagrange dual function as:

.. math:: \theta_d(u,v)=\underset{x}{\text{min}}\quad L(x,u,v)

where the subscription :math:`d` denotes the dual problem. It is worth
mentioning that the infimum here does not require :math:`x` to be taken
in the feasible set.

Given the primal problem :ref:`(1) <preknow-eq-1>`, we
define its Lagrange dual problem as:

.. math::

   \begin{array}{rl}
   \underset{u,v}{\max}& \theta_d(u, v) \\
   \text { s.t. } & u \geq 0
   \end{array}\nonumber

From the definitions we easily obtain that the optimal value of the dual
problem, named as :math:`g^*`, satisfies:
:math:`g^*=\underset{u\ge0,v}{\text{max}}\underset{x}{\text{min}}\quad L(x,u,v)`.

.. grid:: 2

    .. grid-item::
        :columns: 12 6 6 3

        .. card::
            :class-header: sd-bg-info sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-3

            Lemma3
            ^^^
            The dual problem is a convex optimization problem.

    .. grid-item::
        :columns: 12 6 6 9

        .. card::
            :class-header: sd-bg-info sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-3

            Proof of Lemma 3
            ^^^
            By definition,
            :math:`\theta_d(u,v)=\underset{x}{\text{min}}\quad L(x,u,v)` can be viewed as
            point-wise infimum of affine functions of :math:`u` and :math:`v`, thus
            is concave. :math:`u \geq 0` is affine constraints. Hence dual problem
            is a concave maximization problem, which is a convex optimization
            problem.

Strong and Week Duality
~~~~~~~~~~~~~~~~~~~~~~~

In the above introduction, we learned about the definition of primal and dual problems. You may find that the dual problem has a suitable property,
that the dual problem is convex.

.. note::

    The naive idea is that since the dual problem is convex,
    that is, convenient to solve, can the solution of the primal problem be converted to the solution of the dual problem?

We will discuss the weak and strong duality to show you the connection between the primal and dual problems.

.. tab-set::

    .. tab-item:: Weak Duality

        .. card::
            :class-header: sd-bg-primary  sd-text-white sd-font-weight-bold
            :class-card:  sd-outline-info  sd-rounded-3

            Introduction to Weak Duality
            ^^^
            The Lagrangian dual problem yields a lower bound for the primal problem.
            It always holds true that :math:`f^*\ge g^*`. We define that as weak
            duality. *Proof.* We have the definitions that:

            .. math:: f^*=\underset{x}{\text{min}}\underset{u \geq 0, v}{\max} \quad L(x, u, v) \quad g^*=\underset{u\ge0,v}{\text{max}}\underset{x}{\text{min}}\quad L(x,u,v)

            Then:

            .. math::

                \begin{aligned}
                    g^*&=\underset{u\ge0,v}{\text{max}}\underset{x}{\text{min}}\quad L(x,u,v)=\underset{x}{\text{min}}\quad L(x,u^*,v^*)\nonumber\\
                    &\le L(x^*,u^*,v^*)\le \underset{u\ge 0,v}{\text{max}}\quad L(x^*,u,v)\nonumber\\
                    &=\underset{x}{\text{min}}\underset{u \geq 0, v}{\max} \quad L(x, u, v)=f^*\nonumber
                \end{aligned}

            The weak duality is intuitive because it simply takes a small step based
            on the definition. However, it make little sense for us to solve Problem
            :ref:`(1) <preknow-eq-1>`, because :math:`f^*\neq g^*`.
            So we will introduce strong duality and luckily, with that we can obtain
            :math:`f^*=g^*`.


    .. tab-item:: Strong Duality

        .. card::
            :class-header: sd-bg-primary  sd-text-white sd-font-weight-bold
            :class-card:  sd-outline-info  sd-rounded-3

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
~~~~~~~

In this section we introduce you to the Lagrange method, which converts
the solution of a constrained optimization problem into a solution to an
unconstrained optimization problem. We also introduce that under certain
conditions, the solution of a complex primal problem can also be
converted to a relatively simple solution of a dual problem. SafeRL's
algorithms are essentially solutions to constrained problems, so the
Lagrange method is an important basis for many of these algorithms.
