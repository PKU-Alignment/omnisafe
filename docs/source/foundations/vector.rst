Vector and Martrix
==================

Vector Projection
-----------------

The projection of a vector :math:`\boldsymbol{y} \in \mathbb{R}^m` onto the
span of :math:`\left\{\boldsymbol{x}_1, \ldots, \boldsymbol{x}_n\right\}` (here
we assume :math:`\boldsymbol{x}_i \in \mathbb{R}^m` and denote
:math:`\operatorname{span}(\{\boldsymbol{x}_1,
\ldots, \boldsymbol{x}_n\})` as  :math:`\boldsymbol{X}` ) is the vector
:math:`\boldsymbol{v} \in \boldsymbol{X}`,
such that :math:`\boldsymbol{v}` is as close as possible to :math:`\boldsymbol
{y}`, as measured by the Euclidean norm
:math:`\|\boldsymbol{v}-\boldsymbol{y}\|_2`. We denote the projection as
:math:`\operatorname{Proj}\left(\boldsymbol {y} ;
\left\{\boldsymbol{x}_1, \ldots, \boldsymbol{x}_n\right\}\right)`
and can define it formally as

.. math:: \operatorname{Proj}(\boldsymbol{y} ;\{\boldsymbol{x}_1, \ldots, \boldsymbol{x}_n\})=\mathop{\arg\min}\limits_{\boldsymbol{v} \in \boldsymbol{X}}\|\boldsymbol{y}-\boldsymbol{v}\|_2

Given a full rank matrix :math:`\mathbf{A} \in \mathbb{R}^{m \times n}`
with :math:`m \geq n`, we can define the projection of a vector
:math:`\boldsymbol{y} \in \mathbb{R}^m` onto the range of :math:`\mathbf{A}` as
follows:

.. math:: \operatorname{Proj}(\boldsymbol{y} ; \mathbf{A})=\mathop{\arg\min}\limits_{\boldsymbol{v} \in \mathcal{R}(\mathbf{A})}\|\boldsymbol{v}-\boldsymbol{y}\|_2=\mathbf{A}\left(\mathbf{A}^{\top} \mathbf{A}\right)^{-1} \mathbf{A}^{\top} \boldsymbol{y}


Norms
-----

.. tab-set::

    .. tab-item:: Vector Norm

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-1
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

            -  **Max-norm:** :math:`\|x\|_{\infty}=\max _i\left|x_i\right|`.

            -  **2-norm:** :math:`\|\boldsymbol{x}\|_2=\sqrt{\sum_{i=1}^n x_i^2}`, also
               called Euclidean norm. Note that
               :math:`\|\boldsymbol{x}\|_2^2=\boldsymbol{x}^{\top} \boldsymbol{x}`.

            -  **1-norm:** :math:`\|\boldsymbol{x}\|_1=\sum_{i=1}^n\left|x_i\right|`.

            -  **0-norm:**
               :math:`\|\boldsymbol{x}\|_0=\sum_{i=1}^n \mathbb{I}\left(\left|x_i\right|>0\right)`.
               This is a pseudo-norm since it does not satisfy homogeneity. It
               counts the number of non-zero elements in :math:`\boldsymbol{x}`. If we
               define :math:`0^0=0`, we can write this as
               :math:`\|\boldsymbol{x}\|_0=\sum_{i=1}^n x_i^0`

    .. tab-item:: Matrix Norm

        .. card::
            :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
            :class-card:  sd-outline-info  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Introduction of Matrix Norm
            ^^^
            Suppose a matrix
            :math:`\mathbf{A} \in \mathbb{R}^{m \times n}` as defining a linear
            function :math:`f(\boldsymbol{x})=\mathbf{A} \boldsymbol{x}`. We define the induced norm
            of :math:`\mathbf{A}` as the maximum amount by which :math:`f` can
            lengthen any unit-norm input:

            .. math:: \|\mathbf{A}\|_p=\max _{\boldsymbol{x} \neq 0} \frac{\|\mathbf{A} \boldsymbol{x}\|_p}{\|\boldsymbol{x}\|_p}=\max _{\|\boldsymbol{x}\|=1}\|\mathbf{A} \boldsymbol{x}\|_p

            Typically :math:`p=2`, in which case

            .. math:: \|\mathbf{A}\|_2=\sqrt{\lambda_{\max }\left(\mathbf{A}^{\top} \mathbf{A}\right)}=\max _i \sigma_i

            where :math:`\sigma_i` is the :math:`i^{th}`  singular value. The Nuclear
            norm also called the trace norm, is defined as

            .. math:: \|\mathbf{A}\|_*=\operatorname{tr}\left(\sqrt{\mathbf{A}^{\top} \mathbf{A}}\right)=\sum_i \sigma_i

            where :math:`\sqrt{\mathbf{A}^{\top} \mathbf{A}}` is the matrix square
            root. Since the singular values are always non-negative, we have

            .. math:: \|\mathbf{A}\|_*=\sum_i\left|\sigma_i\right|=\|\boldsymbol{\sigma}\|_1

            Using this as a regularizer encourages many singular values to become
            zero, resulting in a low-rank matrix. More generally, we can define the
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
