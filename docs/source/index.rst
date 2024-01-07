.. OmniSafe documentation master file, created by
    sphinx-quickstart on Fri Nov  4 19:59:00 2022.
    You can adapt this file completely to your liking, but it should at least
    contain the root `toctree` directive.

.. Welcome to OmniSafe's documentation!
.. ====================================

Introduction
============


Welcome To OmniSafe Tutorial
----------------------------

.. image:: image/logo.png
    :scale: 45%

Welcome to `OmniSafe <https://github.com/PKU-Alignment/omnisafe>`_ in Safe RL!
OmniSafe is an infrastructural framework designed to accelerate safe
reinforcement learning (RL) research. It provides a comprehensive and reliable
benchmark for safe RL algorithms, and also an out-of-box modular toolkit for
researchers. Safe RL intends to develop algorithms that minimize the risk of
unintended harm or unsafe behavior.

.. hint::

    **Safe Reinforcement Learning** can be defined as the process of learning policies that maximize the expectation of the return in problems
    in which it is important to ensure reasonable system performance and/or respect safety constraints during the learning and/or deployment processes.

**This tutorial is useful for RL learners of roughly three levels.**

.. grid:: 12 4 4 4


    .. grid-item-card::
        :class-header: sd-bg-success sd-text-white sd-font-weight-bold
        :class-card: sd-outline-success  sd-rounded-1
        :columns: 12 6 6 4

        For Beginners
        ^^^^^^^^^^^^^
        If you have only basic knowledge of linear algebra and probability theory and are new to machine learning, we recommend starting with the mathematical fundamentals section of this tutorial.

    .. grid-item-card::
        :class-header: sd-bg-info sd-text-white sd-font-weight-bold
        :class-card: sd-outline-info  sd-rounded-1
        :columns: 12 6 6 4

        For Average Users
        ^^^^^^^^^^^^^^^^^
        If you have a general understanding of RL algorithms but need to familiarize yourself with Safe RL,
        this tutorial introduces some classic Safe RL algorithms to you so you can get started quickly.

    .. grid-item-card::
        :class-header: sd-bg-primary sd-text-white sd-font-weight-bold
        :class-card: sd-outline-primary  sd-rounded-1
        :columns: 12 6 6 4

        For Experts
        ^^^^^^^^^^^
        If you are already an expert in the field of RL, our tutorial can still offer you new insights with its systematic introduction to Safe RL algorithms. Furthermore, it will enable you to quickly design your own algorithms.


Why We Built This
-----------------

In recent years, `RL`_ (Reinforcement Learning) algorithms,
especially `Deep RL`_ algorithms, have demonstrated remarkable performance in
various tasks.
Notable examples include:

.. hint::
    - Achieving high scores on `Atari <https://atari.com/>`_  using only visual input.
    - Completing complex control tasks in high dimensions.
    - Beating human grandmasters at Go tournaments.

However, in the process of policy updating in reinforcement learning (RL),
agents sometimes learn to **engage in cheating or even dangerous behaviors** in
order to improve their performance. While these agents may achieve high scores
rapidly, their actions may not align with the desired outcome.

Therefore, Safe RL algorithms aim to train agents to achieve desired goals
while adhering to constraints and safety requirements, addressing the challenge
of maintaining agent safety during the policy updating process in RL. The
primary objective of safe RL algorithms is to ensure that agents maintain
safety and avoid behaviors that could lead to negative consequences or violate
predefined constraints.

.. admonition:: However
    :class: warning

    Even experienced researchers in RL may face challenges when it comes to
    quickly grasping the intricacies of Safe RL algorithms and efficiently
    programming their implementations.

To address this issue, OmniSafe aims to provide a comprehensive and systematic
introduction to Safe RL algorithms, along with streamlined and robust code,
making it easier for researchers to delve into Safe RL.

.. tab-set::

    .. tab-item:: Problem I
        :sync: key1

        .. card::
            :class-header: sd-bg-warning  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-warning  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Puzzling Math
            ^^^^^^^^^^^^^
            Safe RL algorithms belong to a class of rigorously grounded
            algorithms that are built upon a strong mathematical foundation.
            While these algorithms possess detailed theoretical derivations,
            their lack of a unified symbolic system can pose challenges for
            beginners in terms of systematic and comprehensive learning.

    .. tab-item:: Problem II
        :sync: key2

        .. card::
            :class-header: sd-bg-warning sd-text-white sd-font-weight-bold
            :class-card: sd-outline-warning  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Hard-to-find Codes
            ^^^^^^^^^^^^^^^^^^
            Most of the existing Safe RL algorithms do not have **open-source**
            code available, which makes it difficult for beginners to
            understand the algorithms at the code level. Furthermore,
            researchers may encounter issues such as incorrect implementations,
            unfair comparisons, and misleading conclusions, which could have
            been avoided with open-source code.

.. tab-set::

    .. tab-item:: Soulution I
        :sync: key1

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Friendly Math
            ^^^^^^^^^^^^^
            The OmniSafe tutorial offers a **standardized notation system** that
            enables beginners to acquire a complete and systematic
            understanding of the theory behind Safe RL algorithms.

    .. tab-item:: Solution II
        :sync: key2

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Robust Code
            ^^^^^^^^^^^
            The OmniSafe tutorial provides **a comprehensive introduction** to
            each algorithm, including a detailed explanation of the code
            implementation. Beginners can easily understand the connection
            between the algorithmic concepts and the code, while experts can
            gain valuable insights into Safe RL by studying the code-level
            details of each algorithm.


.. _`RL`: https://en.wikipedia.org/wiki/Reinforcement_learning
.. _`Deep RL`: http://ufldl.stanford.edu/tutorial/

Code Design Principles
----------------------

.. grid:: 12 4 4 4
    :gutter: 1


    .. grid-item-card::
        :class-header: sd-bg-success sd-text-white sd-font-weight-bold
        :class-card: sd-outline-success  sd-rounded-1
        :columns: 12 5 5 4

        Consistent and Inherited
        ^^^^^^^^^^^^^^^^^^^^^^^^
        Our code follows a comprehensive and logical system, enabling users to
        understand the interconnection between each algorithm. For instance, if
        one comprehends the Policy Gradient algorithm, they can quickly grasp
        the code implementation of the PPO algorithm by reading a new function.

    .. grid-item-card::
        :class-header: sd-bg-info sd-text-white sd-font-weight-bold
        :class-card: sd-outline-info  sd-rounded-1
        :columns: 12 5 5 4

        Robust and Readable
        ^^^^^^^^^^^^^^^^^^^^
        Our code not only serves as a tutorial but also as a practical tool.
        For those who want to learn about the implementation of Safe RL
        algorithms, the highly readable code in OmniSafe provides an easy and
        quick way to get started. For those who
        want to develop their algorithms, OmniSafe's **highly modular and
        reusable** code can be an excellent resource.

    .. grid-item-card::
        :class-header: sd-bg-primary sd-text-white sd-font-weight-bold
        :class-card: sd-outline-primary  sd-rounded-1
        :columns: 12 5 5 4

        Long-lived
        ^^^^^^^^^^^^^^^^^^^^^^^^^^
        Unlike other codes that heavily rely on external libraries, OmniSafe
        minimizes its dependency on third-party libraries. This design
        prevents the project from becoming obsolete due to changes in the
        third-party library code, and optimizes the user experience by reducing
        the number of dependencies that need to be installed to run OmniSafe.

Before Reading
--------------

Before you start reading the OmniSafe tutorial, we want you to
understand the usage of colors in this tutorial.

In this tutorial, in general, the :bdg-info:`light blue boxes` indicate
mathematically relevant derivations, including but not limited to
:bdg-info-line:`Theorem`, :bdg-info-line:`Lemma`, :bdg-info-line:`Proposition`,
:bdg-info-line:`Corollary`, and :bdg-info-line:`their proofs`, while the
:bdg-success:`green boxes` indicate specifically
:bdg-success-line:`implementations`, both :bdg-success-line:`theoretical` and
:bdg-success-line:`code-based`.
We give an example below:

.. dropdown:: Example of OmniSafe color usage styles (Click here)
    :animate: fade-in-slide-down
    :class-title: sd-font-weight-bold sd-outline-primary sd-text-primary
    :class-body: sd-font-weight-bold

    .. card::
        :class-header: sd-bg-info sd-text-white sd-font-weight-bold
        :class-card: sd-outline-info  sd-rounded-1
        :class-footer: sd-font-weight-bold
        :link: cards-clickable
        :link-type: ref

        Theorem I (Difference between two arbitrary policies)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        **For any function** :math:`f : S \rightarrow \mathbb{R}` and any policies :math:`\pi` and :math:`\pi'`, define :math:`\delta_f(s,a,s') \doteq R(s,a,s') + \gamma f(s')-f(s)`,

        .. math::
            :nowrap:

            \begin{eqnarray}
                \epsilon_f^{\pi'} &\doteq& \max_s \left|\mathbb{E}_{a\sim\pi'~,s'\sim P }\left[\delta_f(s,a,s')\right] \right|\tag{3}\\
                L_{\pi, f}\left(\pi'\right) &\doteq& \mathbb{E}_{\tau \sim \pi}\left[\left(\frac{\pi'(a | s)}{\pi(a|s)}-1\right)\delta_f\left(s, a, s'\right)\right]\tag{4} \\
                D_{\pi, f}^{\pm}\left(\pi^{\prime}\right) &\doteq& \frac{L_{\pi, f}\left(\pi' \right)}{1-\gamma} \pm \frac{2 \gamma \epsilon_f^{\pi'}}{(1-\gamma)^2} \mathbb{E}_{s \sim d^\pi}\left[D_{T V}\left(\pi^{\prime} \| \pi\right)[s]\right]\tag{5}
            \end{eqnarray}

        where :math:`D_{T V}\left(\pi'|| \pi\right)[s]=\frac{1}{2} \sum_a\left|\pi'(a|s)-\pi(a|s)\right|` is the total variational divergence between action distributions at :math:`s`. The conclusion is as follows:

        .. math:: D_{\pi, f}^{+}\left(\pi'\right) \geq J\left(\pi'\right)-J(\pi) \geq D_{\pi, f}^{-}\left(\pi'\right)\tag{6}

        Furthermore, the bounds are tight (when :math:`\pi=\pi^{\prime}`, all three expressions are identically zero).

        The proof of the :bdg-ref-info-line:`Theorem 1<Theorem 1>` can be seen in the :bdg-ref-info:`Appendix`, click on this :bdg-info-line:`card` to jump to view.

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

                    env = omnisafe.Env('SafetyPointGoal1-v0')

                    agent = omnisafe.Agent('CPO', env)
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

                    We use ``train_policy.py`` as the entrance file. You can train the agent with
                    CPO simply using ``train_policy.py``, with arguments about CPO and environments
                    does the training. For example, to run CPO in SafetyPointGoal1-v0 , with
                    1 torch thread and seed 0, you can use the following command:

                    .. code-block:: bash
                        :linenos:

                        cd examples
                        python train_policy.py --algo CPO --env-id SafetyPointGoal1-v0 --parallel 1 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 1

You may not yet understand the above theory and the specific meaning of the
code, but do not worry, we will make a detailed introduction later in the
:doc:`../saferl/cpo` tutorial.

Citing OmniSafe
---------------

If you find OmniSafe useful or use OmniSafe in your research, please cite it in your publications.

.. code-block:: bibtex

    @article{omnisafe,
      title   = {OmniSafe: An Infrastructure for Accelerating Safe Reinforcement Learning Research},
      author  = {Jiaming Ji, Jiayi Zhou, Borong Zhang, Juntao Dai, Xuehai Pan, Ruiyang Sun, Weidong Huang, Yiran Geng, Mickel Liu, Yaodong Yang},
      journal = {arXiv preprint arXiv:2305.09304},
      year    = {2023}
    }

Long-Term Support and Support History
-------------------------------------

**OmniSafe** is mainly developed by the Safe RL research team directed by `Prof. Yaodong Yang <https://github.com/PKU-YYang>`_.
Our Safe RL research team members include `Borong Zhang <https://github.com/muchvo>`_ , `Jiayi Zhou <https://github.com/Gaiejj>`_, `JTao Dai <https://github.com/calico-1226>`_,  `Weidong Huang <https://github.com/hdadong>`_, `Ruiyang Sun <https://github.com/rockmagma02>`_, `Xuehai Pan <https://github.com/XuehaiPan>`_ and `Jiaming Ji <https://github.com/zmsn-2077>`_.
If you have any questions in the process of using OmniSafe, or if you are
willing to contribute to
this project, don't hesitate to ask your question on `the GitHub issue page <https://github.com/PKU-Alignment/omnisafe/issues/new/choose>`_, we will reply to you in 2-3 working days.

------

.. toctree::
    :hidden:
    :maxdepth: 3
    :caption: get started

    start/installation
    start/usage

.. toctree::
    :hidden:
    :maxdepth: 3
    :caption: mathematical theory

    foundations/notations
    foundations/vector
    foundations/lagrange

.. toctree::
    :hidden:
    :maxdepth: 3
    :caption: base rl algorithms

    baserl/trpo
    baserl/ppo

.. toctree::
    :hidden:
    :maxdepth: 3
    :caption: safe rl algorithms

    saferl/cpo
    saferl/pcpo
    saferl/focops
    saferl/lag

.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: base rl algorithms api

    baserlapi/on_policy
    baserlapi/off_policy
    baserlapi/model_based

.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: safe rl algorithms api

    saferlapi/first_order
    saferlapi/second_order
    saferlapi/lagrange
    saferlapi/penalty_function
    saferlapi/model_based

.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: common api

    common/buffer
    common/exp_grid
    common/lagrange
    common/normalizer
    common/logger
    common/simmer_agent
    common/stastics_tool
    common/offline_data

.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: utils api

    utils/config
    utils/distributed
    utils/math
    utils/model
    utils/tools
    utils/plotter

.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: models api

    model/actor
    model/critic
    model/actor_critic
    model/modelbased_model
    model/modelbased_planner
    model/offline

.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: envs api

    envs/core
    envs/wrapper
    envs/safety_gymnasium
    envs/mujoco_env
    envs/discrete_env
    envs/adapter


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
