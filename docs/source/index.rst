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

Welcome to `OmniSafe <https://jmlr.org/papers/v16/garcia15a.html>`__ in Safe RL!
OmniSafe is a comprehensive and reliable benchmark for safe reinforcement learning, encompassing more than 20 different kinds of algorithms covering a multitude of SafeRL domains and delivering a new suite of testing environments.

.. hint::

    For beginners, it is necessary first to introduce you to Safe RL(Safe Reinforcement Learning).
    *Safe Reinforcement Learning* can be defined as the learning agents that maximize the expectation of the return on problems, ensure reasonable system performance, and respect safety constraints during the learning and deployment processes.

**This tutorial is useful for reinforcement learning learners of many levels.**

.. grid:: 12 4 4 4


    .. grid-item-card::
        :class-item: sd-font-weight-bold
        :class-header: sd-bg-success sd-text-white sd-font-weight-bold
        :class-card: sd-outline-success  sd-rounded-1
        :columns: 12 6 6 4

        For Beginners
        ^^^^^^^^^^^^^
        If you are a beginner in machine learning with only some simple knowledge of linear algebra and probability theory, you can start with the mathematical fundamentals section of this tutorial.

    .. grid-item-card::
        :class-item: sd-font-weight-bold
        :class-header: sd-bg-info sd-text-white sd-font-weight-bold
        :class-card: sd-outline-info  sd-rounded-1
        :columns: 12 6 6 4

        For Average Users
        ^^^^^^^^^^^^^^^^^
        If you have a general understanding of RL algorithms but need to familiarize yourself with Safe RL, This tutorial introduces it so you can get started quickly.

    .. grid-item-card::
        :class-item: sd-font-weight-bold
        :class-header: sd-bg-primary sd-text-white sd-font-weight-bold
        :class-card: sd-outline-primary  sd-rounded-1
        :columns: 12 6 6 4

        For Master
        ^^^^^^^^^^
        If you are already an expert in the field of RL, you can also gain new insights from our systematic introduction to Safe RL algorithms.
        Also, this tutorial will allow you to design your algorithms using OmniSafe quickly.


Why We Built This
-----------------

In recent years, `RL`_ (Reinforcement Learning) algorithms, especially `Deep RL`_ algorithms, have performed well in many tasks.
Examples include:

- Achieving high scores on Atari games with only visual input.
- Completing complex control tasks in high dimensions.
- Beating human grandmasters at Go tournaments.

However, in the process of strategy updating by RL, the agents often learn **cheating or even dangerous behaviors** to improve their performance.
Such an agent that can quickly achieve high scores differs from our desired result.
Therefore, Safe RL algorithms are dedicated to solving the problem of how to train an agent to learn to achieve the desired training goal without violating constraints simultaneously.

.. admonition:: However
    :class: warning

    Even experienced RL researchers have difficulty understanding Safe RL's algorithms in a short time and quickly programming their implementation.

Therefore, OmniSafe will facilitate the subsequent study of Safe RL by providing both a **detailed and systematic introduction to the algorithm** and a **streamlined and robust code**.

.. tab-set::

    .. tab-item:: Problem I
        :sync: key1

        .. card::
            :class-header: sd-bg-warning  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-warning  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Puzzling Math
            ^^^^^^^^^^^^^
            Safe RL algorithms are a class of algorithms built on a rigorous mathematical system.
            These algorithms have a detailed theoretical derivation, but they lack a unified symbolic system, which makes it difficult for beginners to learn them systematically and comprehensively.

    .. tab-item:: Problem II
        :sync: key2

        .. card::
            :class-header: sd-bg-warning sd-text-white sd-font-weight-bold
            :class-card: sd-outline-warning  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Hard-to-find Codes
            ^^^^^^^^^^^^^^^^^^
            Most of the existing Safe RL algorithms **do not have open-source code**, making it difficult for beginners to grasp the ideas of the algorithms at the code level, and researchers suffer from incorrect implementations, unfair comparisons, and misleading conclusions.

.. tab-set::

    .. tab-item:: Soulution I
        :sync: key1

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Friendly Math
            ^^^^^^^^^^^^^
            OmniSafe tutorial provides a **unified and standardized notation system** that allows beginners to learn the theory of Safe RL algorithms completely and systematically.

    .. tab-item:: Solution II
        :sync: key2

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Robust Code
            ^^^^^^^^^^^
            OmniSafe tutorial gives a **code-level** introduction in each algorithm introduction, allowing learners who are new to Safe RL theory to understand how to relate algorithmic ideas to code and give experts in the field of Safe RL new insights into algorithm implementation.



.. _`RL`: https://en.wikipedia.org/wiki/Reinforcement_learning
.. _`Deep RL`: http://ufldl.stanford.edu/tutorial/

Code Design Principles
----------------------

.. grid:: 12 4 4 4
    :gutter: 1


    .. grid-item-card::
        :class-item: sd-font-weight-bold
        :class-header: sd-bg-success sd-text-white sd-font-weight-bold
        :class-card: sd-outline-success  sd-rounded-1
        :columns: 12 5 5 4

        Consistent and Inherited
        ^^^^^^^^^^^^^^^^^^^^^^^^
        Our code has a complete logic system that allows you to understand the connection between each algorithm and the similarities together with differences.
        For example, if you understand the Policy Gradient algorithm, then you can learn the PPO algorithm by simply reading a new function and immediately grasping the code implementation of the PPO algorithm.

    .. grid-item-card::
        :class-item: sd-font-weight-bold
        :class-header: sd-bg-info sd-text-white sd-font-weight-bold
        :class-card: sd-outline-info  sd-rounded-1
        :columns: 12 5 5 4

        Robust and Readable
        ^^^^^^^^^^^^^^^^^^^^
        Our code can play the role of both a tutorial and a tool.
        If you still need to become familiar with algorithms' implementations in Safe RL, the highly readable code in OmniSafe can help you get started quickly.
        You can see how each algorithm performs.
        If you want to build your algorithms, OmniSafe's highly robust code can also be an excellent tool!

    .. grid-item-card::
        :class-item: sd-font-weight-bold
        :class-header: sd-bg-primary sd-text-white sd-font-weight-bold
        :class-card: sd-outline-primary  sd-rounded-1
        :columns: 12 5 5 4

        Long-lived
        ^^^^^^^^^^^^^^^^^^^^^^^^^^
        Unlike other code that relies on a large number of external libraries, OmniSafe minimizes the dependency on third-party libraries.
        This avoids shortening the life of the project due to iterative changes in third-party library code and also optimizes the user's experience in installing and using OmniSafe, because they do not have to install lots of dependencies to run OmniSafe.

Before Reading
--------------

Before you start having fun reading the OmniSafe tutorial, we want you to understand the usage of colors in this tutorial.
In this tutorial, in general, the :bdg-info:`light blue boxes` indicate mathematically relevant derivations, including but not limited to :bdg-info-line:`Theorem`, :bdg-info-line:`Lemma`, :bdg-info-line:`Proposition`, :bdg-info-line:`Corollary`, and :bdg-info-line:`their proofs`, while the :bdg-success:`green boxes` indicate specifically :bdg-success-line:`implementations`, both :bdg-success-line:`theoretical` and :bdg-success-line:`code-based`.
We give an example below:

.. dropdown:: Example of OmniSafe color usage styles (Click here)
    :animate: fade-in-slide-down
    :class-title: sd-font-weight-bold sd-outline-primary sd-text-primary
    :class-body: sd-font-weight-bold

    .. card::
        :class-header: sd-bg-info sd-text-white sd-font-weight-bold
        :class-card: sd-outline-success  sd-rounded-1
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

            .. tab-item:: Config dict style

                .. code-block:: python
                    :linenos:

                    import omnisafe

                    env_id = 'SafetyPointGoal1-v0'
                    custom_cfgs = {
                        'train_cfgs': {
                            'total_steps': 1024000,
                            'vector_env_nums': 1,
                            'parallel': 1,
                        },
                        'algo_cfgs': {
                            'update_cycle': 2048,
                            'update_iters': 1,
                        },
                        'logger_cfgs': {
                            'use_wandb': False,
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
                        python train_policy.py --algo CPO --env-id SafetyPointGoal1-v0 --parallel 1 --total-steps 1024000 --device cpu --vector-env-nums 1 --torch-threads 1

You may not yet understand the above theory and the specific meaning of the code, but do not worry, we will make a detailed introduction later in the :doc:`../saferl/cpo` tutorial.

Long-Term Support and Support History
-------------------------------------

**OmniSafe** is mainly developed by the SafeRL research team directed by `Prof. Yaodong Yang <https://github.com/orgs/OmniSafeAI/people/PKU-YYang>`_,
Our SafeRL research team members include `Borong Zhang <https://github.com/muchvo>`_ , `Jiayi Zhou <https://github.com/Gaiejj>`_, `JTao Dai <https://github.com/calico-1226>`_,  `Weidong Huang <https://github.com/hdadong>`_, `Ruiyang Sun <https://github.com/rockmagma02>`_, `Xuehai Pan <https://github.com/XuehaiPan>`_ and `Jiamg Ji <https://github.com/zmsn-2077>`_.
If you have any questions in the process of using OmniSafe, or if you are willing to contribute to
this project, don't hesitate to ask your question on `the GitHub issue page <https://github.com/OmniSafeAI/omnisafe/issues/new/choose>`_, we will reply to you in 2-3 working days.

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
    :caption: base rl algorithm

    baserl/trpo
    baserl/ppo

.. toctree::
    :hidden:
    :maxdepth: 3
    :caption: safe rl algorithm

    saferl/cpo
    saferl/pcpo
    saferl/focops
    saferl/lag

.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: base rl api

    baserlapi/on_policy

.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: safe rl api

    saferlapi/first_order
    saferlapi/second_order
    saferlapi/lagrange
    saferlapi/penalty_function

.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: common

    common/buffer
    common/exp_grid
    common/lagrange
    common/normalizer
    common/logger

.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: utils

    utils/config
    utils/distributed
    utils/math
    utils/model
    utils/tools

.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: models

    model/actor
    model/critic
    model/actor_critic

.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: envs

    envs/core
    envs/wrapper
    envs/safety_gymnasium
    envs/adapter


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
