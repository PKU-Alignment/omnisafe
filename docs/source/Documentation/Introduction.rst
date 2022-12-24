Introduction
============

.. contents:: Table of Contents
   :depth: 2


Welcome To OmniSafe Tutorial
----------------------------

Welcome to `OmniSafe <https://jmlr.org/papers/v16/garcia15a.html>`__ in Safe RL!
OmniSafe is a comprehensive and reliable benchmark for safe reinforcement learning, encompassing more than 20 different classes of algorithms covering a multitude of SafeRL domains, and delivering a new suite of testing environments.

.. admonition:: Hint
    :class: hint

    For beginners, it is necessary first to introduce you to Safe RL(Safe Reinforcement Learning).
    Safe Reinforcement Learning can be defined as the process of learning agent which maximize the expectation of the return on problems as well as ensure reasonably system performance and respect safety constraints during the learning and deployment processes.

**This tutorial is useful for reinforcement learning learners of many levels.**

.. grid:: 12 4 4 4


    .. grid-item-card::
        :class-item: sd-font-weight-bold
        :class-header: sd-bg-success sd-text-white sd-font-weight-bold
        :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3
        :columns: 12 6 6 4

        For Beginners
        ^^^^^^^^^^^^^
        If you are a beginner in machine learning with only some simple knowledge of linear algebra and probability theory, you can start with the mathematical fundamentals section of this tutorial.

    .. grid-item-card::
        :class-item: sd-font-weight-bold
        :class-header: sd-bg-info sd-text-white sd-font-weight-bold
        :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3
        :columns: 12 6 6 4

        For Average Users
        ^^^^^^^^^^^^^^^^^
        If you have a general understanding of RL algorithms but are unfamiliar with the concept of Safe RL.
        This tutorial provides an introduction to it so you can get started quickly.

    .. grid-item-card::
        :class-item: sd-font-weight-bold
        :class-header: sd-bg-primary sd-text-white sd-font-weight-bold
        :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3
        :columns: 12 6 6 4

        For Master
        ^^^^^^^^^^
        If you are already an expert in the field of RL, you can also gain new insights from our systematic introduction to Safe RL algorithms.
        Also, this tutorial will allow you to design your algorithms using OmniSafe quickly.


Why We Built This
-----------------

In recent years, `RL`_ (Reinforcement Learning) algorithms, especially `Deep RL`_ algorithms have achieved good performance in many tasks.
Examples include achieving high scores on Atari games with only visual input, completing complex control tasks in high dimensions, and beating human grandmasters at Go tournaments.
However, in the process of strategy updating by RL, the agents often learn **cheating or even dangerous behaviors** to improve their performance.
Such an agent that can quickly achieve high scores differs from our desired result.
Therefore, Safe RL algorithms are dedicated to solving the problem of how to train an agent to learn to achieve the desired simultaneously training goal without violating constraints.

.. admonition:: However
    :class: warning

    Even experienced RL researchers have difficulty understanding Safe RL's algorithms in a short time and quickly programming their implementation.

Therefore, OmniSafe will facilitate the subsequent study of Safe RL by providing both a **detailed and systematic introduction to the algorithm** and a **streamlined and robust code**.

.. tab-set::

    .. tab-item:: Problem I
        :sync: key1

        .. card::
            :class-header: sd-bg-danger  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-danger sd-border-{3} sd-shadow-sm sd-rounded-3
            :class-footer: sd-font-weight-bold

            Puzzling Math
            ^^^^^^^^^^^^^
            Safe RL algorithms are a class of algorithms built on a rigorous mathematical system.
            These algorithms have a detailed theoretical derivation, but they lack a unified symbolic system, which makes it difficult for beginners to learn them systematically and comprehensively.

    .. tab-item:: Problem II
        :sync: key2

        .. card::
            :class-header: sd-bg-danger sd-text-white sd-font-weight-bold
            :class-card: sd-outline-danger sd-border-{3} sd-shadow-sm sd-rounded-3
            :class-footer: sd-font-weight-bold

            Hard-to-find Codes
            ^^^^^^^^^^^^^^^^^^
            Most of the existing Safe RL algorithms **do not have open-source code**, making it difficult for beginners to grasp the ideas of the algorithms at the code level, and researchers suffer from incorrect implementations, unfair comparisons, and misleading conclusions.

.. tab-set::

    .. tab-item:: Soulution I
        :sync: key1

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outlinesuccess sd-border-{3} sd-shadow-sm sd-rounded-3
            :class-footer: sd-font-weight-bold

            Friendly Math
            ^^^^^^^^^^^^^
            OmniSafe tutorial provides a **unified and standardized notation system** that allows beginners to learn the theory of Safe RL algorithms in a complete and systematic way.

    .. tab-item:: Solution II
        :sync: key2

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outlinesuccess sd-border-{3} sd-shadow-sm sd-rounded-3
            :class-footer: sd-font-weight-bold

            Robust Code
            ^^^^^^^^^^^
            OmniSafe tutorial gives a **code-level** introduction in each algorithm introduction, allowing learners who are new to Safe RL theory to understand how to relate algorithmic ideas to code, and give experts in the field of Safe RL new insights into algorithm implementation.



.. _`RL`: https://en.wikipedia.org/wiki/Reinforcement_learning
.. _`Deep RL`: http://ufldl.stanford.edu/tutorial/

Code Design Principles
----------------------

.. grid:: 12 4 4 4
    :gutter: 1


    .. grid-item-card::
        :class-item: sd-font-weight-bold
        :class-header: sd-bg-success sd-text-white sd-font-weight-bold
        :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3
        :columns: 12 5 5 4

        Consistent and Inherited
        ^^^^^^^^^^^^^^^^^^^^^^^^
        Our code has a complete logic system that allows you to understand the connection between each algorithm and the similarities together with differences.
        For example, if you understand the Policy Gradient algorithm, then you can learn the PPO algorithm by simply reading the a new function and immediately grasping the code implementation of the PPO algorithm.

    .. grid-item-card::
        :class-item: sd-font-weight-bold
        :class-header: sd-bg-info sd-text-white sd-font-weight-bold
        :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3
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
        :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3
        :columns: 12 5 5 4

        Independent and Long-lived
        ^^^^^^^^^^^^^^^^^^^^^^^^^^
        Unlike other code that relies on a large number of external libraries, OmniSafe minimizes the dependency on third-party libraries.
        This avoids shortening the life of the project due to iterative changes in third-party library code also optimizes the users experience in installing and using OmniSafe, because they do not have to install lots of dependencies to run OmniSafe.

Before Reading
--------------

Before you start having fun reading the OmniSafe tutorial, we want you to understand the usage of colors in this tutorial.
In this tutorial, in general, the :bdg-info:`light blue boxes` indicate mathematically relevant derivations, including but not limited to :bdg-info-line:`Theorem`, :bdg-info-line:`Lemma`, :bdg-info-line:`Proposition`, :bdg-info-line:`Corollary`, and :bdg-info-line:`their proofs`, while the :bdg-success:`green boxes` indicate specific :bdg-success-line:`implementations`, both :bdg-success-line:`theoretical` and :bdg-success-line:`code-based`.
We give an example below:

.. dropdown:: Example of OmniSafe color usage styles (Click here)
    :animate: fade-in-slide-down
    :color: light
    :class-title: sd-font-weight-bold sd-outline-primary sd-text-secondary
    :class-body: sd-font-weight-bold

    .. card::
        :class-header: sd-bg-info sd-text-white sd-font-weight-bold
        :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3
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
        :class-card: sd-outline-success sd-border-{3} sd-shadow-sm sd-rounded-3 sd-font-weight-bold
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
                    agent = omnisafe.Agent('CPO', env, custom_cfgs=custom_dict)
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

                    We use ``train_on_policy.py`` as the entrance file. You can train the agent with
                    CPO simply using ``train_on_policy.py``, with arguments about CPO and environments
                    does the training. For example, to run CPO in SafetyPointGoal1-v0 , with
                    4 cpu cores and seed 0, you can use the following command:

                    .. code-block:: guess
                        :linenos:

                        cd omnisafe/examples
                        python train_on_policy.py --env-id SafetyPointGoal1-v0 --algo CPO --parallel 5 --epochs 1

You may not yet understand the above theory and the specific meaning of the code, but do not worry, we will make a detailed introduction later in the :doc:`../SafeRL/CPO` tutorial.

Long-Term Support and Support History
-------------------------------------

**OmniSafe** is currently maintained by Borong Zhang , `Jiayi Zhou <https://github.com/Gaiejj>`_, `JTao Dai <https://github.com/calico-1226>`_, `Weidong Huang <https://github.com/hdadong>`_, `Xuehai Pan <https://github.com/XuehaiPan>`_ and `Jiamg Ji <https://github.com/zmsn-2077>`_.
If you have any question in the process of using OmniSafe, of if you are willing to make a contribution to
this project, don't hesitate to ask your question in `the GitHub issue page <https://github.com/PKU-MARL/omnisafe/issues/new/choose>`_, we will reply you in 2-3 working days.
