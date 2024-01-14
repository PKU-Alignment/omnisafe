Lagrangian Methods
==================

Quick Facts
-----------

.. card::
    :class-card: sd-outline-info  sd-rounded-1
    :class-body: sd-font-weight-bold

    #. Lagrangian Method can be applied to almost :bdg-info-line:`any` RL algorithm.
    #. Lagrangian Method turns :bdg-danger-line:`unsafe` algorithm to a :bdg-info-line:`safe` one.
    #. The OmniSafe implementation of Lagrangian Methods covers up to 6 kinds of :bdg-info-line:`on policy` and :bdg-info-line:`off policy` algorithm.
    #. An :bdg-ref-info-line:`API Documentation <ppolagapi>` is available for PPOLag.



Lagrangian Methods Theorem
--------------------------

Background
~~~~~~~~~~

In the previous introduction of algorithms,
we know that Safe RL mainly solves the constraint optimization problem of CMDP.

.. hint::

    Constrained optimization problems tend to be more challenging than unconstrained optimization problems.

Therefore, the natural idea is to convert a constrained optimization problem
into an unconstrained optimization problem.
Then solve it using classical optimization algorithms,
such as stochastic gradient descent.
Lagrangian methods is a kind of method solving constraint problems that are
widely used in machine learning.
By using adaptive penalty coefficients to enforce constraints,
Lagrangian methods convert the solution of a constrained optimization problem
to the solution of an unconstrained optimization problem.
In this section, we will briefly introduce Lagrangian methods,
and give corresponding implementations in **TRPO** and **PPO**.
TRPO and PPO are the algorithms we introduced earlier.
If you lack understanding of it, it doesn't matter.
Please refer to the :doc:`TRPO tutorial<../baserl/trpo>` and
:doc:`PPO tutorial<../baserl/ppo>`.

.. grid:: 2

    .. grid-item::
        :columns: 12 6 6 6

        .. card::
            :class-header: sd-bg-primary sd-text-white sd-font-weight-bold
            :class-card: sd-outline-primary  sd-rounded-1 sd-font-weight-bold

            Advantages of Lagrangian Methods
            ^^^
            -  Relatively simple to implement.

            -  The principle is straightforward to understand.

            -  Can be applied to a variety of algorithms.

            -  Highly scalable.

    .. grid-item::
        :columns: 12 6 6 6

        .. card::
            :class-header: sd-bg-warning sd-text-white sd-font-weight-bold
            :class-card: sd-outline-warning  sd-rounded-1 sd-font-weight-bold

            Problems of Lagrangian Methods
            ^^^
            -  Different hyperparameters need to be set for different tasks.

            -  Not necessarily valid for all tasks.

            -  Problems of overshoot.

            -  Difficult to handle multiple cost tasks directly.

------

Optimization Objective
~~~~~~~~~~~~~~~~~~~~~~

As we mentioned in the previous chapters, the optimization problem of CMDPs can be expressed as follows:


.. math::
    :label: lag-eq-1


    \max_{\pi \in \Pi_{\boldsymbol{\theta}}} &J^R(\pi) \\
    \text {s.t.}~~& J^{\mathcal{C}}(\pi) \leq d


where :math:`\Pi_{\boldsymbol{\theta}} \subseteq \Pi` denotes the set of parametrized policies with parameters :math:`{\boldsymbol{\theta}}`.
In local policy search for CMDPs,
we additionally require policy iterates to be feasible for the CMDP,
so instead of optimizing over :math:`\Pi_{\boldsymbol{\theta}}`,
algorithm should optimize over :math:`\Pi_{\boldsymbol{\theta}} \cap \Pi_C`.
Specifically, for the TRPO and PPO algorithms,
constraints on the differences between old and new policies should also be added.
To solve this constrained problem, please read the :doc:`TRPO tutorial<../baserl/trpo>`.
The final optimization goals are as follows:


.. math::
    :label: lag-eq-2

    \pi_{k+1}&=\arg \max _{\pi \in \Pi_{\boldsymbol{\theta}}} J^R(\pi) \\
    \text { s.t. } ~~ J^{\mathcal{C}}(\pi) &\leq d \\
    D\left(\pi, \pi_k\right) &\leq \delta


where :math:`D` is some distance measure and :math:`\delta` is the step size.

------

Lagrangian Method Theorem
-------------------------

Lagrangian methods
~~~~~~~~~~~~~~~~~~

Constrained MDP (CMDP) are often solved using the Lagrange methods.
In Lagrange methods, the CMDP is converted into an equivalent unconstrained problem.
In addition to the objective, a penalty term is added for infeasibility,
thus making infeasible solutions sub-optimal.

.. card::
    :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
    :class-card: sd-outline-info  sd-rounded-1
    :class-footer: sd-font-weight-bold
    :link: lagrange_theorem
    :link-type: ref

    Theorem 1
    ^^^
    Given a CMDP, the unconstrained problem can be written as:

    .. math::
        :label: lag-eq-3

        \min _{\lambda \geq 0} \max _{\boldsymbol{\theta}} G(\lambda, {\boldsymbol{\theta}})=\min _{\lambda \geq 0} \max _{\boldsymbol{\theta}} [J^R(\pi)-\lambda J^C(\pi)]


    where :math:`G` is the Lagrangian and :math:`\lambda \geq 0` is the Lagrange multiplier (a penalty coefficient).
    Notice, as :math:`\lambda` increases, the solution to the Problem :eq:`lag-eq-1` converges to that of the Problem :eq:`lag-eq-3`.
    +++
    The theorem base of :bdg-info:`Theorem 1` can be found in :bdg-info-line:`Lagrange Duality`, click this card to jump to view.

.. hint::

        The Lagrangian method is a **two-step** process.

        - First, we solve the unconstrained problem :eq:`lag-eq-3` to find a feasible solution :math:`{\boldsymbol{\theta}}^*`
        - Then, we increase the penalty coefficient :math:`\lambda` until the constraint is satisfied.

        The final solution is :math:`\left({\boldsymbol{\theta}}^*, \lambda^*\right)`.
        The goal is to find a saddle point :math:`\left({\boldsymbol{\theta}}^*\left(\lambda^*\right), \lambda^*\right)` of the Problem :eq:`lag-eq-1`,
        which is a feasible solution. (A feasible solution of the CMDP is a solution which satisfies :math:`J^C(\pi) \leq d` )

------

Practical Implementation
------------------------

Intuitively, we train the agent to maximize the reward in the classical
strategy gradient descent algorithm.
If a particular action :math:`a` in state :math:`s` can bring a relatively
higher reward,
we increase the probability that the agent will choose action :math:`a` under
:math:`s`,
and conversely, we will reduce this probability.

.. hint::

    Lagrangian methods add two extra steps to the above process.

    - One is to adjust the reward function,
      and if the agent's actions violate the constraint, the reward will reduce accordingly.
    - The second is a slow update of the penalty factor.
      If the agent violates fewer constraints, the penalty coefficient will gradually decrease,
      and conversely, it will gradually increase.

Next we will introduce the specific implementation of the Lagrange method in
the TRPO and PPO algorithms.

Policy update
~~~~~~~~~~~~~

.. tab-set::

    .. tab-item:: Fast Step

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Surrogate function update
            ^^^
            Previously, in TRPO and PPO, we used to have the agent sample a series of data from the environment,
            and at the end of the episode, use this data to update the agent several times,
            as described in Problem :eq:`lag-eq-2`.
            With the addition of the Lagrange method,
            we need to make a change to the original surrogate function, as it is shown below:

            .. math::

                \max _{\pi \in \prod_{\boldsymbol{\theta}}}[J^R(\pi)-\lambda J^C(\pi)] \\
                \text { s.t. } D\left(\pi, \pi_k\right) \leq \delta


            In a word, we only need to punish the agent with its reward by
            :math:`\lambda` with each step of updates. In fact, this is just a minor
            change made on TRPO and PPO.

    .. tab-item:: Slow Step

        .. card::
            :class-header: sd-bg-success  sd-text-white sd-font-weight-bold
            :class-card:  sd-outline-success  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Lagrange multiplier update
            ^^^
            After all rounds of policy updates to the agent are complete, we
            will perform an update on the Lagrange multiplier, that is:

            .. math::

                \min _\lambda(1-\lambda) [J^R(\pi)-\lambda J^C(\pi)] \\
                \text { s.t. } \lambda \geq 0


            Specifically, on the :math:`k^{t h}` update, the above equation is often
            written as below in the actual calculation process:

            .. math::

                \lambda_{k+1}=\max \left(\lambda_k+ \eta_\lambda\left(J^C(\pi)-d\right), 0\right)


            where :math:`\eta_\lambda` is the learning rate of :math:`\lambda`.

            Ultimately, we only need to add the above two steps to the TRPO and PPO;
            then we will get the TRPOLag and the PPOLag.

            .. attention::
                :class: warning

                In practice, we often need to manually set the initial value of :math:`\lambda` as well as the learning rate :math:`\eta_\lambda`.
                Unfortunately, Lagrange algorithms are algorithms that **are sensitive to hyperparameter selection**.

                - If the initial value of :math:`\lambda` or learning rate :math:`\eta_\lambda` is chosen to be large,
                  the agent may suffer from a low reward.
                - Else, it may violate the constraints.

                So we often struggle to choose a compromise hyperparameter to balance reward and constraints.

------

Code with OmniSafe
~~~~~~~~~~~~~~~~~~

Safe RL algorithms for :bdg-success-line:`TRPO`, :bdg-success-line:`PPO`, :bdg-success-line:`NPG`, :bdg-success-line:`DDPG`, :bdg-success-line:`SAC` and :bdg-success-line:`TD3` are currently implemented in OmniSafe using Lagrangian methods.
This section will explain how to deploy Lagrangian methods on PPO algorithms at the code level using PPOLag as an example.
OmniSafe has :bdg-success:`Lagrange` as a separate module and you can easily deploy it on most RL algorithms.

Quick start
"""""""""""

.. card::
    :class-header: sd-bg-success sd-text-white sd-font-weight-bold
    :class-card: sd-outline-success  sd-rounded-1 sd-font-weight-bold
    :class-footer: sd-font-weight-bold

    Run PPOLag in OmniSafe
    ^^^
    Here are 3 ways to run PPOLag in OmniSafe:

    * Run Agent from preset yaml file
    * Run Agent from custom config dict
    * Run Agent from custom terminal config

    .. tab-set::

        .. tab-item:: Yaml file style

            .. code-block:: python
                :linenos:

                import omnisafe


                env_id = 'SafetyPointGoal1-v0'

                agent = omnisafe.Agent('PPOLag', env_id)
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

                agent = omnisafe.Agent('PPOLag', env_id, custom_cfgs=custom_cfgs)
                agent.learn()


        .. tab-item:: Terminal config style

            We use ``train_policy.py`` as the entrance file. You can train the agent with PPOLag simply using ``train_policy.py``, with arguments about PPOLag and environments does the training.
            For example, to run PPOLag in SafetyPointGoal1-v0 , with 1 torch thread, seed 0 and single environment, you can use the following command:

            .. code-block:: bash
                :linenos:

                cd examples
                python train_policy.py --algo PPOLag --env-id SafetyPointGoal1-v0 --parallel 1 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 1

------

Architecture of functions
"""""""""""""""""""""""""

-  ``PPOLag.learn()``

   - ``PPOLag._env.rollout()``
   - ``PPOLag._update()``

     - ``PPOLag._buf.get()``
     - ``PPOLag.update_lagrange_multiplier(ep_costs)``
     - ``PPOLag._update_actor()``
     - ``PPOLag._update_cost_critic()``
     - ``PPOLag._update_reward_critic()``

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

                The following configs are specific to PPOLag algorithm.

                - clip (float): Clipping parameter for PPOLag.

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

    .. tab-item:: Lagrange

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success  sd-rounded-1 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            Lagrange Configs
            ^^^

            - cost_limit (float): Tolerance of constraint violation.
            - lagrangian_multiplier_init (float): Initial value of Lagrange multiplier.
            - lambda_lr (float): Learning rate of Lagrange multiplier.
            - lambda_optimizer (str): Optimizer for Lagrange multiplier.

------

References
----------

-  `Constrained Policy Optimization <https://arxiv.org/abs/1705.10528>`__
-  `Trust Region Policy Optimization <https://arxiv.org/abs/1502.05477>`__
-  `Proximal Policy Optimization Algorithms <https://arxiv.org/pdf/1707.06347.pdf>`__
-  `Benchmarking Safe Exploration in Deep Reinforcement Learning <https://www.semanticscholar.org/paper/Benchmarking-Safe-Exploration-in-Deep-Reinforcement-Achiam-Amodei/4d0f6a6ffcd6ab04732ff76420fd9f8a7bb649c3#:~:text=Benchmarking%20Safe%20Exploration%20in%20Deep%20Reinforcement%20Learning%20Joshua,to%20learn%20optimal%20policies%20by%20trial%20and%20error.>`__
