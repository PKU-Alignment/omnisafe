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



Lagrangian Methods Theorem
--------------------------

Background
~~~~~~~~~~

In the previous introduction of algorithms,
we know that SafeRL mainly solves the constraint optimization problem of CMDP.

.. hint::

    Constrained optimization problems tend to be more challenging than unconstrained optimization problems.

Therefore, the natural idea is to convert a constrained optimization problem into an unconstrained optimization problem.
Then solve it using classical optimization algorithms,
such as stochastic gradient descent.
Lagrangian Methods is a kind of method solving constraint problems that are widely used in machine learning.
By using adaptive penalty coefficients to enforce constraints,
Lagrangian methods convert the solution of a constrained optimization problem to the solution of an unconstrained optimization problem.
In the :bdg-info-line:`section`, we will briefly introduce Lagrangian methods,
and give corresponding implementations in **TRPO** and **PPO**.
TRPO and PPO are the algorithms we introduced earlier,
if you lack understanding of it, it doesn't matter.
Please read the :doc:`TRPO tutorial<../BaseRL/TRPO>` and :doc:`PPO tutorial<../BaseRL/PPO>` we wrote,
you will soon understand how it works.

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
            -  Different hyper-parameters need to be set for different tasks.

            -  Not necessarily valid for all tasks.

            -  Problems of overshoot.

            -  Difficult to handle multiple cost tasks directly.

------

Optimization Objective
~~~~~~~~~~~~~~~~~~~~~~

As we mentioned in the previous chapters, the optimization problem of CMDPs can be expressed as follows:

.. _`lag-eq-1`:

.. math::

    \max_{\pi \in \Pi_\theta} &J^R(\pi) \\
    \text {s.t.}~~& J^{\mathcal{C}}(\pi) \leq d


where :math:`\Pi_\theta \subseteq \Pi` denotes the set of parametrized policies with parameters :math:`\theta`.
In local policy search for CMDPs,
we additionally require policy iterates to be feasible for the CMDP,
so instead of optimizing over :math:`\Pi_\theta`,
algorithm should optimize over :math:`\Pi_\theta \cap \Pi_C`.
Specifically, for the TRPO and PPO algorithms,
constraints on the differences between old and new policies should also be added.
To solve this constrained problem, please read the :doc:`TRPO tutorial<../BaseRL/TRPO>`.
The final optimization goals are as follows:

.. _`lag-eq-2`:

.. math::

    &\pi_{k+1}=\arg \max _{\pi \in \Pi_\theta} J^R(\pi) \\
    \text { s.t. } ~~ &J^{\mathcal{C}}(\pi) \leq d \\
    &D\left(\pi, \pi_k\right) \leq \delta


where :math:`D` is some distance measure and :math:`\delta` is the step size.

------

Lagrangian Method Theorem
-------------------------

Lagrangian methods
~~~~~~~~~~~~~~~~~~

Constrained MDP's are often solved using the Lagrange methods.
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

    .. _`lag-eq-3`:

    .. math::

        \min _{\lambda \geq 0} \max _\theta G(\lambda, \theta)=\min _{\lambda \geq 0} \max _\theta [J^R(\pi)-\lambda J^C(\pi)]


    where :math:`G` is the Lagrangian and :math:`\lambda \geq 0` is the Lagrange multiplier (a penalty coefficient).
    Notice, as :math:`\lambda` increases, the solution to the Problem :ref:`(1)<lag-eq-1>` converges to that of the Problem :ref:`(3)<lag-eq-3>`.
    +++
    The theorem base of :bdg-info:`Theorem 1` can be found in :bdg-info-line:`Lagrange Duality`, click this card to jump to view.

.. hint::

        The Lagrangian method is a **two-step** process.

        #. First, we solve the unconstrained problem :ref:`(3)<lag-eq-3>` to find a feasible solution :math:`\theta^*`
        #. Then, we increase the penalty coefficient :math:`\lambda` until the constraint is satisfied.

        The final solution is :math:`\left(\theta^*, \lambda^*\right)`.
        The goal is to find a saddle point :math:`\left(\theta^*\left(\lambda^*\right), \lambda^*\right)` of the Problem :ref:`(1)<lag-eq-1>`,
        which is a feasible solution. (A feasible solution of the CMDP is a solution which satisfies :math:`J^C(\pi) \leq d` )

------

Practical Implementation
------------------------

intuitively, we train the agent to maximize the reward in the classical strategy gradient descent algorithm.
If a particular action :math:`a` in state :math:`s` can bring a relatively higher reward,
we increase the probability that the agent will choose action :math:`a` under :math:`s`,
and conversely, we will reduce this probability.

.. hint::

    Lagrangian methods add two extra steps to the above process.

    - One is to adjust the reward function,
      and if the agent's actions violate the constraint, the reward will reduce accordingly.
    - The second is a slow update of the penalty factor.
      If the agent violates fewer constraints, the penalty coefficient will gradually decrease,
      and conversely, it will gradually increase.

Next we will introduce the specific implementation of the Lagrange method in the TRPO and PPO algorithms.

Policy update
~~~~~~~~~~~~~

.. tab-set::

    .. tab-item:: Fast Step

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-info  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Surrogate function update
            ^^^
            Previously, in TRPO and PPO, we used to have the agent sample a series of data from the environment,
            and at the end of the episode, use this data to update the agent several times,
            as described in Problem :ref:`(2)<lag-eq-2>`.
            With the addition of the Lagrange method,
            we need to make a change to the original surrogate function, as it is shown below:

            .. math::

                \max _{\pi \in \prod_\theta}[J^R(\pi)-\lambda J^C(\pi)] \\
                \text { s.t. } D\left(\pi, \pi_k\right) \leq \delta


            In a word, we only need to punish the agent with its reward by
            :math:`\lambda` with each step of updates. In fact, this is just a minor
            change made on TRPO and PPO.

    .. tab-item:: Slow Step

        .. card::
            :class-header: sd-bg-success  sd-text-white sd-font-weight-bold
            :class-card:  sd-outline-info  sd-rounded-1
            :class-footer: sd-font-weight-bold

            Lagrange multiplier update
            ^^^
            After all rounds of policy updates to the agent are complete, We will
            perform an update on the Lagrange multiplier that is:

            .. math::

                \min _\lambda(1-\lambda) [J^R(\pi)-\lambda J^C(\pi)] \\
                \text { s.t. } \lambda \geq 0


            Specifically, on the :math:`k^{t h}` update, the above align is often
            written as below in the actual calculation process:

            .. math::

                \lambda_{k+1}=\max \left(\lambda_k+ \eta_\lambda\left(J^C(\pi)-d\right), 0\right)


            where :math:`\eta_\lambda` is the learning rate of :math:`\lambda`.

            Ultimately, we only need to add the above two steps to the TRPO and PPO;
            then we will get the TRPO-lag and the PPO-lag.

            .. attention::
                :class: warning

                In practice, We often need to manually set the initial value of as well as the learning rate.
                Unfortunately, Lagrange algorithms are algorithms that **are sensitive to hyperparameter selection**.

                - If the initial value of :math:`\lambda` or learning rate is chosen to be large,
                  the agent may suffer from a low reward.
                - Else, it may violate the constraints.

                So we often struggle to choose a compromise hyperparameter to balance reward and constraints.

------

Code with OmniSafe
~~~~~~~~~~~~~~~~~~

Safe RL algorithms for :bdg-success-line:`TRPO`, :bdg-success-line:`PPO`, :bdg-success-line:`NPG`, :bdg-success-line:`DDPG`, :bdg-success-line:`SAC` and :bdg-success-line:`TD3` are currently implemented in omnisafe using Lagrangian methods.
This section will explain how to deploy Lagrangian methods on PPO algorithms at the code level using PPOLag as an example.
OmniSafe has :bdg-success:`Lagrange` as a separate module and you can easily deploy it on most RL algorithms.

Quick start
"""""""""""

.. card::
    :class-header: sd-bg-success sd-text-white sd-font-weight-bold
    :class-card: sd-outline-success  sd-rounded-1 sd-font-weight-bold
    :class-footer: sd-font-weight-bold

    Run PPOLag in Omnisafe
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

                agent = omnisafe.Agent('PPOLag', env_id, custom_cfgs=custom_cfgs)
                agent.learn()


        .. tab-item:: Terminal config style

            We use ``train_on_policy.py`` as the entrance file. You can train the agent with PPOLag simply using ``train_on_policy.py``, with arguments about PPOLag and environments does the training.
            For example, to run PPOLag in SafetyPointGoal1-v0 , with 4 cpu cores and seed 0, you can use the following command:

            .. code-block:: bash
                :linenos:

                cd examples
                python train_policy.py --algo PPOLag --env-id SafetyPointGoal1-v0 --parallel 1 --total-steps 1024000 --device cpu --vector-env-nums 1 --torch-threads 1
                
------

Architecture of functions
"""""""""""""""""""""""""

-  ``PPOLag.learn()``

   - ``env.roll_out()``
   - ``PPOLag.update()``

     - ``PPOLag.buf.get()``
     - ``PPOLag.pre_process_data(raw_data)``
     - ``PPOLag.update_lagrange_multiplier(ep_costs)``
     - ``PPOLag.update_policy_net()``
     - ``PPOLag.update_cost_net()``
     - ``PPOLag.update_value_net()``


- ``PPOLag.log()``

------

Documentation of new functions
""""""""""""""""""""""""""""""

.. tab-set::

    .. tab-item:: PPOLag.compute_loss_pi(data: dict)

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success  sd-rounded-1 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            PPOLag.compute_loss_pi(data: dict)
            ^^^
            Compute the loss of policy network, flowing the next steps:

            (1) Compute the clip surrogate function.

            .. code-block:: python
                :linenos:

                dist, _log_p = self.ac.pi(data['obs'], data['act'])
                ratio = torch.exp(_log_p - data['log_p'])
                ratio_clip = torch.clamp(ratio, 1 - self.clip, 1 + self.clip)
                loss_pi = -(torch.min(ratio * data['adv'], ratio_clip * data['adv'])).mean()
                loss_pi -= self.entropy_coef * dist.entropy().mean()


            (2) Punish the actor for violating the constraint.

            .. code-block:: python
                :linenos:

                penalty = self.lambda_range_projection(self.lagrangian_multiplier).item()
                loss_pi += penalty * ((ratio * data['cost_adv']).mean())
                loss_pi /= 1 + penalty


    .. tab-item:: Lagrange.update_lagrange_multiplier(ep_costs: float)

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success  sd-rounded-1 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            Lagrange.update_lagrange_multiplier(ep_costs: float)
            ^^^
            Update Lagrange multiplier (:math:`\lambda`)

            .. hint::
                ``ep_costs`` obtained from: ``self.logger.get_stats('EpCost')[0]``
                are already averaged across MPI processes.

            .. code-block:: python
                :linenos:

                self.lambda_optimizer.zero_grad()
                lambda_loss = self.compute_lambda_loss(ep_costs)
                lambda_loss.backward()
                self.lambda_optimizer.step()
                self.lagrangian_multiplier.data.clamp_(0)

            .. hint::
                ``self.lagrangian_multiplier.data.clamp_(0)`` is used to avoid negative values of :math:`\lambda`

------

Parameters
""""""""""

.. tab-set::

    .. tab-item:: Specific Parameters

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success  sd-rounded-1 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            Specific Parameters
            ^^^
            -  target_kl(float): Constraint for KL-distance to avoid too far gap
            -  cg_damping(float): parameter plays a role in building Hessian-vector
            -  cg_iters(int): Number of iterations of conjugate gradient to perform.
            -  cost_limit(float): Constraint for agent to avoid too much cost

    .. tab-item:: Basic parameters

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success  sd-rounded-1 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            Basic parameters
            ^^^
            -  algo (string): The name of algorithm corresponding to current class,
               it does not actually affect any things which happen in the following.
            -  actor (string): The type of network in actor, discrete or continuous.
            -  model_cfgs (dictionary) : Actor and critic's net work configuration,
               it originates from ``algo.yaml`` file to describe ``hidden layers`` , ``activation function``, ``shared_weights`` and ``weight_initialization_mode``.

               -  shared_weights (bool) : Use shared weights between actor and critic network or not.

               -  weight_initialization_mode (string) : The type of weight initialization method.

                  -  pi (dictionary) : parameters for actor network ``pi``

                     -  hidden_sizes:

                        -  64
                        -  64

                     -  activations: tanh

                  -  val (dictionary) parameters for critic network ``v``

                     -  hidden_sizes:

                        -  64
                        -  64

                        .. hint::

                            ======== ================  ========================================================================
                            Name        Type              Description
                            ======== ================  ========================================================================
                            ``v``    ``nn.Module``     Gives the current estimate of **V** for states in ``s``.
                            ``pi``   ``nn.Module``     Deterministically or continuously computes an action from the agent,
                                                       conditioned on states in ``s``.
                            ======== ================  ========================================================================

                  -  activations: tanh
                  -  env_id (string): The name of environment we want to roll out.
                  -  seed (int): Define the seed of experiments.
                  -  parallel (int): Define the seed of experiments.
                  -  epochs (int): The number of epochs we want to roll out.
                  -  steps_per_epoch (int):The number of time steps per epoch.
                  -  pi_iters (int): The number of iteration when we update actor network per mini batch.
                  -  critic_iters (int): The number of iteration when we update critic network per mini batch.

    .. tab-item:: Optional parameters

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success  sd-rounded-1 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            Optional parameters
            ^^^
            -  use_cost_critic (bool): Use cost value function or not.
            -  linear_lr_decay (bool): Use linear learning rate decay or not.
            -  exploration_noise_anneal (bool): Use exploration noise anneal or not.
            -  reward_penalty (bool): Use cost to penalize reward or not.
            -  kl_early_stopping (bool): Use KL early stopping or not.
            -  max_grad_norm (float): Use maximum gradient normalization or not.
            -  scale_rewards (bool): Use reward scaling or not.

    .. tab-item:: Buffer parameters

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success  sd-rounded-1 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            Buffer parameters
            ^^^
            .. hint::
                  ============= =============================================================================
                     Name                    Description
                  ============= =============================================================================
                  ``Buffer``      A buffer for storing trajectories experienced by an agent interacting
                                  with the environment, and using **Generalized Advantage Estimation (GAE)**
                                  for calculating the advantages of state-action pairs.
                  ============= =============================================================================

            .. warning::
                Buffer collects only raw data received from environment.

            -  gamma (float): The gamma for GAE.
            -  lam (float): The lambda for reward GAE.
            -  adv_estimation_method (float):Roughly what KL divergence we think is
               appropriate between new and old policies after an update. This will
               get used for early stopping. (Usually small, 0.01 or 0.05.)
            -  standardized_reward (int):  Use standardized reward or not.
            -  standardized_cost (bool): Use standardized cost or not.

------

References
----------

-  `Constrained Policy Optimization <https://arxiv.org/abs/1705.10528>`__
-  `Trust Region Policy Optimization <https://arxiv.org/abs/1502.05477>`__
-  `Proximal Policy Optimization Algorithms <https://arxiv.org/pdf/1707.06347.pdf>`__
-  `Benchmarking Safe Exploration in Deep Reinforcement Learning <https://www.semanticscholar.org/paper/Benchmarking-Safe-Exploration-in-Deep-Reinforcement-Achiam-Amodei/4d0f6a6ffcd6ab04732ff76420fd9f8a7bb649c3#:~:text=Benchmarking%20Safe%20Exploration%20in%20Deep%20Reinforcement%20Learning%20Joshua,to%20learn%20optimal%20policies%20by%20trial%20and%20error.>`__
