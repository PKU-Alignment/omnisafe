Environments Customization
===========================

OmniSafe supports a flexible environment customization interface. Users only need to make minimal
interface adaptations within the simplest template provided by OmniSafe to complete the environment
customization.

.. note::
    The highlight of OmniSafe's environment customization is that **users only need to modify the code at the environment layer**, to enjoy OmniSafe's complete set of training, saving, and data logging mechanisms. This allows users who install from PyPI to use it easily and only focus on the dynamics of the environment.


Get Started with the Simplest Template
--------------------------------------

OmniSafe offers a minimal implementation of an environment template as an example of a customized
environments, :doc:`../envs/custom`.
We recommend reading this template in detail and customizing it based on it.

.. card::
    :class-header: sd-bg-success sd-text-white
    :class-card: sd-outline-success  sd-rounded-1

    Frequently Asked Questions
    ^^^
    1. What changes are necessary to embed the environment into OmniSafe?
    2. My environment requires specific parameters; can these be integrated into OmniSafe's parameter mechanism?
    3. I need to log information during training; how can I achieve this?
    4. After embedding the environment, how do I run the algorithms in OmniSafe for training?

For the above questions, we provide a complete Jupyter Notebook example (Please see our tutorial on
GitHub page). We will demonstrate how to start from the most common environments in
`Gymnasium <https://gymnasium.farama.org/>`_ style, implement
environment customization and complete the training process.


Customization of Your Environments
-----------------------------------

From Source Code
^^^^^^^^^^^^^^^^

If you are installing from the source code, you can follow the steps below:

.. card::
    :class-header: sd-bg-success sd-text-white
    :class-card: sd-outline-success  sd-rounded-1

    Build from Source Code
    ^^^
    1. Create a new file under `omnisafe/envs/`, for example, `omnisafe/envs/my_env.py`.
    2. Customize the environment in `omnisafe/envs/my_env.py`. Assuming the class name is `MyEnv`, and the environment name is `MyEnv-v0`.
    3. Add `from .my_env import MyEnv` in `omnisafe/envs/__init__.py`.
    4. Run the following command in the `omnisafe/examples` folder:

    .. code-block:: bash
        :linenos:

        python train_policy.py --algo PPOLag --env MyEnv-v0

From PyPI
^^^^^^^^^

.. card::
    :class-header: sd-bg-success sd-text-white
    :class-card: sd-outline-success  sd-rounded-1

    Build from PyPI
    ^^^
    1. Customize the environment in any folder. Assuming the class name is `MyEnv`, and the environment name is `MyEnv-v0`.
    2. Import OmniSafe and the environment registration decorator.
    3. Run the training.

    For a short but detailed example, please see `examples/train_from_custom_env.py`
