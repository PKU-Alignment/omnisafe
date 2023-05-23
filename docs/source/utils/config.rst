OmniSafe Config
===============

.. currentmodule:: omnisafe.utils.config

.. autosummary::

    Config
    ModelConfig
    get_default_kwargs_yaml
    check_all_configs
    __check_algo_configs
    __check_logger_configs


Config
------

OmniSafe uses yaml file to store all the configurations. The configuration file
is stored in ``omnisafe/configs``. The configuration file is divided into
several parts.

Take ``PPOLag`` as an example, the configuration file is as follows:

.. list-table::

    *   -   Config
        -   Description
    *   -   ``train_cfgs``
        -   Training configurations.
    *   -   ``algo_cfgs``
        -   Algorithm configurations
    *   -   ``logger_cfgs``
        -   Logger configurations
    *   -   ``model_cfgs``
        -   Model configurations
    *   -   ``lagrange_cfgs``
        -   Lagrange configurations

Specifically, the ``train_cfgs`` is as follows:

.. list-table::

    *   -   Config
        -   Description
        -   Value
    *   -   ``device``
        -   Device to use.
        -   ``cuda`` or ``cpu``
    *   -   ``torch_threads``
        -   Number of threads to use.
        -   16
    *   -   ``vector_env_nums``
        -   Number of vectorized environments.
        -   1
    *   -   ``parallel``
        -   Number of parallel agent, similar to A3C.
        -   1
    *   -   ``total_steps``
        -   Total number of training steps.
        -   1000000

Other configurations are similar to ``train_cfgs``. You can refer to the ``omnisafe/configs`` for more details.

.. card::
    :class-header: sd-bg-success sd-text-white
    :class-card: sd-outline-success  sd-rounded-1

    Documentation
    ^^^

    .. autoclass:: Config
        :members:
        :private-members:

Model Config
------------

.. card::
    :class-header: sd-bg-success sd-text-white
    :class-card: sd-outline-success  sd-rounded-1

    Documentation
    ^^^

    .. autoclass:: ModelConfig
        :members:
        :private-members:



Common Method
-------------

.. card::
    :class-header: sd-bg-success sd-text-white
    :class-card: sd-outline-success  sd-rounded-1

    Documentation
    ^^^

    .. autofunction:: get_default_kwargs_yaml

.. card::
    :class-header: sd-bg-success sd-text-white
    :class-card: sd-outline-success  sd-rounded-1

    Documentation
    ^^^

    .. autofunction:: check_all_configs

.. card::
    :class-header: sd-bg-success sd-text-white
    :class-card: sd-outline-success  sd-rounded-1

    Documentation
    ^^^

    .. autofunction:: __check_algo_configs

.. card::
    :class-header: sd-bg-success sd-text-white
    :class-card: sd-outline-success  sd-rounded-1

    Documentation
    ^^^

    .. autofunction:: __check_logger_configs
