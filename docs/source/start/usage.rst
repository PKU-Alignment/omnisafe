Usage Video
===========

Quick Facts
-----------

.. card::
    :class-card: sd-outline-info  sd-rounded-1
    :class-body: sd-font-weight-bold

    #. Here we provide :bdg-info-line:`Examples` of how to use OmniSafe.
    #. You can train policy by running :bdg-info-line:`omnisafe train` .
    #. You can customize the configuration of the algorithm by running :bdg-info-line:`omnisafe train-config` .
    #. You can run a benchmark by running :bdg-info-line:`omnisafe benchmark` .
    #. You can run an evaluation by running :bdg-info-line:`omnisafe eval` .
    #. You can get some helps by running :bdg-info-line:`omnisafe help` .

Train policy
------------

.. card::
    :class-header: sd-bg-success sd-text-white
    :class-card: sd-outline-success  sd-rounded-1

    Example
    ^^^

    You can train a policy by running:

    .. code-block:: bash

        omnisafe train
        --algo PPO
        --total-steps 1024
        --vector-env-nums 1
        --custom-cfgs algo_cfgs:update_cycle
        --custom-cfgs 512

    Here we provide a video example:

    .. raw:: html

        <script async id="asciicast-inh7PHnvoAZCi88CeGY1EpRe9" src="https://asciinema.org/a/inh7PHnvoAZCi88CeGY1EpRe9.js"></script>


.. hint::
    The above command will train a policy with PPO algorithm, and the total training steps is 1024. The vector environment number is 1, which means that the training process will use 1 CPU core. The ``algo_cfgs:update_cycle`` is the update cycle of the PPO algorithm, which means that the policy will be updated every 512 steps.

Customize Configuration
-----------------------

.. card::
    :class-header: sd-bg-success sd-text-white
    :class-card: sd-outline-success  sd-rounded-1

    Example
    ^^^

    You can also customize the configuration of the algorithm by running:

    .. code-block:: bash

        omnisafe train-config "./saved_source/train_config.yaml"

    Here we provide a video example:

    .. raw:: html

        <script async id="asciicast-qCptIXhxYB2MWEytijrriVhUm" src="https://asciinema.org/a/qCptIXhxYB2MWEytijrriVhUm.js"></script>

.. hint::
    The above command will use a configuration file `train_config.yaml <https://github.com/OmniSafeAI/omnisafe/blob/main/tests/saved_source/train_config.yaml>`_ in the `saved_source <https://github.com/OmniSafeAI/omnisafe/tree/main/tests/saved_source>`_ directory to train policy. We have provided an example showing the file layer of the configuration file. You can customize the configuration of the algorithm in this file.

Run Benchmark
-------------

.. card::
    :class-header: sd-bg-success sd-text-white
    :class-card: sd-outline-success  sd-rounded-1

    Example
    ^^^
    You can run a benchmark by running:

    .. code-block:: bash

        omnisafe benchmark test_benchmark 2 "./saved_source/benchmark_config.yaml"

    Here we provide a video example:

    .. raw:: html

        <script async id="asciicast-gg6edB7OWiFENACpQzpfgFRx6" src="https://asciinema.org/a/gg6edB7OWiFENACpQzpfgFRx6.js"></script>

.. hint::
    The above command will run a benchmark with 2 CPU threads. The configuration file `benchmark_config.yaml <https://github.com/OmniSafeAI/omnisafe/blob/main/tests/saved_source/benchmark_config.yaml>`_ is in the `saved_source <https://github.com/OmniSafeAI/omnisafe/tree/main/tests/saved_source>`_ directory. We have provided an example showing the file layer of the configuration file. You can customize the configuration of the benchmark in this file.

Run Evaluation
--------------

.. card::
    :class-header: sd-bg-success sd-text-white
    :class-card: sd-outline-success  sd-rounded-1

    Example
    ^^^
    You can run an evaluation by running:

    .. code-block:: bash

        omnisafe eval ./saved_source/PPO-{SafetyPointGoal1-v0} "--num-episode" "1"

    Here we provide a video example:

    .. raw:: html

        <script async id="asciicast-UbRWY6EI6Nl7R27Lk3Rpk4HI5" src="https://asciinema.org/a/UbRWY6EI6Nl7R27Lk3Rpk4HI5.js"></script>

.. hint::
    The above command will run an evaluation with 2 CPU threads. The model parameters is in the `saved_source <https://github.com/OmniSafeAI/omnisafe/tree/main/tests/saved_source>`_ directory.

Get Help
--------

.. card::
    :class-header: sd-bg-success sd-text-white
    :class-card: sd-outline-success  sd-rounded-1

    Example
    ^^^
    If you have any questions, you can get help by running:

    .. code-block:: bash

        omnisafe --help

    Then you will see:

    .. raw:: html

        <script async id="asciicast-xQZ6RBafyXonZEqbVQ3htLPJT" src="https://asciinema.org/a/xQZ6RBafyXonZEqbVQ3htLPJT.js"></script>

.. hint::
    The above command will show the help information of OmniSafe,
    which may help you to some degree.
    If you still have any questions,
    just feel free to open an issue.
