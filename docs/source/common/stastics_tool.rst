OmniSafe Statistics Tools
=========================

.. currentmodule:: omnisafe.common.statistics_tools

Usage Example
-------------
Here we provide a simple example of how to use the :class:`StatisticsTools` class.
Suppose you want to tune the ``algo_cfgs:batch_size`` parameter of your algorithm,
then your ``run_experiment_grid.py`` file could look like this:

.. code-block:: python
    :linenos:

    if __name__ == '__main__':
        eg = ExperimentGrid(exp_name='Example')

        # Set the algorithms.
        example_policy = ['PPOLag', 'TRPOLag']

        # Set the environments.
        eg.add('env_id', 'SafetyAntVelocity-v1')

        eg.add('algo', example_policy)
        eg.add('train_cfgs:torch_threads', [1])
        eg.add('algo_cfgs:batch_size', [32, 64, 128])
        eg.add('logger_cfgs:use_wandb', [False])
        eg.add('seed', [0])
        # total experiment num is better to be divisible by num_pool
        # meanwhile, users should decide this value according to their machine
        eg.run(train, num_pool=6, gpu_id=None)

Then you run the experiment with the following command:

.. code-block:: bash

    cd ~/omnisafe/examples/benchmarks
    python run_experiment_grid.py

When the experiment is running, you can check it at ``omnisafe/examples/benchmarks/exp-x/Example`` .

Each experiment will be named with a hash value.
which encode different set of parameters.
In this example we set 6 kinds of parameters, that is 2 ``algorithms`` \* 3 ``batch_size``,
so 6 hash values will be generated, which denotes 6 different sets of parameters.

In this example, they are:

.. code-block:: bash

    SafetyAntVelocity-v1---1f58ce80fc9540b32a925d95694e3f836f80a5511e9e5c834e77195a2e9c3944
    SafetyAntVelocity-v1---7a451ea3e08cfb6caf64d05c307be9b6c32a509dc425f758387f90f96939d720
    SafetyAntVelocity-v1---7cefb92954e284496a08c3ca087af3971f8a37ba1845242208ef2c6afcaf4d27
    SafetyAntVelocity-v1---564ef55d6dac0002b8ecf848a240fe05de8639cc33229b4f773157dd2f828e71
    SafetyAntVelocity-v1---9997d3e3b2555d9f0da2703b24b376aa5ddd73d8abaffe95288b23bfd7304779
    SafetyAntVelocity-v1---50699a2818176e088a359b124296d67ac6fb130336c5f7b66f356b34f361e356

After the experiment is finished, you can use the ``~/omnisafe/examples/analyze_experiment_results.py``
script to analyze the results. For example, to plot the average return/cost of the
``SafetyAntVelocity-v1`` environment, you can set the ``~/omnisafe/examples/analyze_experiment_results.py``
file as follows:

.. code-block:: python
    :linenos:

    # just fill in the path in which experiment grid runs.
    PATH = '/home/gaiejj/PKU/omnisafe_zjy/examples/benchmarks/exp-x/Example'
    if __name__ == '__main__':
        st = StatisticsTools()
        st.load_source(PATH)
        # just fill in the name of the parameter of which value you want to compare.
        # then you can specify the value of the parameter you want to compare,
        # or you can just specify how many values you want to compare in single graph at most,
        # and the function will automatically generate all possible combinations of the graph.
        # but the two mode can not be used at the same time.
        st.draw_graph(parameter='algo_cfgs:batch_size', values=None, compare_num=3, cost_limit=None, show_image=True)

Then 2 images will be generated in the ``~/omnisafe/examples/`` directory.
Each image would be named with the hash value of the experiment.
If you want to compare the performance of different parameters,
you can refer to the hash value in the experiment directory.

Statistics Tools
----------------

.. card::
    :class-header: sd-bg-success sd-text-white
    :class-card: sd-outline-success  sd-rounded-1

    Documentation
    ^^^

    .. autoclass:: StatisticsTools
        :members:
        :private-members:
