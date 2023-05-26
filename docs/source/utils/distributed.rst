OmniSafe Distributed
====================

.. currentmodule:: omnisafe.utils.distributed

.. autosummary::

    setup_distributed
    get_rank
    world_size
    fork
    avg_tensor
    avg_grads
    sync_params
    avg_params
    dist_avg
    dist_sum
    dist_max
    dist_min
    dist_op
    dist_statistics_scalar

Set up distributed training
---------------------------

.. card::
    :class-header: sd-bg-success sd-text-white
    :class-card: sd-outline-success  sd-rounded-1

    Documentation
    ^^^

    .. autofunction:: setup_distributed
    .. autofunction:: get_rank
    .. autofunction:: world_size
    .. autofunction:: fork

Tensor Operations
-----------------

.. card::
    :class-header: sd-bg-success sd-text-white
    :class-card: sd-outline-success  sd-rounded-1

    Documentation
    ^^^

    .. autofunction:: avg_tensor
    .. autofunction:: avg_grads
    .. autofunction:: sync_params
    .. autofunction:: avg_params

Distributed Operations
----------------------

.. card::
    :class-header: sd-bg-success sd-text-white
    :class-card: sd-outline-success  sd-rounded-1

    Documentation
    ^^^

    .. autofunction:: dist_avg
    .. autofunction:: dist_sum
    .. autofunction:: dist_max
    .. autofunction:: dist_min
    .. autofunction:: dist_op
    .. autofunction:: dist_statistics_scalar
