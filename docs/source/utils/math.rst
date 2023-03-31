OmniSafe Math
=============

.. currentmodule:: omnisafe.utils.math

.. autosummary::

    get_transpose
    get_diagonal
    safe_inverse
    discount_cumsum
    conjugate_gradients
    gaussian_kl
    SafeTanhTransformer
    TanhNormal


Tensor Operations
-----------------

.. card::
    :class-header: sd-bg-success sd-text-white
    :class-card: sd-outline-success  sd-rounded-1

    Documentation
    ^^^

    .. autofunction:: get_transpose
    .. autofunction:: get_diagonal
    .. autofunction:: safe_inverse
    .. autofunction:: discount_cumsum
    .. autofunction:: conjugate_gradients


Distribution Operations
-----------------------

.. card::
    :class-header: sd-bg-success sd-text-white
    :class-card: sd-outline-success  sd-rounded-1

    Documentation
    ^^^

    .. autofunction:: gaussian_kl

    .. autoclass:: SafeTanhTransformer
        :members:
        :private-members:

    .. autoclass:: TanhNormal
        :members:
        :private-members:
