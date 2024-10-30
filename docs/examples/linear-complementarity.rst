Linear Complementarity Problem
------------------------------

A common problem that can be cast to a VI is the linear complementarity problem: given :math:`\mathbf{q} \in \mathbb{R}^n` and :math:`0 \prec \mathbf{M} \in \mathbb{R}^{n \times n}`, one want to solve the following 

.. math::
    :label: complementarity

    \text{find $\mathbf{x} \in \mathbb{R}^n_{\geq 0}$ s.t. $\mathbf{y} = \mathbf{M}\mathbf{x} + \mathbf{q}$, $\mathbf{y}^\top \mathbf{x} = 0$}

By setting :math:`F(\mathbf{x}) = - \mathbf{M}\mathbf{x} - \mathbf{q}` and :math:`\mathcal{S} = \mathbb{R}_{\geq 0}` it can be readily verified that each solution for :math:`(\mathbf{x} - \mathbf{x}^*)^\top F(\mathbf{x}^*) \geq 0` is also a solution for :eq:`complementarity` [1]_. 

.. literalinclude:: ../../examples/linear-complementarity.py
   :language: python

References
^^^^^^^^^^
.. [1] Harker, P. T., & Pang, J. S. (1990). For the linear complementarity problem. Lectures in Applied Mathematics, 26, 265-284.