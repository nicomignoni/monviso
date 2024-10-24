Two Players Zero-Sum Game
-------------------------

Many example of non-cooperative behavior between two adversarial agents can be modelled through zero-sum games. Let us consider vectors :math:`\mathbf{x}_i \in \Delta_i` as the decision variable of the :math:`i`-th player, with :math:`i \in \{1,2\}`, where :math:`\Delta_i \subset \mathbb{R}^{n_i}` is the simplex constraints set defined as :math:`\Delta_i := \{\mathbf{x} \in \mathbb{R}^{n_i} : \mathbf{1}^\top \mathbf{x} = 1\}`, for all :math:`i \in \{1,2\}`. Let :math:`\mathbf{x} := \text{col}(\mathbf{x}_i)_{i = 1}^2`. The players try to solve the following problem

.. math::
    \min_{\mathbf{x}_1 \in \Delta_1} \max_{\mathbf{x}_2 \in \Delta_2} \Phi(\mathbf{x}_1, \mathbf{x}_2)

whose (Nash) equilibrium solution is achieved for :math:`\mathbf{x}^*` satisfying the following

.. math::
    :label: saddle

    \Phi(\mathbf{x}^*_1, \mathbf{x}_2) \leq \Phi(\mathbf{x}^*_1, \mathbf{x}^*_2) \leq \Phi(\mathbf{x}_1, \mathbf{x}^*_2), \quad \forall \mathbf{x} \in \Delta_1 \times \Delta_2

For the sake of simplicity, we consider :math:`\Phi(\mathbf{x}_1, \mathbf{x}_2) := \mathbf{x}^\top_1 \mathbf{H} \mathbf{x}_2`, for some :math:`\mathbf{H} \in \mathbb{R}^{n_1 \times n_2}`. Doing so, the equilibrium condition in the previous equation can be written as a VI, with the mapping :math:`F : \mathbb{R}^{n_1 + n_2} \to \mathbb{R}^{n_1 + n_2}` defined as

.. math::
    F(\mathbf{x}) = \begin{bmatrix} \mathbf{H} \mathbf{x}_1 \\ -\mathbf{H}^\top \mathbf{x}_2 \end{bmatrix} = \begin{bmatrix} & \mathbf{H} \\ -\mathbf{H}^\top & \end{bmatrix} \mathbf{x}

and :math:`\mathcal{S} = \Delta_1 \times \Delta_2`

.. literalinclude:: ../../examples/zero-sum-game.py
   :language: python

References
^^^^^^^^^^
.. [1] Lemke, C. E., & Howson, Jr, J. T. (1964). Equilibrium points of bimatrix games. Journal of the Society for industrial and Applied Mathematics, 12(2), 413-423.