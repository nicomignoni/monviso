Skew-symmetric operator
-----------------------

A simple example of monotone operator that is not (even locally) strongly monotone is the skewed-symmetric operator, :math:`F : \mathbb{R}^{MN} \to \mathbb{R}^{MN}`, which is described as follows

.. math::
    F(\mathbf{x}) = \begin{bmatrix} \mathbf{A}_1 & & \\ & \ddots & \\ & & \mathbf{A}_M \end{bmatrix} \mathbf{x}

for a given :math:`M \in \mathbb{N}`, where :math:`\mathbf{A}_i = \text{tril}(\mathbf{B}_i) - \text{triu}(\mathbf{B}_i)`, for some arbitrary :math:`0 \preceq \mathbf{B}_i \in \mathbb{R}^{N \times N}`, for all :math:`i = 1, \dots, M`. 

.. literalinclude:: ../../examples/skew-symmetric.py
   :language: python

References
^^^^^^^^^^

.. [1] Bauschke, H. H., & Combettes, P. L. Convex Analysis and Monotone Operator Theory in Hilbert Spaces.