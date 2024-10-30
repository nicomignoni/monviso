Feasibility Problem
-------------------

Let us consider :math:`M` balls in :math:`\mathbb{R}^n`, where the :math:`i`-th ball of radius :math:`r_i > 0` centered in :math:`\mathbf{c}_i \in \mathbb{R}^n` is given by :math:`\mathcal{B}_i(\mathbf{c}_i, r_i) \subset \mathbb{R}^n`. We are intereseted in finding a point belonging to their itersection, i.e., we want to solve the following

.. math::
    :label: intersection

    \text{find} \ \mathbf{x} \ \text{subject to} \ \mathbf{x} \in \bigcap_{i = 1}^M \mathcal{B}_i(\mathbf{c}_i, r_i)

It is straighforward to verify that the projection of a point onto :math:`\mathcal{B}_i(\mathbf{c}_i,r_i)` is evaluates as

.. math::
    :label: projection

    \mathsf{P}_i(\mathbf{x}) := 
    \text{proj}_{\mathcal{B}_i(\mathbf{c}_i,r_i)}(\mathbf{x}) = 
    \begin{cases}
        \displaystyle r_i\frac{\mathbf{x} - 
        \mathbf{c}_i}{\|\mathbf{x} - \mathbf{c}_i\|} & \text{if} \ \|\mathbf{x} - \mathbf{c}_i\| > r_i \\
        x & \text{otherwise}
    \end{cases}

Due to the non-expansiveness of the projection in :eq:`projection`, one can find a solution for :eq:`intersection` as the fixed point of the following iterate 

.. math::
    :label: krasnoselskii-mann
    
    \mathbf{x}_{k+1} = \mathsf{T}(\mathbf{x}_k) = \frac{1}{M}\sum_{i = 1}^M\mathsf{P}_i(\mathbf{x}_k)

which result from the well-known Krasnoselskii-Mann iterate. By letting :math:`F = \mathsf{I} - \mathsf{T}`, where :math:`\mathsf{I}` denotes the identity operator, the fixed point for :eq:`krasnoselskii-mann` can be treated as the canonical VI [1]_. 

.. literalinclude:: ../../examples/feasibility-problem.py
   :language: python

References
^^^^^^^^^^
.. [1] Bauschke, H. H., & Borwein, J. M. (1996). On projection algorithms for solving convex feasibility problems. SIAM review, 38(3), 367-426.