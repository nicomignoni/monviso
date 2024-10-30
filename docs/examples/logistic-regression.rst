Sparse logistic regression
--------------------------

Consider a dataset of :math:`M` rows and :math:`N` columns, so that :math:`\mathbf{A} = \text{col}(\mathbf{a}^\top_i)_{i =1}^M \in \mathbb{R}^{M \times N}` is the dataset matrix, and :math:`\mathbf{a}_i \in \mathbb{R}^{N}` is the :math:`i`-th features vector for the :math:`i`-th dataset row. Moreover, let :math:`\mathbf{b} \in \mathbb{R}^M` be the target vector, so that :math:`b_i \in \{-1,1\}` is the (binary) ground truth for the :math:`i`-th data entry. 
The sparse logistic regression consists of finding the weight vector :math:`\mathbf{x} \in \mathbb{R}^N` that minimizes the following loss function 

.. math::
    :label: regression
    :nowrap:
    
    \begin{align}
        f(\mathbf{x}) := \sum_{i = 1}^M \log\left(1 + \frac{1}{\exp(b_i \mathbf{a}^\top_i \mathbf{x})} \right) + \gamma \|\mathbf{x}\|_1
        \\ = \underbrace{\mathbf{1}^\top_M \log(1 + \exp(-\mathbf{b} \odot \mathbf{A} \mathbf{x}))}_{=:s(\mathbf{x})} + \underbrace{\gamma \|\mathbf{x}\|_1}_{=:g(\mathbf{x})}
    \end{align}

where :math:`\gamma \in \mathbb{R}_{> 0}` is the :math:`\ell_1`-regulation strength. The gradient for :math:`s(\cdot)`, :math:`\nabla s_\mathbf{x}(\mathbf{x})`, is calculated as

.. math::
    F(\mathbf{x}) = \nabla s_\mathbf{x}(\mathbf{x}) = -\frac{\mathbf{A}^\top \odot (\mathbf{1}_N \otimes \mathbf{b}^\top) \odot \exp(-\mathbf{b} \odot \mathbf{A} \mathbf{x})}{1 + \exp(-\mathbf{b} \odot \mathbf{A} \mathbf{x})} \mathbf{1}_M


The problem of finding the minimizer for :eq:`regression` can be cast as a canonical VI, with :math:`F(\mathbf{x}) := \nabla s_\mathbf{x}(\mathbf{x})` [1]_. 

.. literalinclude:: ../../examples/logistic-regression.py
   :language: python

References
^^^^^^^^^^
.. [1] Mishchenko, K. (2023). Regularized Newton method with global convergence. SIAM Journal on Optimization, 33(3), 1440-1462.
