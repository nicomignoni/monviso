Linear-Quadratic Dynamic Game
-----------------------------

As shown in [Proposition 2] [1]_, the receding horizon open-loop Nash equilibria (NE) can be reformulated as a non-symmetric variational inequality. Specifically, consider a set of agents :math:`\mathcal{N} = \{1,\dots,N\}` characterizing a state vector :math:`\mathbf{x}[t] \in \mathbb{R}^n`, whose (linear) dynamics is described as

.. math::
    \mathbf{x}[t+1] = \mathbf{A}\mathbf{x}[t] + \sum_{i \in \mathcal{N}} \mathbf{B}_i \mathbf{u}_i[t]

for :math:`t = 1, \dots, T`. Each agent :math:`i` selfishly tries to choose :math:`\mathbf{u}_i[t] \in \mathbb{R}^m` in order to minimize the following cost function

.. math::
    J_i(\mathbf{u}_i|\mathbf{x}_0, \mathbf{u}_{-i}) = \frac{1}{2}\sum_{t=0}^{T-1} \|\mathbf{x}[t|\mathbf{x}_0, \mathbf{u}]\|^2_{\mathbf{Q}_i} + \|\mathbf{u}_i[t] \|^2_{\mathbf{R}_i}

for some :math:`0 \preceq \mathbf{Q}_i \in \mathbb{R}^{n \times n}` and :math:`0 \prec \mathbf{R}_i \in \mathbb{R}^{m \times m}`, with :math:`\mathbf{u}_{-i} = \text{col}(\mathbf{u}_j)_{j \in \mathcal{N}\setminus \{i\}}` and :math:`\mathbf{u}_j = \text{col}(\mathbf{u}_j[t])_{t=1}^T`. Moreover, :math:`\mathbf{u} = \text{col}(\mathbf{u}_i)_{i \in \mathcal{N}}`. The set of feasible inputs, for each agent :math:`i \in \mathcal{N}`, is :math:`\mathcal{U}_i(\mathbf{x}_0,\mathbf{u}_{-i}) := \{\mathbf{u}_i \in \mathbb{R}^{mT} : \mathbf{u}_i[t] \in \mathcal{U}_i(\mathbf{u}_{-i}[t]), \ \forall t = 0,\dots,T-1; \ \mathbf{x}[t|\mathbf{x}_0, \mathbf{u}] \in \mathcal{X}, \ \forall t = 1,\dots,T\}`, where :math:`\mathcal{X} \in \mathbb{R}^n` is the set of feasible system states. Finally, :math:`\mathcal{U}(\mathbf{x}_0) = \{\mathbf{u} \in \mathbb{R}^{mTN}: \mathbf{u}_i \in \mathcal{U}(\mathbf{x}_0,\mathbf{u}_{-i}), \ \forall i \in \mathcal{N}\}`. 
Following [Definition 1] [1]_, the sequence of input :math:`\mathbf{u}^*_i \in \mathcal{U}_i(\mathbf{x}_0,\mathbf{u}_{-i})`, for all :math:`i \in \mathcal{N}`, characterizes an open-loop NE iff

.. math::
    J(\mathbf{u}^*_i|\mathbf{x}_0,\mathbf{u}^*_{-i}) \leq \inf_{\mathbf{u}_i \in \mathcal{U}_i(\mathbf{x}_0, \mathbf{u}^*_{-i})}\left\{ J(\mathbf{u}^*_i|\mathbf{x}_0,\mathbf{u}_{-i}) \right\}

which is satisfied by the fixed-point of the best response mapping of each agent, defined as

.. math::
    :label: best_response

    \mathbf{u}^*_i = \underset{{\mathbf{u}_i \in \mathcal{U}(\mathbf{x}_0,\mathbf{u}^*_{-i})}}{\text{argmin}} J_i(\mathbf{u}_i|\mathbf{x}_0, \mathbf{u}^*_{-i}), \quad \forall i \in \mathcal{N}

Proposition 2 in \cite{benenati2024linear} states that any solution of \eqref{eq:vi_base} is a solution for \eqref{eq:best_response} when :math:`\mathcal{S} = \mathcal{U}(\mathbf{x}_0)` and :math:`F : \mathbb{R}^{mTN} \to \mathbb{R}^{mTN}$`, defined as

.. math::
    F(\mathbf{u}) = \text{col}(\mathbf{G}^\top_i \bar{\mathbf{Q}}_i)_{i \in \mathcal{N}} (\text{row}(\mathbf{G}_i)_{i \in \mathcal{N}}\mathbf{u} + \mathbf{H} \mathbf{x}_0) +
    \text{blkdiag}(\mathbf{I}_T \otimes \mathbf{R}_i)_{i \in \mathcal{N}} \mathbf{u}

where, for all :math:`i \in \mathcal{N}`, :math:`\bar{\mathbf{Q}}_i = \text{blkdiag}(\mathbf{I}_{T-1} \otimes \mathbf{Q}_i, \mathbf{P}_i)`, :math:`\mathbf{G}_i = \mathbf{e}^\top_{1,T} \otimes \text{col}(\mathbf{A}^t_i \mathbf{B}_i)_{t=0}^{T-1} + \mathbf{I}_T \otimes \mathbf{B}_i` and :math:`\mathbf{H} = \text{col}(\mathbf{A}^t)_{t = 1}^T`. Matrix :math:`\mathbf{P}_i` results from the open-loop NE feedback synthesis as discussed in [Equation 6] [1]_.

References
^^^^^^^^^^
.. [1] Benenati, E., & Grammatico, S. (2024). Linear-Quadratic Dynamic Games as Receding-Horizon Variational Inequalities. arXiv preprint arXiv:2408.15703.