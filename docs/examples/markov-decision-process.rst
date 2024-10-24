Markov Decision Process
-----------------------

A stationary discrete Markov Decision Process (MDP) is characterized by the tuple :math:`(\mathcal{X},\mathcal{A},\mathbb{P},r,\gamma)`, i) where :math:`\mathcal{X}` is the (finite countable) set of states; ii) :math:`\mathcal{A}` is the (finite countable) set of actions; iii) :math:`P : \mathcal{X} \times \mathcal{A} \times \mathcal{X} \to [0,1]` is the transition probability function, such that :math:`P(x,a,x^+)` is the probability of ending up in state :math:`x^+ \in \mathcal{S}` from state :math:`x \in \mathcal{X}` when taking action :math:`a \in \mathcal{A}`; iv) :math:`r : \mathcal{X} \times \mathcal{X} \to \mathbb{R}` is the reward function, so that :math:`r(x,x^+)` returns the reward for 
transitioning from state :math:`x \in \mathcal{X}` to state :math:`x^+ \in \mathcal{X}`; :math:`\gamma \in \mathbb{R}_{> 0}` is a discount factor. The aim is to find a policy, i.e., a function :math:`\pi : \mathcal{S} \to \mathcal{A}`, returning the best action for any given state. A solution concept for MDP is the *value function*, :math:`v^{\pi} : \mathcal{S} \to \mathbb{R}`, defined as

.. math::
    :label: bellman

    v^{\pi}(x) = \overbrace{\sum_{x^+ \in \mathcal{X}} P(x,\pi(x),x^+) \left( r(x,x^+) + \gamma v(x^+) \right)}^{=:\mathsf{T}(v^{\pi})}

returning the "goodness" of policy :math:`\pi`. The expression in :eq:`bellman` is known as *Bellman equation*, and can be expressed as an operator of :math:`v^{\pi}`, i.e., :math:`\mathsf{T}[v^\pi(s)] =: \mathsf{T}(v^{\pi})`. It can be shown that the value function yielded by the optimal policy, :math:`v^*`, results from the fixed-point problem :math:`v^* = \mathsf{T}(v^*)`. Therefore, the latter can be formulated as a canonical VI, with :math:`F = \mathsf{I} - \mathsf{T}`.

.. literalinclude:: ../../examples/markov-decision-process.py
   :language: python

References
^^^^^^^^^^
.. [1] Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.
