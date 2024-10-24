.. monviso documentation master file, created by
   sphinx-quickstart on Tue Oct  8 19:25:41 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ``monviso``!
=================================

.. meta::
   :description: An open source Python package for solving monotone variational
                 inqualities.
   :keywords: monotone variational inequalities, open source, software,

.. currentmodule:: monviso.core.VI

.. toctree::
   :maxdepth: 2

.. autosummary::
   :toctree: functions
      pg
      eg
      popov
      fbf
      frb
      prg
      eag
      arg 
      fogda
      cfogda
      graal
      agraal
      hgraal_1
      hgraal_2
      solution


Quickstart
----------

Let :math:`F(\mathbf{x}) = \mathbf{H} \mathbf{x}` for some 
:math:`\mathbf{H} \succ 0`, :math:`g(\mathbf{x}) = \|\mathbf{x}\|_1`, 
and :math:`\mathcal{S} = \{\mathbf{x} \in \mathbb{R}^n : \mathbf{A} \mathbf{x} 
\leq \mathbf{b}\}`, for some :math:`\mathbf{A} \in \mathbb{R}^{m \times n}` and 
:math:`\mathbf{b} \in \mathbb{R}^n`. It is straightforward to verify that 
:math:`F(\cdot)` is strongly monotone with :math:`\mu = \lambda_{\min}(\mathbf{H})` 
and Lipschitz with :math:`L = \|\mathbf{H}\|_2`. The solution of the VI 
in can be implemented using ``monviso`` as follows

.. code:: python

   import numpy as np
   import cvxpy as cp

   import matplotlib.pyplot as plt

   from monviso import VI

   np.random.seed(2024)

   # Create the problem data
   n, m = 30, 40
   H = np.random.uniform(2,10, size=(n,n))
   A = np.random.uniform(45,50, size=(m,n))
   b = np.random.uniform(3,7, size=(m,))

   # Make H positive definite (using Gershgorin Circle Theorem)
   H = (H + H.T)/2
   centers = np.diag(H)
   H += np.eye(n)*(H[np.argmin(centers),:].sum() + np.abs(centers.min()))

   # Lipschitz and strong monotonicity constants
   mu = np.linalg.eigvals(H).min()
   L = np.linalg.norm(H,2)

   # Define F, g, and S
   F = lambda x: H@x

   x = cp.Variable(n)
   g = cp.norm(x)
   S = [A@x <= b]

   # Define and solve the VI
   vi = VI(F, g, S)

   x0 = np.random.uniform(4, 5, n)
   algorithm_params = {"x": x0, "step_size": 2*mu/L**2}
   sol = vi.solution("pg", algorithm_params, max_iters=25, eval_tol=-np.inf, log_path="result.log")

By checking the logs collected in ``result.log``, we can plot the residual at 
each iteration

.. code:: python

   # Plot the residual 
   residual = np.genfromtxt("result.log", delimiter=",", skip_header=1, usecols=1)
   fig, ax = plt.subplots(figsize=(6.4, 3))
   ax.plot(residual)

   ax.grid(True, alpha=0.2)
   ax.set_xlabel("Iterations ($k$)")
   ax.set_ylabel(
      r"$\|\mathbf{x}_k - \text{proj}_{\mathcal{S}}(\mathbf{x}_k - F(\mathbf{x}_k))\|$"
   )

   plt.savefig("examples/figs/quickstart.svg", bbox_inches="tight")
   plt.show()

.. image:: ../examples/figs/quickstart.svg

Algorithms
----------

The following algorithms are implemented in ``monviso``:

* :func:`Proximal Gradient <pg>`
* :func:`Extra-gradient <eg>`
* :func:`Popov's Method <popov>`
* :func:`Forward-backward-forward <fbf>`
* :func:`Forward-reflected-backward <frb>`
* :func:`Proximal Reflected Gradient <prg>`
* :func:`Extra Anchored Gradient <eag>`
* :func:`Accelerated Reflected Gradient <arg>`
* :func:`(Explicit) Fast Optimistic Gradient Descent-Ascent <fogda>`
* :func:`Constrained Fast Optimistic Gradient Descent-Ascent <cfogda>`
* :func:`Golden Ratio Algorithm <graal>`
* :func:`Adaptive Golden Ratio Algorithm <agraal>`
* :func:`Hybrid Golden Ratio Algorithm I <hgraal_1>`
* :func:`Hybrid Golden Ratio Algorithm II <hgraal_2>` 

Examples
--------

The following examples, related to control, optimization, game theory, finance, 
and machine learning problems, can be reduced to a VI and solved through ``monviso``.

* :doc:`Feasibility problem <examples/feasibility-problem>`
* :doc:`Zero-sum game <examples/zero-sum-game>`
* :doc:`Sparse logistic regression <examples/logistic-regression>`
* :doc:`Skew-symmetric operator <examples/skew-symmetric>`
* :doc:`Markov decision process <examples/markov-decision-process>`
* :doc:`Linear complementarity problem <examples/linear-complementarity>`
* :doc:`Linear-Quadratic game <examples/linear-quadratic-game>`