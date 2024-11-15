import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from examples.utils import *
from monviso import VI

np.random.seed(2024)

n1, n2 = 50, 50

# Game matrix
H = np.random.rand(n1, n2)
H_block = np.block([[np.zeros((n1, n2)), H], [-H.T, np.zeros((n1, n2))]])

# VI operators with Liptshitz constant
F = lambda x: H_block @ x
L = np.linalg.norm(H_block, 2)

# Simplex constraints' set
x = cp.Variable(n1 + n2)
S = [cp.sum(x[:n1]) == 1, cp.sum(x[n1:]) == 1]

# Define the two-players zero sum game as a Variational Inequality
tpzsg = VI(F, S=S)

# Create two initial (feasible) points
x0 = []
for i in range(2):
    x0.append(np.random.rand(n1 + n2))
    x0[i][:n1] /= x0[i][:n1].sum()
    x0[i][n1:] /= x0[i][n1:].sum()

# Solve the VI using the available algorithms
max_iter = 200
for algorithm, params in cases(x0, L, excluded={"pg", "fogda", "cfogda"}).items():
    print(f"Using: {algorithm}")
    sol = tpzsg.solution(
        algorithm,
        params,
        max_iter,
        log_path=f"examples/logs/zero-sum-game/{algorithm}.log",
    )

plot_results(
    "examples/logs/zero-sum-game",
    "examples/figs/zero-sum-game.pdf",
    r"$\|\mathbf{x}_k \! - \! \text{proj}_{\mathcal{S}}(\mathbf{x}_k \! - \! F(\mathbf{x}_k))\|$",
)
