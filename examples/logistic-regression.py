import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from examples.utils import *
from monviso import VI

np.random.seed(2024)

N, M = 500, 200

# Train matrix, target vector, and regularization strength
A = np.random.normal(size=(M, N))
b = np.random.choice([-1, 1], size=M)
gamma = 0.005 * np.linalg.norm(A.T @ b, np.inf)

# VI mapping
F = lambda x: -np.sum(
    (A.T * np.tile(b, (N, 1))) * np.exp(-b * (A @ x)) / (1 + np.exp(-b * (A @ x))),
    axis=1,
)
x = cp.Variable(N)
g = gamma * cp.norm(x, 1)
L = 1.5

# Define the VI problem
slr = VI(F, g)

# Initial points
x0 = [np.random.rand(N) for _ in range(2)]

# Solve the VI using the available algorithms
max_iter = 200
for algorithm, params in cases(x0, L, excluded={"pg", "cfogda"}).items():
    print(f"Using: {algorithm}")
    sol = slr.solution(
        algorithm,
        params,
        max_iter,
        log_path=f"examples/logs/logistic-regression/{algorithm}.log",
    )

plot_results(
    "examples/logs/logistic-regression",
    "examples/figs/logistic-regression.pdf",
    r"$\|F(\mathbf{x}_k))\|$",
)
