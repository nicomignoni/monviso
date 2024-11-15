import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from examples.utils import *
from monviso import VI

np.random.seed(2024)

# Problem data
n, M = 1000, 2000
c = np.random.normal(0, 100, size=(M, n))
r = 1 - np.linalg.norm(c, axis=0)

# Projection operator and VI mapping, with its Liptshitz constant
P = lambda x: np.where(
    np.linalg.norm(x - c) > r, r * (x - c) / np.linalg.norm(x - c, axis=0), x
)
F = lambda x: x - P(x).mean(axis=0)
L = 10

# Define the VI
fp = VI(F, n=n)

# Initial points
x0 = [np.random.rand(n) for _ in range(2)]

# Solve the VI using the available algorithms
max_iter = 200
for algorithm, params in cases(x0, L, excluded={"pg", "cfogda", "fogda"}).items():
    print(f"Using: {algorithm}")
    sol = fp.solution(
        algorithm,
        params,
        max_iter,
        log_path=f"examples/logs/feasibility/{algorithm}.log",
    )

plot_results(
    "examples/logs/feasibility",
    "examples/figs/feasibility.pdf",
    r"$\|\mathbf{x}_k \! - \! \text{proj}_{\mathcal{S}}(\mathbf{x}_k \! - \! F(\mathbf{x}_k))\|$",
)
