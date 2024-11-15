import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from examples.utils import *
from monviso import VI

np.random.seed(2024)

# Problem data
n = 5
q = np.random.uniform(-10, 10, size=n)
M = random_positive_definite_matrix(-10, 10, n)

# Define the mapping and constraints' set
F = lambda x: -(q + M @ x)
L = np.linalg.norm(M, 2)
x = cp.Variable(n)
S = [x >= 0, -q @ x - cp.quad_form(x, M) >= 0]

# Define the VI and the initial(s) points
lcp = VI(F, S=S)
x0 = []
for _ in range(2):
    prob = cp.Problem(cp.Minimize(np.random.rand(n) @ x), constraints=S).solve()
    x0.append(x.value)

# Solve the VI using the available algorithms
max_iter = 200
for algorithm, params in cases(x0, L, excluded={"fogda", "cfogda"}).items():
    print(f"Using: {algorithm}")
    sol = lcp.solution(
        algorithm,
        params,
        max_iter,
        log_path=f"examples/logs/linear-complementarity/{algorithm}.log",
    )

plot_results(
    "examples/logs/linear-complementarity",
    "examples/figs/linear-complementarity.pdf",
    r"$\|\mathbf{x}_k \! - \! \text{proj}_{\mathcal{S}}(\mathbf{x}_k \! - \! F(\mathbf{x}_k))\|$",
)
