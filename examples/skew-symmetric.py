import numpy as np
import scipy as sp
import cvxpy as cp
import matplotlib.pyplot as plt

from .utils import *
from monviso import VI

np.random.rand(2024)

M, N =  20, 10

# Create the problem variables
Bs = [random_positive_definite_matrix(0, 1, N) for _ in range(M)]
A = sp.linalg.block_diag(*[np.tril(B) - np.triu(B) for B in Bs])

F = lambda x: A@x
L = np.linalg.norm(A, 2)

# Create the VI and the initial solution(s)
sso = VI(F, n=N*M)
x0 = [np.random.rand(N*M) for _ in range(2)]

# Solve the VI using the available algorithms
max_iter = 200
for algorithm, params in cases(x0, L, excluded={"pg", "cfogda"}).items():
    print(f"Using: {algorithm}")
    sol = sso.solution(
        algorithm, params, max_iter, eval_func=lambda x: np.linalg.norm(F(x), 2),
        log_path=f"examples/logs/skew-symmetric/{algorithm}.log"
    )

plot_results(
    "examples/logs/skew-symmetric",
    "examples/figs/skew-symmetric.pdf",
    r"$\|F(\mathbf{x}_k))\|$"
)