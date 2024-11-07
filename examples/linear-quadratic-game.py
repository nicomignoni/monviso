import numpy as np
import scipy as sp
import cvxpy as cp
import matplotlib.pyplot as plt

from .utils import *
from monviso import VI

np.random.rand(2024)

# State and input sizes, number of agents, and time steps
n, m, N, T = 13, 4, 5, 3

# Problem data
A = np.random.rand(n, n)
B = [np.random.rand(n, m) for _ in range(N)]
Q = [random_positive_definite_matrix(2, 4, n) for _ in range(N)]
R = [random_positive_definite_matrix(1, 2, m) for _ in range(N)]
P = np.random.rand(n, n)
Q_bar = [sp.linalg.block_diag(np.kron(I(T-1), Q[i]), P) for i in range(N)]
G = [
    np.kron(I(T), B[i]) + np.kron(
        e(0,T),
        np.vstack([np.linalg.matrix_power(A,t)@B[i] for t in range(T)]),
    )
    for i in range(N)
]
H = np.vstack([np.linalg.matrix_power(A,t) for t in range(1,T+1)])
x0 = np.random.rand(n)

# Define the mapping
F1 = np.vstack([G[i].T@Q_bar[i] for i in range(N)])
F2 = np.hstack(G)
F3 = sp.linalg.block_diag(*[np.kron(I(T), R[i]) for i in range(N)])
F = lambda u: F1@(F2@u + H@x0) + F3@u
L = np.linalg.norm(F1@F2 + F3, 2) + 1

# Define a constraints set for the collective input
u = cp.Variable(m*T*N)
S = [u >= 0]
              
# Define the VI and the initial(s) points 
lqg = VI(F, S=S)
u0 = [np.random.rand(m*T*N) for _ in range(2)]

# Solve the VI using the available algorithms
max_iter = 200
for algorithm, params in cases(u0, L, excluded={"cfogda", "fogda"}).items():
    print(f"Using: {algorithm}")
    sol = lqg.solution(
        algorithm, params, max_iter,
        log_path=f"examples/logs/linear-quadratic-game/{algorithm}.log"
    )

plot_results(
    "examples/logs/linear-quadratic-game",
    "examples/figs/linear-quadratic-game.pdf",
    r"$\|\mathbf{x}_k \! - \! \text{proj}_{\mathcal{S}}(\mathbf{x}_k \! - \! F(\mathbf{x}_k))\|$"
)