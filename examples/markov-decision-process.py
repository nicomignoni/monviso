import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from .utils import *
from monviso import VI

np.random.seed(2024)

# Number of states and actions
num_X, num_A = 20, 10

# Discount factor
gamma = 0.8

# Transition probabilities
P = np.random.rand(num_X, num_A, num_X)
P /= P.sum(2, keepdims=True)

# Reward
R = np.random.rand(num_X, num_X)

# Bellman operator (as fixed point) and VI mapping
T = lambda v: np.einsum("ijk,ik -> ij", P, R + gamma * v[None, :]).max(1)
F = lambda x: x - T(x)
L = 3

# Create the VI and the initial solution(s)
mdp = VI(num_X, F)
x0 = [np.random.rand(num_X) for _ in range(2)]

# Solve the VI using the available algorithms
max_iter = 200
for algorithm, params in cases(x0, L, excluded={"pg", "cfogda"}).items():
    print(f"Using: {algorithm}")
    sol = mdp.solution(
        algorithm,
        params,
        max_iter,
        eval_func=lambda x: np.linalg.norm(F(x), 2),
        log_path=f"examples/logs/markov-decision-process/{algorithm}.log",
    )

plot_results(
    "examples/logs/markov-decision-process",
    "examples/figs/markov-decision-process.pdf",
    r"$\|F(\mathbf{x}_k))\|$",
)
