import numpy as np
import scipy as sp
import cvxpy as cp
import matplotlib.pyplot as plt

from .utils import *
from monviso import VI

np.random.rand(2024)

# Number of agents, samples, portfolio size, random vector size
N, K, n, m = 19, 52, 20, 10

A = [np.random.rand(m, n) for _ in range(N)]
b = [np.random.rand(m) for _ in range(N)]
P = [lambda x: A[i]@x + b[i] for i in range(N)]

Q = [random_positive_definite_matrix(0, 10, m) for _ in range(N)]