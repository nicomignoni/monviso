import numpy as np
import cvxpy as cp

import matplotlib.pyplot as plt

from monvi import VI

np.random.seed(2024)

# Create the problem data
n, m = 30, 40
H = np.random.uniform(2,10, size=(n,n))
A = np.random.uniform(4,5, size=(m,n))
b = np.random.uniform(3,7, size=(m,))

# Make H positive definite (using Gershgorin)
H = (H + H.T)/2
H = H + np.diag(((1 - np.eye(n))*H).sum(1))

# Liptshits and strong monotonicity constants
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
sol = vi.solution("pg", algorithm_params, max_iters=30, log_path="result.log")

# Plot the residual 
residual = np.genfromtxt("result.log", delimiter=",", skip_header=1, usecols=1)
fig, ax = plt.subplots(figsize=(6.4, 3))
ax.plot(residual)

ax.grid(True, alpha=0.2)
ax.set_xlabel("Iterations ($k$)")
ax.set_ylabel(
    r"$\|\mathbf{x}_k - \text{proj}_{\mathcal{A}}(\mathbf{x}_k - F(\mathbf{x}_k))\|$"
)

plt.savefig("examples/figs/quickstart.svg", bbox_inches="tight")
plt.show()
