import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

I = np.eye
e = lambda k,n: I(n)[:,k]

GOLDEN_RATIO = 0.5*(np.sqrt(5) + 1)
ALGORITHM_NAMES = (
    "pg", "eg", "popov", "fbf", "frb", "prg", "eag", "arg", 
    "fogda", "cfogda", "graal", "agraal", "hgraal_1", "hgraal_2"
)
TAB20 = plt.get_cmap('tab20')
COLORS = {name: TAB20.colors[i] for i,name in enumerate(ALGORITHM_NAMES)}

def random_positive_definite_matrix(upper_bound, lower_bound, size):
    M = np.random.uniform(upper_bound, lower_bound, size=(size,size))
    M = (M + M.T)/2 # Make symmetric
    # Make positive semidefinite (through Gershgorin circle theorem)
    centers = np.diag(M)
    M += I(size)*(np.abs(M[np.argmin(centers),:]).sum() + np.abs(centers.min()))
    return M
   
def cases(x0, L, z=None, only={}, excluded={}):
    cases_list = {
        "pg": {"x": x0[0], "step_size": 2/L**2},
        "eg": {"x": x0[0], "step_size": 1/L},
        "popov": {"x": x0[0], "y": x0[1], "step_size": 1/(2*L)},
        "fbf": {"x": x0[0], "step_size": 1/L},
        "frb": {"x_current": x0[0], "x_previous": x0[1], "step_size": 1/(2*L)}, 
        "prg": {"x_current": x0[0], "x_previous": x0[1], "step_size": (np.sqrt(2)-1)/L},
        "eag": {"x": x0[0], "step_size": (np.sqrt(2) - 1)/L},
        "arg": {"x_current": x0[0], "x_previous": x0[1], "step_size": 1/(np.sqrt(3)*L)}, 
        "fogda": {"x_current": x0[0], "x_previous": x0[1], "y": x0[0], "step_size": 1/(4*L)},
        "cfogda": {"x_current": x0[0], "x_previous": x0[1], "y": x0[0], "z": z, "step_size": (np.sqrt(2)-1)/L},
        "graal": {"x": x0[0], "y": x0[1], "step_size": GOLDEN_RATIO/(2*L)},
        "agraal": {"x_current": x0[1], "x_previous": x0[0], "step_size": GOLDEN_RATIO/(2*L)},
        "hgraal_1": {"x_current": x0[1], "x_previous": x0[0], "step_size": GOLDEN_RATIO/(2*L)},
        "hgraal_2": {"x_current": x0[1], "x_previous": x0[0], "step_size": GOLDEN_RATIO/(2*L)}
    }

    only = cases_list.keys() if not only else only
    return {
        algorithm_name: cases_list[algorithm_name] for algorithm_name in only \
        if algorithm_name not in excluded
    }

def plot_results(log_path, fig_path, ylabel):
    fig, ax = plt.subplots(figsize=(3.5/2, 1.8), layout="constrained")
    # plt.yscale("log")
    for log_file in glob.glob(f"{log_path}/*.log"):
        algorithm_name = Path(log_file).stem
        eval_func_value = np.genfromtxt(
            f"{log_path}/{algorithm_name}.log", delimiter=",", skip_header=1, 
            usecols=1
        )
        ax.plot(eval_func_value, lw=0.8, label=algorithm_name, color=COLORS[algorithm_name])
        
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.grid(True, alpha=0.2)
    ax.set_xlabel("Iterations ($k$)")
    ax.set_ylabel(ylabel)   
    plt.savefig(fig_path)
    plt.show(block=False)

# Plot settings
latex_preamble = [
    r'\usepackage{amsfonts}',
    r'\usepackage{amssymb}',
    r'\usepackage{amsmath}',
]

plt.rcParams.update({
    "grid.alpha": 0.5,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "legend.frameon": False,
    "ytick.labelsize": 6,
    "xtick.labelsize": 6,
    "font.size": 8,
    "font.family": "sans",
    "font.sans-serif": ["Computer Modern Roman"],
    "text.usetex": True,
    "text.latex.preamble": "".join(latex_preamble)
})

# Plot the legend only
fig, ax = plt.subplots(figsize=(2.1, 0.05))
for name, color in COLORS.items():
    ax.plot(0, 0, label=name, color=color)

plt.axis('off') 
plt.legend(
    bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
    mode="expand", borderaxespad=0, ncol=2,
)

plt.savefig("examples/figs/legend.pdf", bbox_inches="tight")