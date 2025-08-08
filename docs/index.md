# monviso documentation

## Installation

Install `monviso` directly from its GitHub repository using `pip`:
```bash
pip install git+https://github.com/nicomignoni/monviso.git@master
```

## Getting started
If you're already familiar with [variational inequalities](https://en.wikipedia.org/wiki/Variational_inequality) (VI), hop to the [Quickstart](examples/quickstart-nb.ipynb) for an overview on how to use `monviso`. Otherwise, the following provides an (extremely) essential introduction to VIs and to the nomenclature that will be used in the rest of the documentation. 

Given a [vector mapping](https://en.wikipedia.org/wiki/Vector-valued_function) $\mathbf{F} : \mathbb{R}^n \to \mathbb{R}^n$ and a scalar [convex](https://en.wikipedia.org/wiki/Convex_function) (possibly [non-smooth](https://en.wikipedia.org/wiki/Smoothness)) function $g : \mathbb{R}^n \to \mathbb{R}$, solving a VI consists of solving the following 

\begin{equation}
    \label{eq:vi_base}
    \text{find } \mathbf{x}^* \in \mathbb{R}^n \text{ such that } (\mathbf{x} - \mathbf{x}^*)^\top \mathbf{F}(\mathbf{x}^*) - g(\mathbf{x}) - g(\mathbf{x}^*) \geq 0, \quad \forall \mathbf{x} \in \mathbb{R}^n.
\end{equation}

It turns out that a lot of problems in optimal control, optimization, machine learning, game theory, finance, and much more, boil down to solving some instance of $\eqref{eq:vi_base}$. Such a problem is usually solved through iterative methods: one constructs an algorithm that produces a $\mathbf{x}_k$ at each iteration $k$ such that $\mathbf{x}_k \to \mathbf{x}^*$ when $k \to \infty$. 

What `monviso` does is providing a convenient way for accessing and using these iterative methods for solving an instance of $\eqref{eq:vi_base}$. The iterative methods that are currently implemented are listed in the [API documentation](api.md#iterative-methods). Since many of these algorithms rely on evaluating a [proximal operator](https://en.wikipedia.org/wiki/Proximal_operator) (or a [projection step](https://en.wikipedia.org/wiki/Proximal_operator#Properties)), `monviso` builds on top of [`cvxpy`](https://www.cvxpy.org/), a package for modelling and solving [convex optimization problems](https://en.wikipedia.org/wiki/Convex_optimization).  

## Cite as
If you used `monviso` in your project or research, please consider citing the related [paper]():
```
@inproceedings{mignoni2025monviso,
  title={monviso: A Python Package for Solving Monotone Variational Inequalities},
  author={Mignoni, Nicola and Baghbadorani, Reza Rahimi and Carli, Raffaele and Esfahani, Peyman Mohajerin and Dotoli, Mariagrazia and Grammatico, Sergio},
  booktitle={2025 European Control Conference (ECC)},
  year={2025}
  organization={IEEE}
}
```

## Why "`monviso`"?
It stands for *`mon`otone `v`ariational `i`nequalities `so`lutions*. Initially, `so`- stood for "solver", but it was a bit on the nose. After all, `monviso` is *just* a collection of functions. Monviso is also a [mountain in Italy](https://en.wikipedia.org/wiki/Monte_Viso). After a couple of *iterations*, the name in its current form was suggested by [Sergio Grammatico](https://sites.google.com/site/grammaticosergio/home?authuser=0).    

