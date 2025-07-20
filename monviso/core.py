from typing import Callable

import time
import datetime

import numpy as np
import cvxpy as cp

GOLDEN_RATIO = 0.5 * (np.sqrt(5) + 1)

class VI:
    r"""
    Attributes
    ----------
    n : int
        The size of the vector space
    F : callable
        The VI vector mapping, i.e., $\mathbf{F} : \mathbb{R}^n \to \mathbb{R}^n$; a function transforming a ndarray into an other `np.ndarray` of the same size.
    g : callable, optional
        The VI scalar mapping, i.e., $g : \mathbb{R}^n \to \mathbb{R}$; a callable returning a [`cvxpy.Expression`](https://www.cvxpy.org/api_reference/cvxpy.expressions.html#id1)
    S : list of callable, optional
        The constraints set, i.e., $\mathcal{S} \subseteq \mathbb{R}^n$; a list of callables, each returning a [`Constraints`](https://www.cvxpy.org/api_reference/cvxpy.constraints.html#id8)
    """

    def __init__(
            self, 
            n: int, 
            F: Callable, 
            g: Callable | None = None, 
            S: list[Callable] | None = None
        ) -> None:
        self.F = F
        self.g = (lambda _: 0) if g is None else g

        self.y = cp.Variable(n)
        self.param_x = cp.Parameter(self.y.shape)
        self.constraints = [] if S is None else [constraint(self.y) for constraint in S]

        self._prox = cp.Problem(
            cp.Minimize(self.g(self.y) + 0.5 * cp.norm(self.y - self.param_x)),
            self.constraints,
        )
        self._proj = cp.Problem(
            cp.Minimize(0.5 * cp.norm(self.y - self.param_x)), self.constraints
        )

    # Constrained proximal operator
    def prox(self, x: np.ndarray, **cvxpy_solve_params) -> np.ndarray | None:
        r"""
        Given a scalar function $g : \mathbb{R}^n \to \mathbb{R}$ and a constraints set $\mathcal{S} \subseteq \mathbb{R}^n$, the constrained proximal operator is defined as

        $$
        \text{prox}_{g,\mathcal{S}}(\mathbf{x}) = \underset{\mathbf{y} \in \mathcal{S}}{\text{argmin}} \left\{ g(\mathbf{y}) + \frac{1}{2}\|\mathbf{y} - \mathbf{x}\|^2 \right\}
        $$

        Parameters
        ----------
        x : ndarray
            The proximal operator argument point
        **cvxpy_solve_params
            The parameters for the [`cvxpy.Problem.solve`](https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Problem.solve) method.

        Returns
        -------
        ndarray
            The proximal operator resulting point
        """
        self.param_x.value = x
        self._prox.solve(**cvxpy_solve_params)
        return self.y.value

    # Projection operator
    def proj(self, x: np.ndarray, **cvxpy_solve_params) -> np.ndarray | None:
        r"""
        The projection operator of a point $\mathbf{x} \in \mathbb{R}^n$ with respect to set $\mathcal{S} \subseteq \mathbb{R}^n$ returns the closest point to $\mathbf{x}$ 
        that belongs to $\mathcal{S}$, i.e.,

        $$
        \text{proj}_{\mathcal{S}}(\mathbf{x}) = \text{prox}_{0,\mathcal{S}}
        (\mathbf{x}) = \underset{\mathbf{y}
        \in \mathcal{S}}{\text{argmin}} \left\{\frac{1}{2}\|\mathbf{y} -
        \mathbf{x}\|^2 \right\}
        $$

        Parameters
        ----------
        x : ndarray
            The projection operator argument point
        **cvxpy_solve_params
            The parameters for the [`cvxpy.Problem.solve`](https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Problem.solve) method.

        Returns
        -------
        ndarray
            The projected point
        """
        self.param_x.value = x
        self._proj.solve(**cvxpy_solve_params)
        return self.y.value

    def residual(self, x: np.ndarray, **cvxpy_solve_params) -> float:
        """
        Computes the distance between a point and its update projected onto the feasible set.

        Parameters
        ----------
        x : ndarray
            The argument point
        **cvxpy_solve_params
            The parameters for the [`cvxpy.Problem.solve`](https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Problem.solve) method.

        Returns
        -------
        float
            The residual value
        """
        return np.linalg.norm(x - self.prox(x - self.F(x), **cvxpy_solve_params))

    # Proximal Gradient
    def pg(self, x: np.ndarray, step_size: float, **cvxpy_solve_params) -> np.ndarray | None:
        r"""
        Given a constant step-size $\chi > 0$ and an initial vector $\mathbf{x}_0 \in \mathbb{R}^n$, the basic $k$-th iterate of the proximal gradient (PG) algorithm is [^1]:

        $$ \mathbf{x}_{k+1} = \text{prox}_{g,\mathcal{S}}(\mathbf{x}_k - \chi \mathbf{F}(\mathbf{x}_k)) $$

        where $g : \mathbb{R}^n \to \mathbb{R}$ is a scalar convex (possibly non-smooth) function, while $\mathbf{F} : \mathbb{R}^n \to \mathbb{R}^n$ is the VI mapping. 
        Convergence of PG is guaranteed for Lipschitz strongly monotone operators, with monotone constant $\mu > 0$ and Lipschitz constants $L < +\infty$, when $\chi \in (0, 2\mu/L^2)$.

        [^1]: Nemirovskij, A. S., & Yudin, D. B. (1983). Problem complexity
           and method efficiency in optimization.

        Parameters
        ----------
        x : ndarray
            The initial point, corresponding to $\mathbf{x}_0$
        step_size : float
            The steps size value, corresponding to $\chi$
        **cvxpy_solve_params
            The parameters for the [`cvxpy.Problem.solve`](https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Problem.solve) method.

        Yields
        ------
        ndarray
            The iteration's resulting point
        """
        while True:
            x = self.prox(x - step_size * self.F(x), **cvxpy_solve_params)
            yield x

    # Extragradient
    def eg(self, x: np.ndarray, step_size: float, **cvxpy_solve_params) -> np.ndarray | None:
        r"""
        Given a constant step-size $\chi > 0$ and an initial vector 
        $\mathbf{x}_0 \in \mathbb{R}^n$, the $k$-th iterate 
        of the extragradient algorithm (EG) is[^2]:

        $$ 
        \begin{align}
            \mathbf{y}_k &= \text{prox}_{g,\mathcal{S}}(\mathbf{x}_k - 
                \chi \mathbf{F}(\mathbf{x}_k)) \\
            \mathbf{x}_{k+1} &= \text{prox}_{g,\mathcal{S}}(\mathbf{y}_k - 
                \chi \mathbf{F}(\mathbf{x}_k))
        \end{align}
        $$
 
        where $g : \mathbb{R}^n \to \mathbb{R}$ is a scalar convex 
        (possibly non-smooth) function, while $\mathbf{F} : \mathbb{R}^n \to 
        \mathbb{R}^n$ is the VI mapping. The convergence of the EGD algorithm 
        is guaranteed for Lipschitz monotone operators, with Lipschitz constant 
        $L < +\infty$, when $\chi \in \left(0,\frac{1}{L}\right)$.

        [^2]: Korpelevich, G. M. (1976). The extragradient method for finding 
           saddle points and other problems. Matecon, 12, 747-756.

        Parameters
        ----------
        x : ndarray 
            The initial point, corresponding to $\mathbf{x}_0$
        step_size : float
            The step size value, corresponding to $\chi$
        **cvxpy_solve_params
            The parameters for the 
            [`cvxpy.Problem.solve`](https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Problem.solve) 
            method. 

        Yields
        ------
        ndarray
            The iteration's resulting point 
        """
        while True:
            y = self.prox(x - step_size * self.F(x), **cvxpy_solve_params)
            x = self.prox(x - step_size * self.F(y), **cvxpy_solve_params)
            yield x

    # Popov's Method
    def popov(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        step_size: float, 
        **cvxpy_solve_params
    ) -> np.ndarray:
        r"""
        Given a constant step-size $\chi > 0$ and an initial vectors $\mathbf{x}_0,\mathbf{y}_0 \in \mathbb{R}^n$, the $k$-th iterate of Popov's Method (PM) is[^6]:

        $$ 
        \begin{align}
            \mathbf{y}_ {k+1} &= \text{prox}_{g,\mathcal{S}}(\mathbf{x}_k - \chi \mathbf{F}(\mathbf{y}_k)) \\
            \mathbf{x}_{k+1} &= \text{prox}_{g,\mathcal{S}}(\mathbf{y}_{k+1} - \chi \mathbf{F}(\mathbf{x}_k))
        \end{align}
        $$

        where $g : \mathbb{R}^n \to \mathbb{R}$ is a scalar convex (possibly non-smooth) function, while $\mathbf{F} : \mathbb{R}^n \to \mathbb{R}^n$ is the VI mapping. 
        The convergence of PM is guaranteed for Lipschitz monotone operators, with Lipschitz constant $L < +\infty$, when $\chi \in \left(0,\frac{1}{2L}\right)$.

        [^6]: Popov, L.D. A modification of the Arrow-Hurwicz method for search of saddle points. 
        Mathematical Notes of the Academy of Sciences of the USSR 28, 845–848 (1980)

        Parameters
        ----------
        x : ndarray
            The initial point, corresponding to $\mathbf{x}_0$
        y : ndarray
            The initial auxiliary point, corresponding to $\mathbf{y}_0$
        step_size : float
            The step size value, corresponding to $\chi$
        **cvxpy_solve_params
            The parameters for the [`cvxpy.Problem.solve`](https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Problem.solve) method. 

        Yields
        ------
        ndarray
            The iteration's resulting point
        """
        while True:
            y = self.prox(x - step_size * self.F(y), **cvxpy_solve_params)
            x = self.prox(x - step_size * self.F(y), **cvxpy_solve_params)
            yield x

    # Forward-Backward-Forward
    def fbf(self, x: np.ndarray, step_size: float, **cvxpy_solve_params) -> np.ndarray:
        r"""
        Given a constant step-size $\chi > 0$ and an initial vector $\mathbf{x}_0 \in \mathbb{R}^n$, the $k$-th iterate of Forward-Backward-Forward (FBF) algorithm is[^7]:

        $$ 
        \begin{align}
            \mathbf{y}_k &= \text{prox}_{g,\mathcal{S}}(\mathbf{x}_k - 
                \chi \mathbf{F}(\mathbf{x}_k)) \\
            \mathbf{x}_{k+1} &= \mathbf{y}_k - 
                \chi \mathbf{F}(\mathbf{y}_k) + \chi \mathbf{F}(\mathbf{x}_k)
        \end{align}
        $$

        where $g : \mathbb{R}^n \to \mathbb{R}$ is a scalar convex 
        (possibly non-smooth) function, while $\mathbf{F} : \mathbb{R}^n \to 
        \mathbb{R}^n$ is the VI mapping. The convergence of the FBF algorithm 
        is guaranteed for Lipschitz monotone operators, with Lipschitz constant 
        $L < +\infty$, when $\chi \in \left(0,\frac{1}{L}\right)$.

        [^7]: Tseng, P. (2000). A modified forward-backward splitting method 
           for maximal monotone mappings. SIAM Journal on Control and 
           Optimization, 38(2), 431-446.

        Parameters
        ----------
        x : ndarray
            The initial point, corresponding to $\mathbf{x}_0$
        step_size : float
            The step size value, corresponding to $\chi$
        **cvxpy_solve_params
            The parameters for the [`cvxpy.Problem.solve`](https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Problem.solve) method. 

        Yields
        ------
        ndarray
            The iteration's resulting point
        """
        while True:
            y = self.prox(x - step_size * self.F(x), **cvxpy_solve_params)
            x = y - step_size * self.F(y) + step_size * self.F(x)
            yield x

    # Forward-Reflected-Backward
    def frb(
        self,
        x_current: np.ndarray,
        x_previous: np.ndarray,
        step_size: float,
        **cvxpy_solve_params,
    ) -> np.ndarray:
        r"""
        Given a constant step-size $\chi > 0$ and initial vectors
        $\mathbf{x}_1,\mathbf{x}_0 \in \mathbb{R}^n$, the basic
        $k$-th iterate of the Forward-Reflected-Backward (FRB)
        is the following[^8]:

        $$ \mathbf{x}_k = \text{prox}_{g,\mathcal{S}}(\mathbf{x}_k - 2\chi \mathbf{F}(\mathbf{x}_k) + \chi \mathbf{F}(\mathbf{x}_{k-1})) $$

        where $g : \mathbb{R}^n \to \mathbb{R}$ is a scalar convex
        (possibly non-smooth) function, while $\mathbf{F} : \mathbb{R}^n \to
        \mathbb{R}^n$ is the VI mapping. The convergence of the FRB algorithm
        is guaranteed for Lipschitz monotone operators, with Lipschitz constant
        $L < +\infty$, when $\chi \in \left(0,\frac{1}{2L}\right)$.

        [^8]: Malitsky, Y., & Tam, M. K. (2020). A forward-backward splitting
           method for monotone inclusions without cocoercivity. SIAM Journal on
           Optimization, 30(2), 1451-1472.

        Parameters
        ----------
        x_current : ndarray
            The initial point, corresponding to $\mathbf{x}_0$
        x_previous : ndarray
            The initial point, corresponding to $\mathbf{x}_1$
        step_size : float
            The step size value, corresponding to $\chi$
        **cvxpy_solve_params
            The parameters for the
            [`cvxpy.Problem.solve`](https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Problem.solve)
            method.

        Yields
        ------
        ndarray
            The iteration's resulting point
        """
        while True:
            x = self.prox(
                x_current
                - 2 * step_size * self.F(x_current)
                + step_size * self.F(x_previous),
                **cvxpy_solve_params,
            )
            x_previous = x_current
            x_current = x
            yield x

    # Projected Reflected Gradient
    def prg(
        self,
        x_current: np.ndarray,
        x_previous: np.ndarray,
        step_size: float,
        **cvxpy_solve_params,
    ) -> np.ndarray:
        r"""
        Given a constant step-size $\chi > 0$ and initial vectors
        $\mathbf{x}_1,\mathbf{x}_0 \in \mathbb{R}^n$, the basic
        $k$-th iterate of the projected reflected gradient (PRG)
        is the following [^3]:

        $$ \mathbf{x}_{k+1} = \text{prox}_{g,\mathcal{S}}(\mathbf{x}_k - \chi \mathbf{F}(2\mathbf{x}_k - \mathbf{x}_{k-1})) $$

        where $g : \mathbb{R}^n \to \mathbb{R}$ is a scalar convex
        (possibly non-smooth) function, while $\mathbf{F} : \mathbb{R}^n \to
        \mathbb{R}^n$ is the VI mapping.
        The convergence of PRG algorithm is guaranteed for Lipschitz monotone
        operators, with Lipschitz constants $L < +\infty$, when
        $\chi \in (0,(\sqrt{2} - 1)/L)$. Differently from the EGD
        iteration, the PRGD has the advantage of requiring a single
        proximal operator evaluation.

        [^3]: Malitsky, Y. (2015). Projected reflected gradient methods for
           monotone variational inequalities. SIAM Journal on Optimization,
           25(1), 502-520.

        Parameters
        ----------
        x_current : ndarray
            The initial point, corresponding to $\mathbf{x}_0$
        x_previous : ndarray
            The initial point, corresponding to $\mathbf{x}_1$
        step_size : float
            The step size value, corresponding to $\chi$
        **cvxpy_solve_params
            The parameters for the
            [`cvxpy.Problem.solve`](https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Problem.solve)
            method.

        Yields
        ------
        ndarray
            The iteration's resulting point
        """
        while True:
            x = self.prox(
                x_current - step_size * self.F(2 * x_current - x_previous),
                **cvxpy_solve_params,
            )
            x_previous = x_current
            x_current = x
            yield x

    # Extra Anchored Gradient
    def eag(self, x: np.ndarray, step_size: float, **cvxpy_solve_params) -> np.ndarray:
        r"""
        Given a constant step-size $\chi > 0$ and an initial vector 
        $\mathbf{x}_0 \in \mathbb{R}^n$, the $k$-th 
        iterate of extra anchored gradient (EAG) algorithm is [^9]:

        $$
        \begin{align}
            \mathbf{y}_k &= \text{prox}_{g,\mathcal{S}}\left(\mathbf{x}_k - 
                \chi \mathbf{F}(\mathbf{x}_k) + \frac{1}{k+1}(\mathbf{x}_0 - 
                \mathbf{x}_k)\right) \\
            \mathbf{x}_{k+1} &= \text{prox}_{g,\mathcal{S}}\left(\mathbf{x}_k - 
                \chi \mathbf{F}(\mathbf{y}_k) + \frac{1}{k+1}(\mathbf{x}_0 - 
                \mathbf{x}_k)\right)
        \end{align}
        $$

        where $g : \mathbb{R}^n \to \mathbb{R}$ is a scalar convex 
        (possibly non-smooth) function, while $\mathbf{F} : \mathbb{R}^n \to 
        \mathbb{R}^n$ is the VI mapping. The convergence of the EAG algorithm 
        is guaranteed for Lipschitz monotone operators, with Lipschitz constant 
        $L < +\infty$, when $\chi \in \left(0,\frac{1}{\sqrt{3}L}
        \right)$.

        [^9]: Yoon, T., & Ryu, E. K. (2021, July). Accelerated Algorithms for 
           Smooth Convex-Concave Minimax Problems with O (1/k^ 2) Rate on Squared 
           Gradient Norm. In International Conference on Machine Learning (pp. 
           12098-12109). PMLR.

        Parameters
        ----------
        x : ndarray
            The initial point, corresponding to $\mathbf{x}_0$
        step_size : float
            The step size value, corresponding to $\chi$
        **cvxpy_solve_params
            The parameters for the 
            [`cvxpy.Problem.solve`](https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Problem.solve) 
            method. 
        
        Yields
        ------
        ndarray
            The iteration's resulting point
        """
        k = 0
        x0 = x
        while True:
            y = self.prox(
                x - step_size * self.F(x) + (x0 - x) / (k + 1), **cvxpy_solve_params
            )
            x = self.prox(
                x - step_size * self.F(y) + (x0 - x) / (k + 1), **cvxpy_solve_params
            )
            k += 1
            yield x

    # Accelerated Reflected Gradient
    def arg(
        self,
        x_current: np.ndarray,
        x_previous: np.ndarray,
        step_size: float,
        **cvxpy_solve_params,
    ) -> np.ndarray:
        r"""
        Given a constant step-size $\chi > 0$ and initial vectors 
        $\mathbf{x}_1,\mathbf{x}_0 \in \mathbb{R}^n$, the basic 
        $k$-th iterate of the accelerated reflected gradient (ARG) 
        is the following [^10]:

        $$
        \begin{align}
            \mathbf{y}_k &= 2\mathbf{x}_k - \mathbf{x}_{k-1} + \frac{1}{k+1}
            (\mathbf{x}_0 - \mathbf{x}_k) - \frac{1}{k}(\mathbf{x}_k - 
            \mathbf{x}_{k-1}) \\
            \mathbf{x}_{k+1} &= \text{prox}_{g,\mathcal{S}}\left(\mathbf{x}_k - 
                \chi \mathbf{F}(\mathbf{y}_k) + \frac{1}{k+1}(\mathbf{x}_0 - 
                \mathbf{x}_k)\right)
        \end{align}
        $$

        where $g : \mathbb{R}^n \to \mathbb{R}$ is a scalar convex 
        (possibly non-smooth) function, while $\mathbf{F} : \mathbb{R}^n \to 
        \mathbb{R}^n$ is the VI mapping. The convergence of the ARG algorithm 
        is guaranteed for Lipschitz monotone operators, with Lipschitz constant 
        $L < +\infty$, when $\chi \in \left(0,\frac{1}{12L}\right)$.

        [^10]: Cai, Y., & Zheng, W. (2022). Accelerated single-call methods 
           for constrained min-max optimization. arXiv preprint arXiv:2210.03096.

        Parameters
        ----------
        x_current : ndarray
            The initial point, corresponding to $\mathbf{x}_0$
        x_previous : ndarray
            The initial point, corresponding to $\mathbf{x}_1$
        step_size : float
            The step size value, corresponding to $\chi$
        **cvxpy_solve_params
            The parameters for the 
            [`cvxpy.Problem.solve`](https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Problem.solve) 
            method. 

        Yields
        ------
        ndarray
            The iteration's resulting point
        """
        k = 1
        x0 = x_previous
        while True:
            y = (
                2 * x_current
                - x_previous
                + 1 / (k + 1) * (x0 - x_current)
                - 1 / k * (x0 - x_previous)
            )
            x = self.prox(
                x_current - step_size * self.F(y) + 1 / (k + 1) * (x0 - x_current),
                **cvxpy_solve_params,
            )

            x_previous = x_current
            x_current = x
            k += 1
            yield x

    # (Explicit) Fast Optimistic Gradient Descent Ascent
    def fogda(
        self,
        x_current: np.ndarray,
        x_previous: np.ndarray,
        y: np.ndarray,
        step_size: float,
        alpha: float = 2.1,
    ) -> np.ndarray:
        r"""
        Given a constant step-size $\chi > 0$ and initial vectors 
        $\mathbf{x}_1,\mathbf{x}_0,\mathbf{y}_0 \in \mathbb{R}^n$, the 
        basic $k$-th iterate of the explicit fast OGDA (FOGDA) 
        is the following [^11]:

        $$
        \begin{align}
            \mathbf{y}_k &= \mathbf{x}_k + \frac{k}{k+\alpha}(\mathbf{x}_k - 
                \mathbf{x}_{k-1}) - \chi \frac{\alpha}{k+\alpha}
                \mathbf{F}(\mathbf{y}_{k-1}) \\
            \mathbf{x}_{k+1} &= \mathbf{y}_k - \chi \frac{2k+\alpha}
                {k+\alpha} (\mathbf{F}(\mathbf{y}_k) - \mathbf{F}(\mathbf{y}_{k-1}))
        \end{align}
        $$

        where $g : \mathbb{R}^n \to \mathbb{R}$ is a scalar convex 
        (possibly non-smooth) function, while $\mathbf{F} : \mathbb{R}^n \to 
        \mathbb{R}^n$ is the VI mapping. The convergence of the ARG algorithm 
        is guaranteed for Lipschitz monotone operators, with Lipschitz constant 
        $L < +\infty$, when $\chi \in \left(0,\frac{1}{4L}\right)$
        and $\alpha > 2$.

        [^11]: Boţ, R. I., Csetnek, E. R., & Nguyen, D. K. (2023). Fast 
           Optimistic Gradient Descent Ascent (OGDA) method in continuous and 
           discrete time. Foundations of Computational Mathematics, 1-60.

        Parameters
        ----------
        x_current : ndarray
            The initial point, corresponding to $\mathbf{x}_0$.
        x_previous : ndarray
            The initial point, corresponding to $\mathbf{x}_1$.
        y : ndarray
            The initial auxiliary point, corresponding to $\mathbf{y}_0$.
        step_size : float
            The step size value, corresponding to $\chi$.
        alpha : float
            The auxiliary parameter, corresponding to the $\alpha$ parameter.
        **cvxpy_solve_params
            The parameters for the 
            [`cvxpy.Problem.solve`](https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Problem.solve) 
            method. 

        Yields
        ------
        ndarray
            The iteration's resulting point
        """
        k = 0
        while True:
            y_current = (
                x_current
                + k * (x_current - x_previous) / (k + alpha)
                - step_size * alpha * self.F(y) / (k + alpha)
            )
            x = y_current - step_size * (2 * k + alpha) * (
                self.F(y_current) - self.F(y)
            ) / (k + alpha)

            x_previous = x_current
            x_current = x
            y = y_current
            k += 1
            yield x

    # Constrained Fast Optimistic Gradient Descent Ascent
    def cfogda(
        self,
        x_current: np.ndarray,
        x_previous: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        step_size: float,
        alpha: float = 2.1,
        **cvxpy_solve_params,
    ) -> np.ndarray:
        r"""
        Given a constant step-size $\chi > 0$ and initial vectors $\mathbf{x}_1 \in \mathcal{S}$, $\mathbf{z}_1 \in N_{\mathcal{S}}(\mathbf{x}_1)$, $\mathbf{x}_0,\mathbf{y}_0 \in 
        \mathbb{R}^n$, the basic $k$-th iterate of Constrained Fast Optimistic Gradient Descent Ascent (CFOGDA) is the following[^12]:

        $$ 
        \begin{align}
            \mathbf{y}_k &= \mathbf{x}_k + \frac{k}{k+\alpha}(\mathbf{x}_k -
                \mathbf{x}_{k-1}) - \chi \frac{\alpha}{k+\alpha}(
                \mathbf{F}(\mathbf{y}_k) + \mathbf{z}_k) \\
            \mathbf{x}_{k+1} &= \text{prox}_{g,\mathcal{S}}\left(\mathbf{y}_k
                - \chi\left(1 + \frac{k}{k+\alpha}\right)(\mathbf{F}(\mathbf{y}_k)
                - \mathbf{F}(\mathbf{y}_{k-1}) - \zeta_k)\right) \\
            \mathbf{z}_{k+1} &= \frac{k+\alpha}{\chi (2k+\alpha)}(
                \mathbf{y}_k - \mathbf{x}_{k+1}) - (\mathbf{F}(\mathbf{y}_k)
                - \mathbf{F}(\mathbf{y}_{k-1}) - \zeta_k)
        \end{align}
        $$

        where $g : \mathbb{R}^n \to \mathbb{R}$ is a scalar convex (possibly non-smooth) function, while $\mathbf{F} : \mathbb{R}^n \to \mathbb{R}^n$ is the VI mapping. 
        The convergence of the CFOGDA algorithm is guaranteed for Lipschitz monotone operators, with Lipschitz constant $L < +\infty$, when $\chi \in \left(0,\frac{1}{4L}\right)$
        and $\alpha > 2$.

        [^12]: Sedlmayer, M., Nguyen, D. K., & Bot, R. I. (2023, July). A fast 
           optimistic method for monotone variational inequalities. In 
           International Conference on Machine Learning (pp. 30406-30438). PMLR.

        Parameters
        ----------
        x_current : ndarray
            The initial point, corresponding to $\mathbf{x}_0$.
        x_previous : ndarray
            The initial point, corresponding to $\mathbf{x}_1$.
        y : ndarray
            The initial auxiliary point, corresponding to $\mathbf{y}_0$.
        z : ndarray
            The initial auxiliary point, corresponding to $\mathbf{z}_1$.
        step_size : float
            The step size value, corresponding to $\chi$.
        alpha : float, optional
            The auxiliary parameter, corresponding to the $\alpha$ parameter.
        **cvxpy_solve_params
            The parameters for the 
            [`cvxpy.Problem.solve`](https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Problem.solve) 
            method. 

        Yields
        ------
        ndarray
            The iteration's resulting point
        """
        k = 1
        while True:
            y_current = (
                x_current
                + k * (x_current - x_previous) / (k + alpha)
                - step_size * alpha * (self.F(x_current) + z)
            )
            x = self.prox(
                y - step_size * (1 + k / (k + alpha) * (self.F(y_current) - self.F(y) - z)),
                **cvxpy_solve_params,
            )
            z = (k + alpha) * (y_current - x) / (step_size * (2 * k + alpha)) - (
                self.F(y_current) - self.F(y) - z
            )

            x_previous = x_current
            x_current = x
            y = y_current
            k += 1
            yield x

    # Golden Ratio Algorithm
    def graal(
        self,
        x: np.ndarray,
        y: np.ndarray,
        step_size: float,
        phi: float = GOLDEN_RATIO,
        **cvxpy_solve_params,
    ) -> np.ndarray:
        r"""
        Given a constant step-size $\chi > 0$ and initial vectors $\mathbf{x}_0,\mathbf{y}_0 \in \mathbb{R}^n$, the basic $k$-th iterate the golden ratio algorithm (GRAAL) is the 
        following [^4]:

        $$ 
        \begin{align*}
            \mathbf{y}_{k+1} &= \frac{(\phi - 1)\mathbf{x}_k + \phi\mathbf{y}_k}
            {\phi} \\
            \mathbf{x}_{k+1} &= \text{prox}_{g,\mathcal{S}}(\mathbf{y}_{k+1} - \chi 
                \mathbf{F}(\mathbf{x}_k))
        \end{align*}
        $$

        The convergence of GRAAL algorithm is guaranteed for Lipschitz monotone operators, with Lipschitz constants $L < +\infty$, 
        when $\chi \in \left(0,\frac{\varphi}{2L}\right]$ and $\phi \in (1,\varphi]$, where $\varphi = \frac{1+\sqrt{5}}{2}$ is the golden ratio.


        [^4]: Malitsky, Y. (2020). Golden ratio algorithms for variational 
           inequalities. Mathematical Programming, 184(1), 383-410.

        Parameters
        ----------
        x : ndarray
            The initial point, corresponding to $\mathbf{x}_0$
        y : ndarray
            The initial auxiliary point, corresponding to $\mathbf{y}_0$
        step_size : float
            The step size value, corresponding to $\chi$
        phi : float
            The golden ratio step size, corresponding to $\phi$
        **cvxpy_solve_params
            The parameters for the 
            [`cvxpy.Problem.solve`](https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Problem.solve) 
            method. 

        Yields
        ------
        ndarray
            The iteration's resulting point 
        """
        while True:
            y = ((phi - 1) * x + y) / phi
            x = self.prox(y - step_size * self.F(x), **cvxpy_solve_params)
            yield x

    # Adaptive Golden Ratio Algorithm
    def agraal(
        self,
        x_current: np.ndarray,
        x_previous: np.ndarray,
        step_size: float,
        phi: float = GOLDEN_RATIO,
        step_size_large: float = 1e6,
        **cvxpy_solve_params,
    ) -> np.ndarray:
        r"""
        The Adaptive Golden Ratio Algorithm (aGRAAL) algorithm is a variation of the Golden Ratio Algorithm ([monviso.VI.graal][]), with adaptive step size. 
        Following [^5], let $\theta_0 = 1$, $\rho = 1/\phi + 1/\phi^2$, where $\phi \in (0,\varphi]$ and $\varphi = \frac{1+\sqrt{5}}{2}$ is the golden ratio. 
        Moreover, let $\bar{\chi} \gg 0$ be a constant (arbitrarily large) step-size. 
        Given the initial terms $\mathbf{x}_0,\mathbf{x}_1 \in \mathbb{R}^n$, $\mathbf{y}_0 = \mathbf{x}_1$, and $\chi_0 > 0$, the $k$-th iterate for aGRAAL is the following:
         
        $$
        \begin{align*} 
        \chi_k &= \min\left\{\rho\chi_{k-1},
              \frac{\phi\theta_k \|\mathbf{x}_k
              -\mathbf{x}_{k-1}\|^2}{4\chi_{k-1}\|\mathbf{F}(\mathbf{x}_k)
              -\mathbf{F}(\mathbf{x}_{k-1})\|^2}, \bar{\chi}\right\} \\
        \mathbf{y}_{k+1} &= \frac{(\phi - 1)\mathbf{x}_k + \phi\mathbf{y}_k}{\phi} \\
        \mathbf{x}_{k+1} &= \text{prox}_{g,\mathcal{S}}(\mathbf{y}_{k+1} - \chi 
            \mathbf{F}(\mathbf{x}_k)) \\
        \theta_k &= \phi\frac{\chi_k}{\chi_{k-1}} 
        \end{align*}
        $$

        The convergence guarantees discussed for GRAAL also hold for aGRAAL. 

        [^5]: Malitsky, Y. (2020). Golden ratio algorithms for variational inequalities. Mathematical Programming, 184(1), 383-410.
        Parameters
        ----------
        x_current : ndarray
            The initial point, corresponding to $\mathbf{x}_0$
        x_previous : ndarray
            The initial point, corresponding to $\mathbf{x}_1$
        step_size : float
            The step size initial value, corresponding to $\chi_0$
        phi : float
            The golden ratio step size, corresponding to $\phi$
        step_size_large : float, optional
            A constant (arbitrarily) large value for the step size
        **cvxpy_solve_params
            The parameters for the [`cvxpy.Problem.solve`](https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Problem.solve) method. 

        Yields
        ------
        ndarray
            The iteration's resulting point 
        """
        rho = 1 / phi + 1 / phi**2
        theta = 1
        y = x_current

        while True:
            # lambda update
            step_size_current = np.min(
                (
                    rho * step_size,
                    np.divide(
                        phi * theta * np.linalg.norm(x_current - x_previous, 2),
                        4
                        * step_size
                        * np.linalg.norm(self.F(x_current) - self.F(x_previous), 2),
                    ),
                    step_size_large,
                )
            )

            # graal step
            y = ((phi - 1) * x_current + y) / phi
            x = self.prox(
                y - step_size_current * self.F(x_current), **cvxpy_solve_params
            )

            theta = phi * step_size_current / step_size

            x_previous = x_current
            x_current = x
            step_size = step_size_current
            yield x

    # Hybrid Golden Ratio Algorithm I
    def hgraal_1(
        self,
        x_current: np.ndarray,
        x_previous: np.ndarray,
        step_size: float,
        phi: float = GOLDEN_RATIO,
        step_size_large: float = 1e6,
        **cvxpy_solve_params,
    ) -> np.ndarray:
        r"""
        The HGRAAL-1 algorithm is a variation of the Adaptive Golden Ratio Algorithm ([monviso.VI.agraal][]). 
        Following [^13], let $\theta_0 = 1$, $\rho = 1/\phi + 1/\phi^2$, where $\phi \in (0,\varphi]$ and $\varphi = \frac{1+\sqrt{5}}{2}$ is the golden ratio. 
        The residual at point $\mathbf{x}_k$ is given by $J : \mathbb{R}^n \to \mathbb{R}$, defined as follows:

        $$ J(\mathbf{x}_k) = \|\mathbf{x}_k - \text{prox}_{g,\mathcal{S}} (\mathbf{x}_k - \mathbf{F}(\mathbf{x}_k))\| $$

        Moreover, let $\bar{\chi} \gg 0$ be a constant (arbitrarily large) step-size. 
        Given the initial terms $\mathbf{x}_0,\mathbf{x}_1 \in\mathbb{R}^n$, $\mathbf{y}_0 = \mathbf{x}_1$, and $\chi_0 > 0$, the $k$-th iterate for HGRAAL-1 is the following:

        $$ 
        \begin{align}
            \chi_k &= \min\left\{\rho\chi_{k-1},
                \frac{\phi\theta_k \|\mathbf{x}_k
                -\mathbf{x}_{k-1}\|^2}{4\chi_{k-1}\|\mathbf{F}(\mathbf{x}_k)
                -\mathbf{F}(\mathbf{x}_{k-1})\|^2}, \bar{\chi}\right\} \\
            c_k &= \left(\langle J(\mathbf{x}_k) - J(\mathbf{x}_{k-1}) > 0 \rangle 
                \text{ and } \langle f_k \rangle \right) 
                \text{ or } \left\langle \min\{J(\mathbf{x}_{k-1}), J(\mathbf{x}_k)\} < 
                J(\mathbf{x}_k) + \frac{1}{\bar{k}} \right\rangle \\
            f_k &= \text{not $\langle c_k \rangle$} \\
            \bar{k} &= \begin{cases} \bar{k}+1 & \text{if $c_k$ is true} \\ 
                \bar{k} & \text{otherwise} \end{cases} \\
            \mathbf{y}_{k+1} &= 
                \begin{cases}
                    \dfrac{(\phi - 1)\mathbf{x}_k + \phi\mathbf{y}_k}{\phi} & 
                    \text{if $c_k$ is true} \\
                    \mathbf{x}_k & \text{otherwise}
                \end{cases} \\
            \mathbf{x}_{k+1} &= \text{prox}_{g,\mathcal{S}}(\mathbf{y}_{k+1} - \chi 
                \mathbf{F}(\mathbf{x}_k)) \\
            \theta_k &= \phi\frac{\chi_k}{\chi_{k-1}} 
        \end{align}
        $$

        [^13]: Rahimi Baghbadorani, R., Mohajerin Esfahani, P., & Grammatico, S. (2024). A hybrid algorithm for monotone variational inequalities. 
        (Manuscript submitted for publication).

        Parameters
        ----------
        x_current : ndarray
            The initial point, corresponding to $\mathbf{x}_0$
        x_previous : ndarray
            The initial point, corresponding to $\mathbf{x}_1$
        step_size : float
            The step size initial value, corresponding to $\chi_0$
        phi : float
            The golden ratio step size, corresponding to $\phi$
        step_size_large : float, optional
            A constant (arbitrarily) large value for the step size
        **cvxpy_solve_params
            The parameters for the 
            [`cvxpy.Problem.solve`](https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Problem.solve) 
            method. 

        Yields
        ------
        ndarray
            The iteration's resulting point 
        """
        rho = 1 / phi + 1 / phi**2
        theta = 1
        y = x_current

        flag = False
        k = 1

        # residual computation
        J_min = np.inf

        while True:
            # lambda update
            step_size_current = np.min(
                (
                    rho * step_size,
                    np.divide(
                        phi * theta * np.linalg.norm(x_current - x_previous, 2),
                        4
                        * step_size
                        * np.linalg.norm(self.F(x_current) - self.F(x_previous), 2),
                    ),
                    step_size_large,
                )
            )

            J_current, J_previous = self.residual(x_current), self.residual(x_previous)
            J_min = np.min((J_min, J_previous))

            condition = np.logical_or(
                np.logical_and(J_current - J_previous > 0, flag),
                J_min < J_current + 1 / k,
            )

            y = np.where(condition, ((phi - 1) * x_current + y) / phi, x_current)
            flag = not condition
            k += int(not condition)

            x = self.prox(
                y - step_size_current * self.F(x_current), **cvxpy_solve_params
            )

            theta = phi * step_size_current / step_size

            x_previous = x_current
            x_current = x
            step_size = step_size_current
            yield x

    # Hybrid Golden Ratio Algorithm II
    def hgraal_2(
        self,
        x_current: np.ndarray,
        x_previous: np.ndarray,
        step_size: float,
        phi: float = GOLDEN_RATIO,
        alpha: float = GOLDEN_RATIO,
        step_size_large: float = 1e6,
        phi_large: float = 1e6,
        **cvxpy_solve_params,
    ) -> np.ndarray:
        r"""
        The pseudo-code for the iteration schema can be found at [Algorithm 2][^14].

        [^14]: Rahimi Baghbadorani, R., Mohajerin Esfahani, P., & Grammatico, S.(2024). A hybrid algorithm for monotone variational inequalities.
        (Manuscript submitted for publication).

        Parameters
        ----------
        x_current : ndarray
            The initial point, corresponding to $\mathbf{x}_0$
        x_previous : ndarray
            The initial point, corresponding to $\mathbf{x}_1$
        step_size : float
            The step size initial value, corresponding to $\chi_0$
        phi : float, optional
            The golden ratio step size, corresponding to $\phi$
        alpha : float, optional
            The auxiliary parameter, corresponding to the $\alpha$ parameter.
        step_size_large : float, optional
            A constant (arbitrarily) large value for the step size
        phi_large: float, optional
            A constant (arbitrarily) large value for $\phi$
        **cvxpy_solve_params
            The parameters for the [`cvxpy.Problem.solve`](https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Problem.solve) method.

        Yields
        ------
        ndarray
            The iteration's resulting point
        """

        def s2_update(s2, coefficient):
            return (
                s2
                - step_size * phi * np.linalg.norm(x_current - y, 2) / step_size_current
                + (step_size * phi / step_size_current - 1 - 1 / coefficient)
                * np.linalg.norm(x - y, 2)
                - (step_size * phi / step_size_current - theta)
                * np.linalg.norm(x - x_current, 2)
            )

        rho = 1 / phi + 1 / phi**2
        theta_current = 1
        y_current = x_current
        step_size_current = step_size

        flag = False
        s1_current, s2_current = 0, 0

        while True:
            step_size = np.min(
                (
                    rho * step_size_current,
                    np.divide(
                        alpha
                        * theta_current
                        * np.linalg.norm(x_current - x_previous, 2),
                        4
                        * step_size_current
                        * np.linalg.norm(self.F(x_current) - self.F(x_previous), 2),
                    ),
                    step_size_large,
                )
            )

            y = ((phi - 1) * x_current + y_current) / phi
            x = self.prox(
                y_current - step_size * self.F(x_current), **cvxpy_solve_params
            )
            theta = alpha * step_size / step_size_current

            s1 = (
                s1_current
                + 0.5 * theta_current * np.linalg.norm(x_current - x_previous, 2)
                - step_size * np.linalg.norm(x_current - y, 2) / step_size_current
                + (step_size * phi / step_size_current - 1 - 1 / phi_large)
                * np.linalg.norm(x - y, 2)
                - (step_size * phi / step_size_current - 1.5 * theta)
                * np.linalg.norm(x - x_current, 2)
            )

            s2 = s2_update(s2_current, phi_large)

            condition = np.logical_or(
                np.logical_and(s1 <= 0, flag), np.logical_and(s2 <= 0, not flag)
            )

            if condition:
                phi = phi_large
                flag = True
            else:
                phi = alpha
                if flag:
                    x = x_current
                    x_current = x_previous
                    y = y_current
                    theta = theta_current
                    step_size = step_size_current
                    s1, s2 = 0, 0
                    flag = False
                else:
                    s1 = 0
                    s2 = s2_update(s2_current, alpha)

            x_previous = x_current
            x_current = x
            y_current = y
            theta_current = theta
            step_size_current = step_size
            s1_current = s1
            s2_current = s2

            yield x

    def solution(
        self,
        algorithm_name: str,
        algorithm_params: dict,
        max_iters: int,
        eval_func=None,
        eval_tol: float = 1e-9,
        log_path: str | None = None,
        **cvxpy_solve_params,
    ) -> np.ndarray | None:
        """
        Solve the variational inequality, using the indicated algorithm.

        Parameters
        ----------
        algorithm_name : str
            The name of the algorithm to use
        algorithm_params : dict
            The algorithm's parameters
        max_iters : int
            The maximum number of iterations
        eval_func : callable, optional
            The function used to evaluate the convergence
        eval_tol : float, optional
            The minimum tolerance over the evaluation function
        log_path : str, optional
            The path for saving the log file
        **cvxpy_solve_params
            The parameters for the [`cvxpy.Problem.solve`](https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Problem.solve) method.

        Returns
        -------
        ndarray
            The VI solution
        """
        # default tolerance evaluation funtion: the distance between the
        # current point x and the descent step x - F(x) projected onto the
        # constraints set
        if not eval_func:
            eval_func = self.residual

        # default path for iterations' logs
        if not log_path:
            log_path = datetime.datetime.now().strftime("movi-%d-%m-%yT%H-%M-%S.log")

        init_time = time.process_time()
        algorithm = getattr(self, algorithm_name)
        with open(log_path, "w", encoding="utf-8") as log_file:
            # write the header in the logs
            log_file.write("iter,eval_func_value,time\n")

            # main loop
            algorithm_all_params = algorithm_params | cvxpy_solve_params
            for k, x in enumerate(algorithm(**algorithm_all_params)):
                eval_func_value = eval_func(x)
                log_file.write(
                    f"{k},{eval_func_value},{time.process_time() - init_time}\n"
                )

                # stopping criteria
                if k >= max_iters - 1 or eval_func_value <= eval_tol:
                    return x
