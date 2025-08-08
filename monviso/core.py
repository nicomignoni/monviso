from typing import Callable

import numpy as np
import cvxpy as cp

GOLDEN_RATIO = 0.5 * (np.sqrt(5) + 1)

class VI:
    r"""
    Attributes
    ----------
    F : callable
        The VI vector mapping, i.e., $\mathbf{F} : \mathbb{R}^n \to \mathbb{R}^n$; a function transforming a `np.ndarray` into another `np.ndarray` of the same size.
    g : callable, optional
        The VI scalar mapping, i.e., $g : \mathbb{R}^n \to \mathbb{R}$; a callable returning a [`cvxpy.Expression`](https://www.cvxpy.org/api_reference/cvxpy.expressions.html#id1)
    y : cp.Variable, optional
        The variable of the to be computed by the proximal operator, i.e., $\mathbf{y}$.    
    S : list of cp.Constraints, optional
        The constraints set, i.e., $\mathcal{S} \subseteq \mathbb{R}^n$; a list of [`cvxpy.Constraint`](https://www.cvxpy.org/api_reference/cvxpy.constraints.html#id8)
    analytical_prox : Callable, optional
        The analytical user-defined function for the proximal operator.
    """

    def __init__(
            self, 
            F: Callable, 
            g: Callable | None = None, 
            y: cp.Variable | None = None, 
            S: list[cp.Constraint] | None = None,
            analytical_prox: Callable | None = None
        ) -> None:

        self.F = F

        if analytical_prox is not None:
            self.prox = analytical_prox
        elif y is not None:
            g = (lambda _: 0) if g is None else g
            S = [] if S is None else S
            param_x = cp.Parameter(y.shape)
            prob = cp.Problem(cp.Minimize(g(y) + 0.5 * cp.norm(y - param_x)), S)
            
            def prox(x, **cvxpy_solve_kwargs):
                param_x.value = x
                prob.solve(**cvxpy_solve_kwargs)
                return y.value

            self.prox = prox
        elif y is None and S is None and analytical_prox is None:
            self.prox = lambda x: x
        else:
            raise Exception()

    def residual(self, x: np.ndarray, **kwargs) -> float:
        return np.linalg.norm(x - self.prox(x - self.F(x), **kwargs))

    # Proximal Gradient
    def pg(self, xk: np.ndarray, step_size: float, **kwargs):
        xk1 = self.prox(xk - step_size * self.F(xk), **kwargs)
        return xk1

    # Extragradient
    def eg(self, xk: np.ndarray, step_size: float, **kwargs):
        yk = self.prox(xk - step_size * self.F(xk), **kwargs)
        xk1 = self.prox(xk - step_size * self.F(yk), **kwargs)
        return xk1

    # Popov's Method
    def popov(
        self, 
        xk: np.ndarray, 
        yk: np.ndarray, 
        step_size: float, 
        **kwargs
    ):
        yk1 = self.prox(xk - step_size * self.F(yk), **kwargs)
        xk1 = self.prox(xk - step_size * self.F(yk1), **kwargs)
        return xk1, yk1

    # Forward-Backward-Forward
    def fbf(self, xk: np.ndarray, step_size: float, **kwargs):
        Fxk = self.F(xk)

        yk = self.prox(xk - step_size * Fxk, **kwargs)
        xk1 = yk - step_size * (self.F(yk) - Fxk)
        return xk1

    # Forward-Reflected-Backward
    def frb(
        self,
        xk: np.ndarray,
        x1k: np.ndarray,
        step_size: float,
        **kwargs,
    ):
        xk1 = self.prox(xk - step_size * (2 * self.F(xk) + self.F(x1k)), **kwargs)
        return xk1

    # Projected Reflected Gradient
    def prg(
        self,
        xk: np.ndarray,
        x1k: np.ndarray,
        step_size: float,
        **kwargs,
    ):
        xk1 = self.prox(xk - step_size * self.F(2*xk - x1k), **kwargs)
        return xk1

    # Extra Anchored Gradient
    def eag(
        self, 
        xk: np.ndarray, 
        x0: np.ndarray, 
        k: int,
        step_size: float, 
        **kwargs
    ):
        yk = self.prox(xk - step_size * self.F(xk) + (x0 - xk) / (k + 1), **kwargs)
        xk1 = self.prox(xk - step_size * self.F(yk) + (x0 - xk) / (k + 1), **kwargs)
        return xk1

    # Accelerated Reflected Gradient
    def arg(
        self,
        xk: np.ndarray,
        x1k: np.ndarray,
        x0: np.ndarray,
        k: int,
        step_size: float,
        **kwargs,
    ):
        yk = 2 * xk - x1k + (x0 - xk) / (k + 1) - (xk - x1k) / k
        xk1 = self.prox(xk - step_size * self.F(yk) + (x0 - xk) / (k + 1), **kwargs)
        return xk1

    # (Explicit) Fast Optimistic Gradient Descent Ascent
    def fogda(
        self,
        xk: np.ndarray,
        x1k: np.ndarray,
        y1k: np.ndarray,
        k: int,
        step_size: float,
        alpha: float = 2.1,
    ):
        yk = xk + k * (xk - x1k) / (k + alpha) - step_size * alpha * self.F(y1k) / (k + alpha)
        xk1 = yk - step_size * (2*k + alpha) * (self.F(yk) - self.F(y1k)) / (k + alpha)
        return xk1, yk

    # Constrained Fast Optimistic Gradient Descent Ascent
    def cfogda(
        self,
        xk: np.ndarray,
        x1k: np.ndarray,
        y1k: np.ndarray,
        zk: np.ndarray,
        k: int,
        step_size: float,
        alpha: float = 2.1,
        **kwargs,
    ):
        yk = xk + k * (xk - x1k) / (k + alpha) - step_size * alpha * (self.F(y1k) + zk)
        xk1 = self.prox(
            yk - step_size * (1 + k / (k + alpha)) * (self.F(yk) - self.F(y1k) - zk),
            **kwargs,
        )
        zk1 = (k + alpha) * (yk - xk1) / (step_size * (2*k + alpha)) - (self.F(yk) - self.F(y1k) - zk)

        return xk1, yk, zk1 

    # Golden Ratio Algorithm
    def graal(
        self,
        xk: np.ndarray,
        yk: np.ndarray,
        step_size: float,
        phi: float = GOLDEN_RATIO,
        **kwargs,
    ):
        yk1 = ((phi - 1) * xk + yk) / phi
        xk1 = self.prox(yk1 - step_size * self.F(xk), **kwargs)
        return xk1, yk1

    # Adaptive Golden Ratio Algorithm
    def agraal(
        self,
        xk: np.ndarray,
        x1k: np.ndarray,
        yk: np.ndarray,
        s1k: float,
        tk: float = 1,
        step_size_large: float = 1e6,
        phi: float = GOLDEN_RATIO,
        **kwargs,
    ):
        rho = 1 / phi + 1 / phi**2

        # step-size update
        sk = np.min(
            (
                rho * s1k,
                np.divide(
                    phi * tk * np.linalg.norm(xk - x1k, 2),
                    4 * s1k * np.linalg.norm(self.F(xk) - self.F(x1k), 2),
                ),
                step_size_large,
            )
        )

        # graal step
        xk1, yk1 = self.graal(xk, yk, sk, phi, **kwargs)

        tk1 = phi * sk / s1k

        return xk1, yk1, sk, tk1

    # Hybrid Golden Ratio Algorithm I
    def hgraal_1(
        self,
        xk: np.ndarray,
        x1k: np.ndarray,
        yk: np.ndarray,
        s1k: float,
        tk: float = 1,
        ck: int = 1,
        phi: float = GOLDEN_RATIO,
        step_size_large: float = 1e6,
        **kwargs,
    ):
        rho = 1 / phi + 1 / phi**2
        flag = False

        # step-size update
        sk = np.min(
            (
                rho * s1k,
                np.divide(
                    phi * tk * np.linalg.norm(xk - x1k, 2),
                    4 * s1k * np.linalg.norm(self.F(xk) - self.F(x1k), 2),
                ),
                step_size_large,
            )
        )

        Jk, J1k = self.residual(xk), self.residual(x1k)
        condition = np.logical_or(
            np.logical_and(Jk - J1k > 0, flag),
            np.min((Jk, J1k)) < Jk + 1 / ck,
        )

        yk1 = np.where(condition, ((phi - 1) * xk + yk) / phi, xk)
        flag = not condition
        ck1 = ck + 1 if condition else ck

        xk1 = self.prox(yk1 - sk * self.F(xk), **kwargs)
        tk1 = phi * sk / s1k

        return xk1, yk1, sk, tk1, ck1

    # # Hybrid Golden Ratio Algorithm II
    # def hgraal_2(
    #     self,
    #     xk: np.ndarray,
    #     x1k: np.ndarray,
    #     step_size: float,
    #     phi: float = GOLDEN_RATIO,
    #     alpha: float = GOLDEN_RATIO,
    #     step_size_large: float = 1e6,
    #     phi_large: float = 1e6,
    #     **kwargs,
    # ):
    #     r"""
    #     The pseudo-code for the iteration schema can be found at [Algorithm 2][^14].
    #
    #     [^14]: Rahimi Baghbadorani, R., Mohajerin Esfahani, P., & Grammatico, S.(2024). A hybrid algorithm for monotone variational inequalities.
    #     (Manuscript submitted for publication).
    #
    #     Parameters
    #     ----------
    #     x_current : ndarray
    #         The initial point, corresponding to $\mathbf{x}_0$
    #     x_previous : ndarray
    #         The initial point, corresponding to $\mathbf{x}_1$
    #     step_size : float
    #         The step size initial value, corresponding to $\chi_0$
    #     phi : float, optional
    #         The golden ratio step size, corresponding to $\phi$
    #     alpha : float, optional
    #         The auxiliary parameter, corresponding to the $\alpha$ parameter.
    #     step_size_large : float, optional
    #         A constant (arbitrarily) large value for the step size
    #     phi_large: float, optional
    #         A constant (arbitrarily) large value for $\phi$
    #     **kwargs
    #         The parameters for the [`cvxpy.Problem.solve`](https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Problem.solve) method.
    #
    #     Yields
    #     ------
    #     ndarray
    #         The iteration's resulting point
    #     """
    #
    #     def s2_update(s2, coefficient):
    #         return (
    #             s2
    #             - step_size * phi * np.linalg.norm(x_current - y, 2) / step_size_current
    #             + (step_size * phi / step_size_current - 1 - 1 / coefficient)
    #             * np.linalg.norm(x - y, 2)
    #             - (step_size * phi / step_size_current - theta)
    #             * np.linalg.norm(x - x_current, 2)
    #         )
    #
    #     rho = 1 / phi + 1 / phi**2
    #     theta_current = 1
    #     y_current = x_current
    #     step_size_current = step_size
    #
    #     flag = False
    #     s1_current, s2_current = 0, 0
    #
    #     while True:
    #         step_size = np.min(
    #             (
    #                 rho * step_size_current,
    #                 np.divide(
    #                     alpha
    #                     * theta_current
    #                     * np.linalg.norm(x_current - x_previous, 2),
    #                     4
    #                     * step_size_current
    #                     * np.linalg.norm(self.F(x_current) - self.F(x_previous), 2),
    #                 ),
    #                 step_size_large,
    #             )
    #         )
    #
    #         y = ((phi - 1) * x_current + y_current) / phi
    #         x = self.prox(
    #             y_current - step_size * self.F(x_current), **kwargs
    #         )
    #         theta = alpha * step_size / step_size_current
    #
    #         s1 = (
    #             s1_current
    #             + 0.5 * theta_current * np.linalg.norm(x_current - x_previous, 2)
    #             - step_size * np.linalg.norm(x_current - y, 2) / step_size_current
    #             + (step_size * phi / step_size_current - 1 - 1 / phi_large)
    #             * np.linalg.norm(x - y, 2)
    #             - (step_size * phi / step_size_current - 1.5 * theta)
    #             * np.linalg.norm(x - x_current, 2)
    #         )
    #
    #         s2 = s2_update(s2_current, phi_large)
    #
    #         condition = np.logical_or(
    #             np.logical_and(s1 <= 0, flag), np.logical_and(s2 <= 0, not flag)
    #         )
    #
    #         if condition:
    #             phi = phi_large
    #             flag = True
    #         else:
    #             phi = alpha
    #             if flag:
    #                 x = x_current
    #                 x_current = x_previous
    #                 y = y_current
    #                 theta = theta_current
    #                 step_size = step_size_current
    #                 s1, s2 = 0, 0
    #                 flag = False
    #             else:
    #                 s1 = 0
    #                 s2 = s2_update(s2_current, alpha)
    #
    #         x_previous = x_current
    #         x_current = x
    #         y_current = y
    #         theta_current = theta
    #         step_size_current = step_size
    #         s1_current = s1
    #         s2_current = s2
    #
    #         yield x
