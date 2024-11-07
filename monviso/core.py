import time
import datetime
import math

import numpy as np
import cvxpy as cp

GOLDEN_RATIO = 0.5*(math.sqrt(5) + 1)


class VI:
    r"""**Variational Inequality (VI)**

    Attributes
    ----------
    F : callable
        The VI vector mapping, a function transforming a ndarray into an other 
        ndarray of the same size. 
    g : cp.expressions.expression.Expression, optional
        The VI scalar mapping, a single ``cvxpy``'s 
        `Expression <https://www.cvxpy.org/api_reference/cvxpy.expressions.html#id1>`_ 
    S : list of cvxpy.constraints.constraint.Constraint, optional
        The constraints set, a list of ``cvxpy``'s 
        `Constraints <https://www.cvxpy.org/api_reference/cvxpy.constraints.html#id8>`_ 
    n : int, optional
        The size of the vector space 
    """

    def __init__(self, F, g=0, S=None, n=None) -> None:
        if S is None:
            S = []

        # Each constraint/expression in the constraints' set, together with g,
        # must depend on a single variable
        extended_S = S + [g]
        variables = set(
            sum([e.variables() for e in extended_S if hasattr(e, "variables")], [])
        )
        assert len(variables) <= 1

        # The constraints set variable must be a vector
        self.y = variables.pop() if len(variables) == 1 else cp.Variable(n)
        assert self.y.ndim == 1

        self.F = F
        self.g = g
        self.S = S

        self.param_x = cp.Parameter(self.y.size)
        self._prox = cp.Problem(cp.Minimize(self.g + 0.5*cp.norm(self.y - self.param_x)), S)
        self._proj = cp.Problem(cp.Minimize(0.5*cp.norm(self.y - self.param_x)), S)

    def prox(self, x, **cvxpy_solve_params):
        r"""**Constrained Proximal Operator**

        Given a scalar function :math:`g  : \mathbb{R}^n \to \mathbb{R}` and
        a constraints set :math:`\mathcal{S} \subseteq \mathbb{R}^n`, the 
        constrained proximal operator is defined as
        
        .. math:: 
            \text{prox}_{g,\mathcal{S}}(\mathbf{x}) = 
            \underset{\mathbf{y} \in \mathcal{S}}{\text{argmin}} \left\{
            g(\mathbf{y}) + \frac{1}{2}\|\mathbf{y} - \mathbf{x}\|^2 \right\}

        Arguments
        ---------
        x : ndarray
            The proximal operator argument point
        **cvxpy_solve_params
            The parameters for the 
            `cvxpy.Problem.solve <https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Problem.solve>`_ 
            method. 

        Returns
        -------
        ndarray
            The proximal operator resulting point
        """
        self.param_x.value = x
        self._prox.solve(**cvxpy_solve_params)
        return self.y.value

    def proj(self, x, **cvxpy_solve_params):
        r"""**Projection Operator**

        The projection operator of a point $\mathbf{x} \in \mathbb{R}^n$ with 
        respect to set $\mathcal{S} \subseteq \mathbb{R}^n$ returns the closest
        point to $\mathbf{x}$ that belongs to $\mathcal{S}$, i.e.,

        .. math:: 
            \text{proj}_{\mathcal{S}}(\mathbf{x}) = \text{prox}_{0,\mathcal{S}}
            (\mathbf{x}) = \underset{\mathbf{y} 
            \in \mathcal{S}}{\text{argmin}} \left\{\frac{1}{2}\|\mathbf{y} - 
            \mathbf{x}\|^2 \right\}

        Arguments
        ---------
        x : ndarray
            The projection operator argument point
        **cvxpy_solve_params
            The parameters for the 
            `cvxpy.Problem.solve <https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Problem.solve>`_ 
            method. 

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
        Computes the distance between a point and its update projected onto
        the feasible set.

        Parameters
        ----------
        x : ndarray
            The argument point
        **cvxpy_solve_params
            The parameters for the 
            `cvxpy.Problem.solve <https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Problem.solve>`_ 
            method. 

        Return
        ------
        float
            The residual value
        """
        return np.linalg.norm(x - self.prox(x - self.F(x), **cvxpy_solve_params))

    ########## Algorithms ##########
    def pg(self,
            x: np.ndarray,
            step_size: float,
            **cvxpy_solve_params
        ) -> np.ndarray:
        r"""**Proximal Gradient**
        
        Given a constant step-size :math:`\lambda > 0` and an initial vector 
        :math:`\mathbf{x}_0 \in \mathbb{R}^n`, the basic :math:`k`-th iterate 
        of the proximal gradient (PG) algorithm is [1]_:

        .. math:: \mathbf{x}_{k+1} = \text{prox}_{g,\mathcal{S}}(\mathbf{x}_k - 
            \lambda F(\mathbf{x}_k))

        where :math:`g : \mathbb{R}^n \to \mathbb{R}` is a scalar convex 
        (possibly non-smooth) function, while :math:`F : \mathbb{R}^n \to 
        \mathbb{R}^n` is the VI mapping. The convergence of PG is guaranteed 
        for Lipshitz strongly monotone operators, with monotone constant 
        :math:`\mu > 0` and Lipshitz constants :math:`L < +\infty`, when 
        :math:`\lambda \in (0, 2\mu/L^2)`.

        Arguments
        ---------
        x : ndarray 
            The initial point, corresponding to :math:`\mathbf{x}_0`
        step_size : float
            The steps size value, corresponding to :math:`\lambda`
        **cvxpy_solve_params
            The parameters for the 
            `cvxpy.Problem.solve <https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Problem.solve>`_ 
            method. 

        Yields
        ------
        ndarray
            The iteration's resulting point 

        References
        ----------
        .. [1] Nemirovskij, A. S., & Yudin, D. B. (1983). Problem complexity 
           and method efficiency in optimization.
        """
        while True:
            x = self.prox(x - step_size*self.F(x), **cvxpy_solve_params)
            yield x

    def eg(self,
            x: np.ndarray,
            step_size: float,
            **cvxpy_solve_params
        ) -> np.ndarray:
        r"""**Extragradient**

        Given a constant step-size :math:`\lambda > 0` and an initial vector 
        :math:`\mathbf{x}_0 \in \mathbb{R}^n`, the :math:`k`-th iterate 
        of the extragradient algorithm (EG) is [2]_:

        .. math:: 
            \begin{align}
                \mathbf{y}_k &= \text{prox}_{g,\mathcal{S}}(\mathbf{x}_k - 
                    \lambda F(\mathbf{x}_k)) \\
                \mathbf{x}_{k+1} &= \text{prox}_{g,\mathcal{S}}(\mathbf{y}_k - 
                    \lambda F(\mathbf{x}_k))
            \end{align}
 
        where :math:`g : \mathbb{R}^n \to \mathbb{R}` is a scalar convex 
        (possibly non-smooth) function, while :math:`F : \mathbb{R}^n \to 
        \mathbb{R}^n` is the VI mapping. The convergence of the EGD algorithm 
        is guaranteed for Lipshitz monotone operators, with Lipshitz constant 
        :math:`L < +\infty`, when :math:`\lambda \in \left(0,\frac{1}{L}\right)`.

        Arguments
        ---------
        x : ndarray 
            The initial point, corresponding to :math:`\mathbf{x}_0`
        step_size : float
            The step size value, corresponding to :math:`\lambda`
        **cvxpy_solve_params
            The parameters for the 
            `cvxpy.Problem.solve <https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Problem.solve>`_ 
            method. 

        Yields
        ------
        ndarray
            The iteration's resulting point 

        References
        ----------
        .. [2] Korpelevich, G. M. (1976). The extragradient method for finding 
           saddle points and other problems. Matecon, 12, 747-756.
        """
        while True:
            y = self.prox(x - step_size*self.F(x), **cvxpy_solve_params)
            x = self.prox(x - step_size*self.F(y), **cvxpy_solve_params)
            yield x

    def popov(self,
            x: np.ndarray,
            y: np.ndarray,
            step_size: float,
            **cvxpy_solve_params
        ) -> np.ndarray:
        r"""**Popov's Method**

        Given a constant step-size :math:`\lambda > 0` and an initial vectors 
        :math:`\mathbf{x}_0,\mathbf{y}_0 \in \mathbb{R}^n`, the :math:`k`-th 
        iterate of Popov's Method (PM) is [6]_:

         .. math:: 
            \begin{align}
                \mathbf{y}_ {k+1} &= \text{prox}_{g,\mathcal{S}}(\mathbf{x}_k - 
                    \lambda F(\mathbf{y}_k)) \\
                \mathbf{x}_{k+1} &= \text{prox}_{g,\mathcal{S}}(\mathbf{y}_{k+1} - 
                    \lambda F(\mathbf{x}_k))
            \end{align}

        where :math:`g : \mathbb{R}^n \to \mathbb{R}` is a scalar convex 
        (possibly non-smooth) function, while :math:`F : \mathbb{R}^n \to 
        \mathbb{R}^n` is the VI mapping. The convergence of PM 
        is guaranteed for Lipshitz monotone operators, with Lipshitz constant 
        :math:`L < +\infty`, when :math:`\lambda \in \left(0,\frac{1}{2L}\right)`.

        Arguments
        ---------
        x : ndarray
            The initial point, corresponding to :math:`\mathbf{x}_0`
        y : ndarray
            The initial auxiliary point, corresponding to :math:`\mathbf{y}_0`
        step_size : float
            The step size value, corresponding to :math:`\lambda`
        **cvxpy_solve_params
            The parameters for the 
            `cvxpy.Problem.solve <https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Problem.solve>`_ 
            method. 

        Yields
        ------
        ndarray
            The iteration's resulting point

        References
        ----------
        .. [6] Popov, L.D. A modification of the Arrow-Hurwicz method for search 
           of saddle points. Mathematical Notes of the Academy of Sciences of 
           the USSR 28, 845–848 (1980)
        """
        while True:
            y = self.prox(x - step_size*self.F(y), **cvxpy_solve_params)
            x = self.prox(x - step_size*self.F(y), **cvxpy_solve_params)
            yield x

    def fbf(self,
            x: np.ndarray,
            step_size: float,
            **cvxpy_solve_params
        ) -> np.ndarray:
        r"""**Forward-Backward-Forward**
        
        Given a constant step-size :math:`\lambda > 0` and an initial vector 
        :math:`\mathbf{x}_0 \in \mathbb{R}^n`, the :math:`k`-th 
        iterate of Forward-Backward-Forward (FBF) algorithm is [7]_:

        .. math:: 
            \begin{align}
                \mathbf{y}_k &= \text{prox}_{g,\mathcal{S}}(\mathbf{x}_k - 
                    \lambda F(\mathbf{x}_k)) \\
                \mathbf{x}_{k+1} &= \mathbf{y}_k - 
                    \lambda F(\mathbf{y}_k) + \lambda F(\mathbf{x}_k)
            \end{align}

        where :math:`g : \mathbb{R}^n \to \mathbb{R}` is a scalar convex 
        (possibly non-smooth) function, while :math:`F : \mathbb{R}^n \to 
        \mathbb{R}^n` is the VI mapping. The convergence of the FBF algorithm 
        is guaranteed for Lipshitz monotone operators, with Lipshitz constant 
        :math:`L < +\infty`, when :math:`\lambda \in \left(0,\frac{1}{L}\right)`.

        Arguments
        ---------
        x : ndarray
            The initial point, corresponding to :math:`\mathbf{x}_0`
        step_size : float
            The step size value, corresponding to :math:`\lambda`
        **cvxpy_solve_params
            The parameters for the 
            `cvxpy.Problem.solve <https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Problem.solve>`_ 
            method. 

        Yields
        ------
        ndarray
            The iteration's resulting point

        References
        ----------
        .. [7] Tseng, P. (2000). A modified forward-backward splitting method 
           for maximal monotone mappings. SIAM Journal on Control and 
           Optimization, 38(2), 431-446.
        """
        while True:
            y = self.prox(x - step_size*self.F(x), **cvxpy_solve_params)
            x = y - step_size*self.F(y) + step_size*self.F(x)
            yield x

    def frb(self,
            x_current: np.ndarray,
            x_previous: np.ndarray,
            step_size: float,
            **cvxpy_solve_params
        ) -> np.ndarray:
        r"""**Forward-Reflected-Backward**

        Given a constant step-size :math:`\lambda > 0` and initial vectors 
        :math:`\mathbf{x}_1,\mathbf{x}_0 \in \mathbb{R}^n`, the basic 
        :math:`k`-th iterate of the Forward-Reflected-Backward (FRB) 
        is the following [8]_:

        .. math::
            \mathbf{x}_k = \text{prox}_{g,\mathcal{S}}(\mathbf{x}_k - 2\lambda F(\mathbf{x}_k) 
                + \lambda F(\mathbf{x}_{k-1}))

        where :math:`g : \mathbb{R}^n \to \mathbb{R}` is a scalar convex 
        (possibly non-smooth) function, while :math:`F : \mathbb{R}^n \to 
        \mathbb{R}^n` is the VI mapping. The convergence of the FRB algorithm 
        is guaranteed for Lipshitz monotone operators, with Lipshitz constant 
        :math:`L < +\infty`, when :math:`\lambda \in \left(0,\frac{1}{2L}\right)`.

        Arguments
        ---------
        x_current : ndarray
            The initial point, corresponding to :math:`\mathbf{x}_0`
        x_previous : ndarray
            The initial point, corresponding to :math:`\mathbf{x}_1`
        step_size : float
            The step size value, corresponding to :math:`\lambda`
        **cvxpy_solve_params
            The parameters for the 
            `cvxpy.Problem.solve <https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Problem.solve>`_ 
            method. 

        Yields
        ------
        ndarray
            The iteration's resulting point

        References
        ----------
        .. [8] Malitsky, Y., & Tam, M. K. (2020). A forward-backward splitting 
           method for monotone inclusions without cocoercivity. SIAM Journal on 
           Optimization, 30(2), 1451-1472.
        """
        while True:
            x = self.prox(x_current - 2*step_size*self.F(x_current) + \
                step_size*self.F(x_previous), **cvxpy_solve_params)
            x_previous = x_current
            x_current = x
            yield x

    def prg(self,
            x_current: np.ndarray,
            x_previous: np.ndarray,
            step_size: float,
            **cvxpy_solve_params
        ) -> np.ndarray:
        r"""**Projected Reflected Gradient**

        Given a constant step-size :math:`\lambda > 0` and initial vectors 
        :math:`\mathbf{x}_1,\mathbf{x}_0 \in \mathbb{R}^n`, the basic 
        :math:`k`-th iterate of the projected reflected gradient (PRG) 
        is the following [3]_:
        
        .. math:: \mathbf{x}_{k+1} = \text{prox}_{g,\mathcal{S}}(\mathbf{x}_k - \lambda 
            F(2\mathbf{x}_k - \mathbf{x}_{k-1}))

        where :math:`g : \mathbb{R}^n \to \mathbb{R}` is a scalar convex 
        (possibly non-smooth) function, while :math:`F : \mathbb{R}^n \to 
        \mathbb{R}^n` is the VI mapping.
        The convergence of PRG algorithm is guaranteed for Lipshitz monotone 
        operators, with Lipshitz constants :math:`L < +\infty`, when 
        :math:`\lambda \in (0,(\sqrt{2} - 1)/L)`. Differently from the EGD 
        iteration, the PRGD has the advantage of requiring a single 
        proximal operator evaluation. 

        Arguments
        ---------
        x_current : ndarray
            The initial point, corresponding to :math:`\mathbf{x}_0`
        x_previous : ndarray
            The initial point, corresponding to :math:`\mathbf{x}_1`
        step_size : float
            The step size value, corresponding to :math:`\lambda`
        **cvxpy_solve_params
            The parameters for the 
            `cvxpy.Problem.solve <https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Problem.solve>`_ 
            method. 

        Yields
        ------
        ndarray
            The iteration's resulting point

        References
        ----------
        .. [3] Malitsky, Y. (2015). Projected reflected gradient methods for 
           monotone variational inequalities. SIAM Journal on Optimization, 
           25(1), 502-520.
        """
        while True:
            x = self.prox(x_current - step_size*self.F(2*x_current - x_previous),
                **cvxpy_solve_params)
            x_previous = x_current
            x_current = x
            yield x

    def eag(self,
            x: np.ndarray,
            step_size: float,
            **cvxpy_solve_params
        ) -> np.ndarray:
        r"""**Extra Anchored Gradient**

        Given a constant step-size :math:`\lambda > 0` and an initial vector 
        :math:`\mathbf{x}_0 \in \mathbb{R}^n`, the :math:`k`-th 
        iterate of extra anchored gradient (EAG) algorithm is [9]_:

        .. math::
            \begin{align}
                \mathbf{y}_k &= \text{prox}_{g,\mathcal{S}}\left(\mathbf{x}_k - 
                    \lambda F(\mathbf{x}_k) + \frac{1}{k+1}(\mathbf{x}_0 - 
                    \mathbf{x}_k)\right) \\
                \mathbf{x}_{k+1} &= \text{prox}_{g,\mathcal{S}}\left(\mathbf{x}_k - 
                    \lambda F(\mathbf{y}_k) + \frac{1}{k+1}(\mathbf{x}_0 - 
                    \mathbf{x}_k)\right)
            \end{align}

        where :math:`g : \mathbb{R}^n \to \mathbb{R}` is a scalar convex 
        (possibly non-smooth) function, while :math:`F : \mathbb{R}^n \to 
        \mathbb{R}^n` is the VI mapping. The convergence of the EAG algorithm 
        is guaranteed for Lipshitz monotone operators, with Lipshitz constant 
        :math:`L < +\infty`, when :math:`\lambda \in \left(0,\frac{1}{\sqrt{3}L}
        \right)`.

        Arguments
        ---------
        x : ndarray
            The initial point, corresponding to :math:`\mathbf{x}_0`
        step_size : float
            The step size value, corresponding to :math:`\lambda`
        **cvxpy_solve_params
            The parameters for the 
            `cvxpy.Problem.solve <https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Problem.solve>`_ 
            method. 
        
        Yields
        ------
        ndarray
            The iteration's resulting point

        References
        ----------
        .. [9] Yoon, T., & Ryu, E. K. (2021, July). Accelerated Algorithms for 
           Smooth Convex-Concave Minimax Problems with O (1/k^ 2) Rate on Squared 
           Gradient Norm. In International Conference on Machine Learning (pp. 
           12098-12109). PMLR.
        """
        k = 0
        x0 = x
        while True:
            y = self.prox(x - step_size*self.F(x) + (x0 - x)/(k+1), **cvxpy_solve_params)
            x = self.prox(x - step_size*self.F(y) + (x0 - x)/(k+1), **cvxpy_solve_params)
            k += 1
            yield x

    def arg(self,
            x_current: np.ndarray,
            x_previous: np.ndarray,
            step_size: float,
            **cvxpy_solve_params
        ) -> np.ndarray:
        r"""**Accelerated Reflected Gradient**

        Given a constant step-size :math:`\lambda > 0` and initial vectors 
        :math:`\mathbf{x}_1,\mathbf{x}_0 \in \mathbb{R}^n`, the basic 
        :math:`k`-th iterate of the accelerated reflected gradient (ARG) 
        is the following [10]_:

        .. math::
            \begin{align}
                \mathbf{y}_k &= 2\mathbf{x}_k - \mathbf{x}_{k-1} + \frac{1}{k+1}
                (\mathbf{x}_0 - \mathbf{x}_k) - \frac{1}{k}(\mathbf{x}_k - 
                \mathbf{x}_{k-1}) \\
                \mathbf{x}_{k+1} &= \text{prox}_{g,\mathcal{S}}\left(\mathbf{x}_k - 
                    \lambda F(\mathbf{y}_k) + \frac{1}{k+1}(\mathbf{x}_0 - 
                    \mathbf{x}_k)\right)
            \end{align}

        where :math:`g : \mathbb{R}^n \to \mathbb{R}` is a scalar convex 
        (possibly non-smooth) function, while :math:`F : \mathbb{R}^n \to 
        \mathbb{R}^n` is the VI mapping. The convergence of the ARG algorithm 
        is guaranteed for Lipshitz monotone operators, with Lipshitz constant 
        :math:`L < +\infty`, when :math:`\lambda \in \left(0,\frac{1}{12L}\right)`.

        Arguments
        ---------
        x_current : ndarray
            The initial point, corresponding to :math:`\mathbf{x}_0`
        x_previous : ndarray
            The initial point, corresponding to :math:`\mathbf{x}_1`
        step_size : float
            The step size value, corresponding to :math:`\lambda`
        **cvxpy_solve_params
            The parameters for the 
            `cvxpy.Problem.solve <https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Problem.solve>`_ 
            method. 

        Yields
        ------
        ndarray
            The iteration's resulting point

        References
        ----------
        .. [10] Cai, Y., & Zheng, W. (2022). Accelerated single-call methods 
           for constrained min-max optimization. arXiv preprint arXiv:2210.03096.
        """
        k = 1
        x0 = x_previous
        while True:
            y = 2*x_current - x_previous + 1/(k+1)*(x0 - x_current) - \
                1/k*(x0 - x_previous)
            x = self.prox(x_current - step_size*self.F(y) + 1/(k+1)*(x0 - x_current),
                **cvxpy_solve_params)

            x_previous = x_current
            x_current = x
            k += 1
            yield x

    def fogda(self,
            x_current: np.ndarray,
            x_previous: np.ndarray,
            y: np.ndarray,
            step_size: float,
            alpha: float = 2.1,
        ) -> np.ndarray:
        r"""**(Explicit) Fast Optimistic Gradient Descent Ascent**

        Given a constant step-size :math:`\lambda > 0` and initial vectors 
        :math:`\mathbf{x}_1,\mathbf{x}_0,\mathbf{y}_0 \in \mathbb{R}^n`, the 
        basic :math:`k`-th iterate of the explicit fast OGDA (FOGDA) 
        is the following [11]_:

        .. math::
            \begin{align}
                \mathbf{y}_k &= \mathbf{x}_k + \frac{k}{k+\alpha}(\mathbf{x}_k - 
                    \mathbf{x}_{k-1}) - \lambda \frac{\alpha}{k+\alpha}
                    F(\mathbf{y}_{k-1}) \\
                \mathbf{x}_{k+1} &= \mathbf{y}_k - \lambda \frac{2k+\alpha}
                    {k+\alpha} (F(\mathbf{y}_k) -F(\mathbf{y}_{k-1}))
            \end{align}

        where :math:`g : \mathbb{R}^n \to \mathbb{R}` is a scalar convex 
        (possibly non-smooth) function, while :math:`F : \mathbb{R}^n \to 
        \mathbb{R}^n` is the VI mapping. The convergence of the ARG algorithm 
        is guaranteed for Lipshitz monotone operators, with Lipshitz constant 
        :math:`L < +\infty`, when :math:`\lambda \in \left(0,\frac{1}{4L}\right)`
        and :math:`\alpha > 2`.

        Arguments
        ---------
        x_current : ndarray
            The initial point, corresponding to :math:`\mathbf{x}_0`.
        x_previous : ndarray
            The initial point, corresponding to :math:`\mathbf{x}_1`.
        y : ndarray
            The initial auxiliary point, corresponding to :math:`\mathbf{y}_0`.
        step_size : float
            The step size value, corresponding to :math:`\lambda`.
        alpha : float
            The auxiliary parameter, corresponding to the :math:`\alpha` parameter.
        **cvxpy_solve_params
            The parameters for the 
            `cvxpy.Problem.solve <https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Problem.solve>`_ 
            method. 

        Yields
        ------
        ndarray
            The iteration's resulting point

        References
        ----------
        .. [11] Boţ, R. I., Csetnek, E. R., & Nguyen, D. K. (2023). Fast 
           Optimistic Gradient Descent Ascent (OGDA) method in continuous and 
           discrete time. Foundations of Computational Mathematics, 1-60.
        """
        k = 0
        while True:
            y_current = x_current + k*(x_current - x_previous)/(k+alpha) - \
                step_size*alpha*self.F(y)/(k+alpha)
            x = y_current - step_size*(2*k + alpha)*(self.F(y_current) - \
                self.F(y))/(k+alpha)

            x_previous = x_current
            x_current = x
            y = y_current
            k += 1
            yield x

    def cfogda(self,
            x_current: np.ndarray,
            x_previous: np.ndarray,
            y: np.ndarray,
            z: np.ndarray,
            step_size: float,
            alpha: float = 2.1,
            **cvxpy_solve_params
        ) -> np.ndarray:
        r"""**Constrained Fast Optimistic Gradient Descent Ascent**

        Given a constant step-size :math:`\lambda > 0` and initial vectors 
        :math:`\mathbf{x}_1 \in \mathcal{S}`, :math:`\mathbf{z}_1 \in 
        N_{\mathcal{S}}(\mathbf{x}_1)`, :math:`\mathbf{x}_0,\mathbf{y}_0 \in 
        \mathbb{R}^n`, the basic :math:`k`-th iterate of Constrained Fast 
        Optimistic Gradient Descent Ascent (CFOGDA) is the following [12]_:

        .. math:: 
            \begin{align}
                \mathbf{y}_k &= \mathbf{x}_k + \frac{k}{k+\alpha}(\mathbf{x}_k -
                    \mathbf{x}_{k-1}) - \lambda \frac{\alpha}{k+\alpha}(
                    F(\mathbf{y}_k) + \mathbf{z}_k) \\
                \mathbf{x}_{k+1} &= \text{prox}_{g,\mathcal{S}}\left(\mathbf{y}_k
                    - \lambda\left(1 + \frac{k}{k+\alpha}\right)(F(\mathbf{y}_k)
                    - F(\mathbf{y}_{k-1}) - \zeta_k)\right) \\
                \mathbf{z}_{k+1} &= \frac{k+\alpha}{\lambda (2k+\alpha)}(
                    \mathbf{y}_k - \mathbf{x}_{k+1}) - (F(\mathbf{y}_k)
                    - F(\mathbf{y}_{k-1}) - \zeta_k)
            \end{align}

        where :math:`g : \mathbb{R}^n \to \mathbb{R}` is a scalar convex 
        (possibly non-smooth) function, while :math:`F : \mathbb{R}^n \to 
        \mathbb{R}^n` is the VI mapping. The convergence of the CFOGDA algorithm 
        is guaranteed for Lipshitz monotone operators, with Lipshitz constant 
        :math:`L < +\infty`, when :math:`\lambda \in \left(0,\frac{1}{4L}\right)`
        and :math:`\alpha > 2`.

        Arguments
        ---------
        x_current : ndarray
            The initial point, corresponding to :math:`\mathbf{x}_0`.
        x_previous : ndarray
            The initial point, corresponding to :math:`\mathbf{x}_1`.
        y : ndarray
            The initial auxiliary point, corresponding to :math:`\mathbf{y}_0`.
        z : ndarray
            The initial auxiliary point, corresponding to :math:`\mathbf{z}_1`.
        step_size : float
            The step size value, corresponding to :math:`\lambda`.
        alpha : float, optional
            The auxiliary parameter, corresponding to the :math:`\alpha` parameter.
        **cvxpy_solve_params
            The parameters for the 
            `cvxpy.Problem.solve <https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Problem.solve>`_ 
            method. 

        Yields
        ------
        ndarray
            The iteration's resulting point

        References
        ----------
        .. [12] Sedlmayer, M., Nguyen, D. K., & Bot, R. I. (2023, July). A fast 
           optimistic method for monotone variational inequalities. In 
           International Conference on Machine Learning (pp. 30406-30438). PMLR.
        """
        k = 1
        while True:
            y_current = x_current + k*(x_current - x_previous)/(k+alpha) - \
                        step_size*alpha*(self.F(x_current) + z)
            x = self.prox(
                y - step_size*(1 + k/(k+alpha)*(self.F(y_current) - \
                self.F(y) - z)), **cvxpy_solve_params
            )
            z = (k+alpha)*(y_current - x)/(step_size*(2*k+alpha)) - \
                (self.F(y_current) - self.F(y) - z)

            x_previous = x_current
            x_current = x
            y = y_current
            k += 1
            yield x

    def graal(self,
            x: np.ndarray,
            y: np.ndarray,
            step_size: float,
            phi: float = GOLDEN_RATIO,
            **cvxpy_solve_params
        ) -> np.ndarray:
        r"""**Golden Ratio Algorithm**

        Given a constant step-size :math:`\lambda > 0` and initial vectors 
        :math:`\mathbf{x}_0,\mathbf{y}_0 \in \mathbb{R}^n`, the basic 
        :math:`k`-th iterate the golden ratio algorithm (GRAAL) is the 
        following [4]_:

        .. math:: 
            \begin{align*}
                \mathbf{y}_{k+1} &= \frac{(\phi - 1)\mathbf{x}_k + \phi\mathbf{y}_k}
                {\phi} \\
                \mathbf{x}_{k+1} &= \text{prox}_{g,\mathcal{S}}(\mathbf{y}_{k+1} - \lambda 
                    F(\mathbf{x}_k))
            \end{align*}

        The convergence of GRAAL algorithm is guaranteed for Lipshitz monotone 
        operators, with Lipshitz constants :math:`L < +\infty`, when 
        :math:`\lambda \in \left(0,\frac{\varphi}{2L}\right]` and :math:`\phi 
        \in (1,\varphi]`, where :math:`\varphi = \frac{1+\sqrt{5}}{2}` is the 
        golden ratio.

        Arguments
        ---------
        x : ndarray
            The initial point, corresponding to :math:`\mathbf{x}_0`
        y : ndarray
            The initial auxiliary point, corresponding to :math:`\mathbf{y}_0`
        step_size : float
            The step size value, corresponding to :math:`\lambda`
        phi : float
            The golden ratio step size, corresponding to :math:`\phi`
        **cvxpy_solve_params
            The parameters for the 
            `cvxpy.Problem.solve <https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Problem.solve>`_ 
            method. 

        Yields
        ------
        ndarray
            The iteration's resulting point 

        References
        ----------
        .. [4] Malitsky, Y. (2020). Golden ratio algorithms for variational 
           inequalities. Mathematical Programming, 184(1), 383-410.
        """
        while True:
            y = ((phi - 1)*x + y)/phi
            x = self.prox(y - step_size*self.F(x), **cvxpy_solve_params)
            yield x

    def agraal(self,
            x_current: np.ndarray,
            x_previous: np.ndarray,
            step_size: float,
            phi: float = GOLDEN_RATIO,
            step_size_large: float = 1e6,
            **cvxpy_solve_params
        ) -> np.ndarray:
        r"""**Adaptive Golden Ratio Algorithm**

        The Adaptive Golden Ratio Algorithm (aGRAAL) algorithm is a variation 
        of the :func:`Golden Ratio Algorithm <monviso.core.VI.graal>`, with 
        adaptive step size. Following [5]_, let :math:`\theta_0 = 1`, 
        :math:`\rho = 1/\phi + 1/\phi^2`, where :math:`\phi \in (0,\varphi]` 
        and :math:`\varphi = \frac{1+\sqrt{5}}{2}` is the golden ratio. 
        Moreover, let :math:`\bar{\lambda} \gg 0` be a constant 
        (arbitrarily large) step-size. Given the initial terms 
        :math:`\mathbf{x}_0,\mathbf{x}_1 \in \mathbb{R}^n`, :math:`\mathbf{y}_0 = 
        \mathbf{x}_1`, and :math:`\lambda_0 > 0`, the :math:`k`-th iterate for 
        aGRAAL is the following:
         
        .. math::
            \begin{align*} 
            \lambda_k &= \min\left\{\rho\lambda_{k-1},
                  \frac{\phi\theta_k \|\mathbf{x}_k
                  -\mathbf{x}_{k-1}\|^2}{4\lambda_{k-1}\|F(\mathbf{x}_k)
                  -F(\mathbf{x}_{k-1})\|^2}, \bar{\lambda}\right\} \\
            \mathbf{y}_{k+1} &= \frac{(\phi - 1)\mathbf{x}_k + \phi\mathbf{y}_k}{\phi} \\
            \mathbf{x}_{k+1} &= \text{prox}_{g,\mathcal{S}}(\mathbf{y}_{k+1} - \lambda 
                F(\mathbf{x}_k)) \\
            \theta_k &= \phi\frac{\lambda_k}{\lambda_{k-1}} 
            \end{align*}

        The convergence guarantees discussed for GRAAL also hold for aGRAAL. 

        Arguments
        ---------
        x_current : ndarray
            The initial point, corresponding to :math:`\mathbf{x}_0`
        x_previous : ndarray
            The initial point, corresponding to :math:`\mathbf{x}_1`
        step_size : float
            The step size initial value, corresponding to :math:`\lambda_0`
        phi : float
            The golden ratio step size, corresponding to :math:`\phi`
        step_size_large : float, optional
            A constant (arbitrarily) large value for the step size
        **cvxpy_solve_params
            The parameters for the 
            `cvxpy.Problem.solve <https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Problem.solve>`_ 
            method. 

        Yields
        ------
        ndarray
            The iteration's resulting point 

        References
        ----------
        .. [5] Malitsky, Y. (2020). Golden ratio algorithms for variational 
           inequalities. Mathematical Programming, 184(1), 383-410.
        """
        rho = 1/phi + 1/phi**2
        theta = 1
        y = x_current

        while True:
            # lambda update
            step_size_current = np.min((
                rho*step_size,
                np.divide(
                    phi*theta*np.linalg.norm(x_current - x_previous, 2), 
                    4*step_size*np.linalg.norm(self.F(x_current) - self.F(x_previous), 2)
                ), step_size_large
            ))

            # graal step
            y = ((phi - 1)*x_current + y)/phi
            x = self.prox(y - step_size_current*self.F(x_current), **cvxpy_solve_params)

            theta = phi*step_size_current/step_size

            x_previous = x_current
            x_current = x
            step_size = step_size_current
            yield x

    def hgraal_1(self,
            x_current: np.ndarray,
            x_previous: np.ndarray,
            step_size: float,
            phi: float = GOLDEN_RATIO,
            step_size_large: float = 1e6,
            **cvxpy_solve_params
        ) -> np.ndarray:
        r"""**Hybrid Golden Ratio Algorithm I**

        The HGRAAL-1 algorithm is a variation of the 
        :func:`Adaptive Golden Ratio Algorithm <monviso.core.VI.agraal>`. 
        Following [13]_, let :math:`\theta_0 = 1`, :math:`\rho = 1/\phi + 
        1/\phi^2`, where :math:`\phi \in (0,\varphi]` and 
        :math:`\varphi = \frac{1+\sqrt{5}}{2}` is the golden ratio. 
        The residual at point :math:`\mathbf{x}_k` is given 
        by :math:`J : \mathbb{R}^n \to \mathbb{R}`, defined as follows:

        .. math:: J(\mathbf{x}_k) = \|\mathbf{x}_k - \text{prox}_{g,\mathcal{S}} 
            (\mathbf{x}_k - F(\mathbf{x}_k))\| 

        Moreover, let :math:`\bar{\lambda} \gg 0` be a constant (arbitrarily large) 
        step-size. Given the initial terms :math:`\mathbf{x}_0,\mathbf{x}_1 \in
        \mathbb{R}^n`, :math:`\mathbf{y}_0 = \mathbf{x}_1`, and :math:`\lambda_0 
        > 0`, the :math:`k`-th iterate for HGRAAL-1 is the following:

        .. math:: 
            \begin{align}
                \lambda_k &= \min\left\{\rho\lambda_{k-1},
                    \frac{\phi\theta_k \|\mathbf{x}_k
                    -\mathbf{x}_{k-1}\|^2}{4\lambda_{k-1}\|F(\mathbf{x}_k)
                    -F(\mathbf{x}_{k-1})\|^2}, \bar{\lambda}\right\} \\
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
                \mathbf{x}_{k+1} &= \text{prox}_{g,\mathcal{S}}(\mathbf{y}_{k+1} - \lambda 
                    F(\mathbf{x}_k)) \\
                \theta_k &= \phi\frac{\lambda_k}{\lambda_{k-1}} 
            \end{align}

        Arguments
        ---------
        x_current : ndarray
            The initial point, corresponding to :math:`\mathbf{x}_0`
        x_previous : ndarray
            The initial point, corresponding to :math:`\mathbf{x}_1`
        step_size : float
            The step size initial value, corresponding to :math:`\lambda_0`
        phi : float
            The golden ratio step size, corresponding to :math:`\phi`
        step_size_large : float, optional
            A constant (arbitrarily) large value for the step size
        **cvxpy_solve_params
            The parameters for the 
            `cvxpy.Problem.solve <https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Problem.solve>`_ 
            method. 

        Yields
        ------
        ndarray
            The iteration's resulting point 

        References
        ----------
        .. [13] Rahimi Baghbadorani, R., Mohajerin Esfahani, P., & Grammatico, S. 
           (2024). A hybrid algorithm for monotone variational inequalities. 
           (Manuscript submitted for publication).
        """
        rho = 1/phi + 1/phi**2
        theta = 1
        y = x_current

        flag = False
        k = 1

        # residual computation
        J_min = np.inf

        while True:
            # lambda update
            step_size_current = np.min((
                rho*step_size,
                np.divide(
                    phi*theta*np.linalg.norm(x_current - x_previous, 2), 
                    4*step_size*np.linalg.norm(self.F(x_current) - self.F(x_previous), 2) 
                ), step_size_large
            ))

            J_current, J_previous = self.residual(x_current), self.residual(x_previous)
            J_min = np.min((J_min, J_previous))

            condition = np.logical_or(
                np.logical_and(J_current - J_previous > 0, flag),
                J_min < J_current + 1/k
            )

            y = np.where(condition, ((phi - 1)*x_current + y)/phi, x_current)
            flag = not condition
            k += int(not condition)

            x = self.prox(y - step_size_current*self.F(x_current), **cvxpy_solve_params)

            theta = phi*step_size_current/step_size

            x_previous = x_current
            x_current = x
            step_size = step_size_current
            yield x

    def hgraal_2(self,
            x_current: np.ndarray,
            x_previous: np.ndarray,
            step_size: float,
            phi: float = GOLDEN_RATIO,
            alpha: float = GOLDEN_RATIO,
            step_size_large: float = 1e6,
            phi_large: float = 1e6,
            **cvxpy_solve_params
        ) -> np.ndarray:
        r"""**Hybrid Golden Ratio Algorithm II**

        The pseudo-code for the iteration schema can be fount at [14]_ [Algorithm 2]. 

        Arguments
        ---------
        x_current : ndarray
            The initial point, corresponding to :math:`\mathbf{x}_0`
        x_previous : ndarray
            The initial point, corresponding to :math:`\mathbf{x}_1`
        step_size : float
            The step size initial value, corresponding to :math:`\lambda_0`
        phi : float, optional
            The golden ratio step size, corresponding to :math:`\phi`
        alpha : float, optional
            The auxiliary parameter, corresponding to the :math:`\alpha` parameter.
        step_size_large : float, optional
            A constant (arbitrarily) large value for the step size
        phi_large: float, optional
            A constant (arbitrarily) large value for :math:`\phi`
        **cvxpy_solve_params
            The parameters for the 
            `cvxpy.Problem.solve <https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Problem.solve>`_ 
            method. 
        
        Yields
        ------
        ndarray
            The iteration's resulting point 

        References
        ----------
        .. [14] Rahimi Baghbadorani, R., Mohajerin Esfahani, P., & Grammatico, S. 
           (2024). A hybrid algorithm for monotone variational inequalities. 
           (Manuscript submitted for publication).
        """
        def s2_update(s2, coefficient):
            return s2 - step_size*phi*np.linalg.norm(x_current - y, 2)/step_size_current \
               + (step_size*phi/step_size_current - 1 - 1/coefficient)*np.linalg.norm(x - y, 2) \
               - (step_size*phi/step_size_current-theta)*np.linalg.norm(x - x_current, 2)

        rho = 1/phi + 1/phi**2
        theta_current = 1
        y_current = x_current
        step_size_current = step_size

        flag = False
        s1_current, s2_current = 0, 0

        while True:
            step_size = np.min((
                rho*step_size_current,
                np.divide(
                    alpha*theta_current*np.linalg.norm(x_current - x_previous, 2), 
                    4*step_size_current*np.linalg.norm(self.F(x_current) - self.F(x_previous), 2)
                ), step_size_large
            ))

            y = ((phi - 1)*x_current + y_current)/phi
            x = self.prox(y_current - step_size*self.F(x_current), **cvxpy_solve_params)
            theta = alpha*step_size/step_size_current

            s1 = s1_current + 0.5*theta_current*np.linalg.norm(x_current - x_previous, 2) \
               - step_size*np.linalg.norm(x_current - y, 2)/step_size_current \
               + (step_size*phi/step_size_current-1-1/phi_large)*np.linalg.norm(x - y, 2) \
               - (step_size*phi/step_size_current-1.5*theta)*np.linalg.norm(x - x_current, 2)

            s2 = s2_update(s2_current, phi_large)

            condition = np.logical_or(
                np.logical_and(s1 <= 0, flag),
                np.logical_and(s2 <= 0, not flag)
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
    ################################

    def solution(self,
            algorithm_name: str,
            algorithm_params: dict,
            max_iters: int,
            eval_func=None,
            eval_tol: float = 1e-9,
            log_path: str = None,
            **cvxpy_solve_params
        ) -> np.ndarray:
        """
        Solve the variational inequality, using the indicated algorithm. 

        Arguments
        ---------
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
            The parameters for the 
            `cvxpy.Problem.solve <https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Problem.solve>`_ 
            method. 

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
                log_file.write(f"{k},{eval_func_value},{time.process_time() - init_time}\n")

                # stopping criteria
                if k >= max_iters-1 or eval_func_value <= eval_tol:
                    return x
