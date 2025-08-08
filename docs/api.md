{% set cvxpy_params = "`**cvxpy_solve_params` – The parameters for the [`cvxpy.Problem.solve`](https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Problem.solve) method." %}
{% set step_size = "**`step_size`** (`float`) – The steps size value, corresponding to $\chi$." %}
{% set step_size_large = "**`step_size_large`** (`float`) – A constant (arbitrarily) large value for the step size, corresponding to $\\bar{\chi}$." %}

{% set ck = "**`ck`** (`int`) – The current counting parameter, corresponding to $c_k$" %}
{% set ck1 = "**`ck1`** (`int`) – The next counting parameter, corresponding to $c_{k+1}$" %}

{% set alpha = "**`alpha`** (`float`, optional) – The auxiliary parameter, corresponding to $\\alpha$." %}
{% set phi = "**`phi`** (`float`, optional) – The golden ratio step size, corresponding to $\phi$." %}

{% set tk = "**`tk`** (`float`, optional) The current auxiliary coefficient, corresponding to $\\theta_k$." %}
{% set tk1 = "**`xk1`** (`ndarray`) – The next auxiliary coefficient, corresponding to $\\theta_{k+1}$." %}

{% set x0 = "**`x0`** (`ndarray`) – The initial point, corresponding to $\mathbf{x}_0$." %} 
{% set x1k = "**`x1k`** (`ndarray`) – The previous point, corresponding to $\mathbf{x}_{k-1}$." %}
{% set xk = "**`xk`** (`ndarray`) – The current point, corresponding to $\mathbf{x}_k$." %} 
{% set xk1 = "**`xk1`** (`ndarray`) – The next point, corresponding to $\mathbf{x}_{k+1}$." %}

{% set y0 = "**`y0`** (`ndarray`) – The initial auxiliary point, corresponding to $\mathbf{y}_0$" %} 
{% set y1k = "**`y1k`** (`ndarray`) – The previous point, corresponding to $\mathbf{y}_{k-1}$" %}
{% set yk = "**`yk`** (`ndarray`) – The current auxiliary point, corresponding to $\mathbf{y}_k$" %} 
{% set yk1 = "**`yk1`** (`ndarray`) – The next auxiliary point, corresponding to $\mathbf{y}_{k+1}$" %}

{% set s1k = "**`s1k`** (`float`) – The previous step-size, corresponding to $s_{k-1}$" %}
{% set sk = "**`sk`** (`float`) – The current steps-size, corresponding to $s_k$" %} 

{% set z0 = "**`z0`** (`ndarray`) – The initial auxiliary point, corresponding to $\mathbf{z}_0$" %} 
{% set z1k = "**`z1k`** (`ndarray`) – The previous point, corresponding to $\mathbf{z}_{k-1}$" %}
{% set zk = "**`zk`** (`ndarray`) – The current auxiliary point, corresponding to $\mathbf{z}_k$" %} 
{% set zk1 = "**`zk1`** (`ndarray`) – The next auxiliary point, corresponding to $\mathbf{z}_{k+1}$" %}

## The `VI` Class
The `VI` class defines the variational inequality. It is characterized by the vector mapping `F`and the `prox` operator. Both take and return a `np.ndarray`. `F` is a callable defined and passed directly by the user, while the `prox` operator can be defined in different ways:

- by passing the `analytical_prox` callable, which takes and returns a `np.ndarray`
- by defining a `y`, which is a `cp.Variable` and, optionally:

    - a callable `g`, taking a `cp.Variable` as argument and retuning a scalar. If not define, it will default to `g = 0`. 
    - a constraint set `S`, with each element being a `cp.Constraint`. If not defined, it will default to `S = []`.

::: monviso.VI
    options:
        members: false

## Iterative methods 

The convention used for naming indexed terms is the following:

- Indexed terms' names are single letters
- `xk` stands for $x_k$
- `xkn` stands for $x_{k+n}$ 
- `xnk` stands for $x_{k-n}$

Therefore, as examples, `y0` is $y_0$, `tk` is $t_k$, `z1k` is $z_{k-1}$, and `sk2` is $s_{k+2}$.
The vector corresponding to the decision variable of the VI is always denoted with $\mathbf{x}$; all other vectors that might be used and / returned are generically referred to as *auxiliary points*.

### Proximal gradient
Given a constant step-size $\chi > 0$ and an initial vector $\mathbf{x}_0 \in \mathbb{R}^n$, the basic $k$-th iterate of the proximal gradient (PG) algorithm is [^1]:

$$ \mathbf{x}_{k+1} = \text{prox}_{g,\mathcal{S}}(\mathbf{x}_k - \chi \mathbf{F}(\mathbf{x}_k)) $$

where $g : \mathbb{R}^n \to \mathbb{R}$ is a scalar convex (possibly non-smooth) function, while $\mathbf{F} : \mathbb{R}^n \to \mathbb{R}^n$ is the VI mapping. Convergence of PG is guaranteed for Lipschitz strongly monotone operators, with monotone constant $\mu > 0$ and Lipschitz constants $L < +\infty$, when $\chi \in (0, 2\mu/L^2)$.

[^1]: Nemirovskij, A. S., & Yudin, D. B. (1983). Problem complexity and method efficiency in optimization.

::: monviso.VI.pg

**Parameters**:

- {{ xk }}
- {{ step_size }}
- {{ cvxpy_params }}

**Returns**:

- {{ xk1 }}


### Extragradient
Given a constant step-size $\chi > 0$ and an initial vector $\mathbf{x}_0 \in \mathbb{R}^n$, the $k$-th iterate of the extragradient algorithm (EG) is[^2]:

$$ 
\begin{align*}
    \mathbf{y}_k &= \text{prox}_{g,\mathcal{S}}(\mathbf{x}_k - 
        \chi \mathbf{F}(\mathbf{x}_k)) \\
    \mathbf{x}_{k+1} &= \text{prox}_{g,\mathcal{S}}(\mathbf{y}_k - 
        \chi \mathbf{F}(\mathbf{x}_k))
\end{align*}
$$

where $g : \mathbb{R}^n \to \mathbb{R}$ is a scalar convex (possibly non-smooth) function, while $\mathbf{F} : \mathbb{R}^n \to \mathbb{R}^n$ is the VI mapping. The convergence of the EGD algorithm is guaranteed for Lipschitz monotone operators, with Lipschitz constant $L < +\infty$, when $\chi \in \left(0,\frac{1}{L}\right)$.

[^2]: Korpelevich, G. M. (1976). The extragradient method for finding 
   saddle points and other problems. Matecon, 12, 747-756.

::: monviso.VI.eg

**Parameters**:

- {{ xk }}
- {{ step_size }}
- {{ cvxpy_params }}

**Returns**:

- {{ xk1 }} 


### Popov's method
Given a constant step-size $\chi > 0$ and an initial vectors $\mathbf{x}_0,\mathbf{y}_0 \in \mathbb{R}^n$, the $k$-th iterate of Popov's Method (PM) is[^3]:

$$ 
\begin{align*}
    \mathbf{y}_{k+1} &= \text{prox}_{g,\mathcal{S}}(\mathbf{x}_k - \chi \mathbf{F}(\mathbf{y}_k)) \\
    \mathbf{x}_{k+1} &= \text{prox}_{g,\mathcal{S}}(\mathbf{y}_{k+1} - \chi \mathbf{F}(\mathbf{x}_k))
\end{align*}
$$

where $g : \mathbb{R}^n \to \mathbb{R}$ is a scalar convex (possibly non-smooth) function, while $\mathbf{F} : \mathbb{R}^n \to \mathbb{R}^n$ is the VI mapping. The convergence of PM is guaranteed for Lipschitz monotone operators, with Lipschitz constant $L < +\infty$, when $\chi \in \left(0,\frac{1}{2L}\right)$.

[^3]: Popov, L.D. A modification of the Arrow-Hurwicz method for search of saddle points. Mathematical Notes of the Academy of Sciences of the USSR 28, 845–848 (1980)

::: monviso.VI.popov

**Parameters**:

- {{ xk }} 
- {{ yk }}
- {{ step_size }}
- {{ cvxpy_params }}

**Returns**:

- {{ xk1 }}
- {{ yk1 }}


### Forward-backward-forward
Given a constant step-size $\chi > 0$ and an initial vector $\mathbf{x}_0 \in \mathbb{R}^n$, the $k$-th iterate of Forward-Backward-Forward (FBF) algorithm is[^4]:

$$ 
\begin{align*}
    \mathbf{y}_k &= \text{prox}_{g,\mathcal{S}}(\mathbf{x}_k - \chi \mathbf{F}(\mathbf{x}_k)) \\
    \mathbf{x}_{k+1} &= \mathbf{y}_k - \chi \mathbf{F}(\mathbf{y}_k) + \chi \mathbf{F}(\mathbf{x}_k)
\end{align*}
$$

where $g : \mathbb{R}^n \to \mathbb{R}$ is a scalar convex (possibly non-smooth) function, while $\mathbf{F} : \mathbb{R}^n \to \mathbb{R}^n$ is the VI mapping. The convergence of the FBF algorithm is guaranteed for Lipschitz monotone operators, with Lipschitz constant $L < +\infty$, when $\chi \in \left(0,\frac{1}{L}\right)$.

[^4]: Tseng, P. (2000). A modified forward-backward splitting method for maximal monotone mappings. SIAM Journal on Control and Optimization, 38(2), 431-446.

::: monviso.VI.fbf

**Parameters**:

- {{ xk }}
- {{ step_size }}
- {{ cvxpy_params }}

**Returns**:

- {{ xk1 }}


### Forward-reflected-backward
Given a constant step-size $\chi > 0$ and initial vectors $\mathbf{x}_1,\mathbf{x}_0 \in \mathbb{R}^n$, the basic $k$-th iterate of the Forward-Reflected-Backward (FRB) is the following[^5]:

$$ \mathbf{x}_{k+1} = \text{prox}_{g,\mathcal{S}}(\mathbf{x}_k - \chi (2\mathbf{F}(\mathbf{x}_k) + \mathbf{F}(\mathbf{x}_{k-1}))) $$

where $g : \mathbb{R}^n \to \mathbb{R}$ is a scalar convex (possibly non-smooth) function, while $\mathbf{F} : \mathbb{R}^n \to \mathbb{R}^n$ is the VI mapping. The convergence of the FRB algorithm is guaranteed for Lipschitz monotone operators, with Lipschitz constant $L < +\infty$, when $\chi \in \left(0,\frac{1}{2L}\right)$.

[^5]: Malitsky, Y., & Tam, M. K. (2020). A forward-backward splitting method for monotone inclusions without cocoercivity. SIAM Journal on Optimization, 30(2), 1451-1472.

::: monviso.VI.frb

**Parameters**:

- {{ xk }}
- {{ x1k }}
- {{ step_size }}
- {{ cvxpy_params }}

**Returns**:

- {{ xk1 }}


### Projected reflected gradient
Given a constant step-size $\chi > 0$ and initial vectors $\mathbf{x}_1,\mathbf{x}_0 \in \mathbb{R}^n$, the basic $k$-th iterate of the projected reflected gradient (PRG) is the following [^6]:

$$ \mathbf{x}_{k+1} = \text{prox}_{g,\mathcal{S}}(\mathbf{x}_k - \chi \mathbf{F}(2\mathbf{x}_k - \mathbf{x}_{k-1})) $$

where $g : \mathbb{R}^n \to \mathbb{R}$ is a scalar convex (possibly non-smooth) function, while $\mathbf{F} : \mathbb{R}^n \to \mathbb{R}^n$ is the VI mapping. The convergence of PRG algorithm is guaranteed for Lipschitz monotone operators, with Lipschitz constants $L < +\infty$, when $\chi \in (0,(\sqrt{2} - 1)/L)$. Differently from the EGD iteration, the PRGD has the advantage of requiring a single proximal operator evaluation.

[^6]: Malitsky, Y. (2015). Projected reflected gradient methods for monotone variational inequalities. SIAM Journal on Optimization, 25(1), 502-520.

::: monviso.VI.prg

**Parameters**:

- {{ xk }}
- {{ x1k }}
- {{ step_size }}
- {{ cvxpy_params }}

**Returns**:

- {{ xk1 }}

### Extra anchored gradient
Given a constant step-size $\chi > 0$ and an initial vector $\mathbf{x}_0 \in \mathbb{R}^n$, the $k$-th  iterate of extra anchored gradient (EAG) algorithm is [^7]:

$$
\begin{align*}
    \mathbf{y}_k &= \text{prox}_{g,\mathcal{S}}\left(\mathbf{x}_k - 
        \chi \mathbf{F}(\mathbf{x}_k) + \frac{1}{k+1}(\mathbf{x}_0 - 
        \mathbf{x}_k)\right) \\
    \mathbf{x}_{k+1} &= \text{prox}_{g,\mathcal{S}}\left(\mathbf{x}_k - 
        \chi \mathbf{F}(\mathbf{y}_k) + \frac{1}{k+1}(\mathbf{x}_0 - 
        \mathbf{x}_k)\right)
\end{align*}
$$

where $g : \mathbb{R}^n \to \mathbb{R}$ is a scalar convex  (possibly non-smooth) function, while $\mathbf{F} : \mathbb{R}^n \to \mathbb{R}^n$ is the VI mapping. The convergence of the EAG algorithm is guaranteed for Lipschitz monotone operators, with Lipschitz constant $L < +\infty$, when $\chi \in \left(0,\frac{1}{\sqrt{3}L} \right)$.

[^7]: Yoon, T., & Ryu, E. K. (2021, July). Accelerated Algorithms for Smooth Convex-Concave Minimax Problems with O (1/k^ 2) Rate on Squared Gradient Norm. In International Conference on Machine Learning (pp. 12098-12109). PMLR.

::: monviso.VI.eag

**Parameters**:

- {{ xk }}
- {{ x0 }}
- {{ k }}
- {{ step_size }}
- {{ cvxpy_params }}

**Returns**

- {{ xk1 }}


### Accelerated reflected gradient
Given a constant step-size $\chi > 0$ and initial vectors $\mathbf{x}_1,\mathbf{x}_0 \in \mathbb{R}^n$, the basic $k$-th iterate of the accelerated reflected gradient (ARG) is the following[^8]:

$$
\begin{align*}
    \mathbf{y}_k &= 2\mathbf{x}_k - \mathbf{x}_{k-1} + \frac{1}{k+1}
    (\mathbf{x}_0 - \mathbf{x}_k) - \frac{1}{k}(\mathbf{x}_k - 
    \mathbf{x}_{k-1}) \\
    \mathbf{x}_{k+1} &= \text{prox}_{g,\mathcal{S}}\left(\mathbf{x}_k - 
        \chi \mathbf{F}(\mathbf{y}_k) + \frac{1}{k+1}(\mathbf{x}_0 - 
        \mathbf{x}_k)\right)
\end{align*}
$$

where $g : \mathbb{R}^n \to \mathbb{R}$ is a scalar convex (possibly non-smooth) function, while $\mathbf{F} : \mathbb{R}^n \to \mathbb{R}^n$ is the VI mapping. The convergence of the ARG algorithm is guaranteed for Lipschitz monotone operators, with Lipschitz constant $L < +\infty$, when $\chi \in \left(0,\frac{1}{12L}\right)$.

[^8]: Cai, Y., & Zheng, W. (2022). Accelerated single-call methods for constrained min-max optimization. arXiv preprint arXiv:2210.03096.

::: monviso.VI.arg

**Parameters**:

- {{ xk }} 
- {{ x1k }}
- {{ x0 }}
- {{ k }}
- {{ step_size }}
- {{ cvxpy_params }}

**Returns**:

- {{ xk1 }}


### (Explicit) fast optimistic gradient descent-ascent
Given a constant step-size $\chi > 0$ and initial vectors $\mathbf{x}_1,\mathbf{x}_0,\mathbf{y}_0 \in \mathbb{R}^n$, the basic $k$-th iterate of the explicit fast OGDA (FOGDA) is the following [^9]:

$$
\begin{align*}
    \mathbf{y}_k &= \mathbf{x}_k + \frac{k}{k+\alpha}(\mathbf{x}_k - 
        \mathbf{x}_{k-1}) - \chi \frac{\alpha}{k+\alpha}
        \mathbf{F}(\mathbf{y}_{k-1}) \\
    \mathbf{x}_{k+1} &= \mathbf{y}_k - \chi \frac{2k+\alpha}
        {k+\alpha} (\mathbf{F}(\mathbf{y}_k) - \mathbf{F}(\mathbf{y}_{k-1}))
\end{align*}
$$

where $g : \mathbb{R}^n \to \mathbb{R}$ is a scalar convex (possibly non-smooth) function, while $\mathbf{F} : \mathbb{R}^n \to \mathbb{R}^n$ is the VI mapping. The convergence of the ARG algorithm is guaranteed for Lipschitz monotone operators, with Lipschitz constant $L < +\infty$, when $\chi \in \left(0,\frac{1}{4L}\right)$ and $\alpha > 2$.

[^9]: Boţ, R. I., Csetnek, E. R., & Nguyen, D. K. (2023). Fast Optimistic Gradient Descent Ascent (OGDA) method in continuous and discrete time. Foundations of Computational Mathematics, 1-60.

::: monviso.VI.fogda

**Parameters**:

- {{ xk }}
- {{ x1k }}
- {{ y1k }}
- {{ k }}
- {{ step_size }}
- {{ alpha }}
- {{ cvxpy_params }}

**Returns**:

- {{ xk1 }}
- {{ yk }}

### Constrained fast optimistic gradient descent-ascent
Given a constant step-size $\chi > 0$ and initial vectors $\mathbf{x}_1 \in \mathcal{S}$, $\mathbf{z}_1 \in N_{\mathcal{S}}(\mathbf{x}_1)$, $\mathbf{x}_0,\mathbf{y}_0 \in \mathbb{R}^n$, the basic $k$-th iterate of Constrained Fast Optimistic Gradient Descent Ascent (CFOGDA) is the following[^10]:

$$ 
\begin{align*}
    \mathbf{y}_k &= \mathbf{x}_k + \frac{k}{k+\alpha}(\mathbf{x}_k - \mathbf{x}_{k-1}) - \chi \frac{\alpha}{k+\alpha}(\mathbf{F}(\mathbf{y}_{k-1}) + \mathbf{z}_k) \\
    \mathbf{x}_{k+1} &= \text{prox}_{g,\mathcal{S}}\left(\mathbf{y}_k - \chi\left(1 + \frac{k}{k+\alpha}\right)(\mathbf{F}(\mathbf{y}_k) - \mathbf{F}(\mathbf{y}_{k-1}) - \zeta_k)\right) \\
    \mathbf{z}_{k+1} &= \frac{k+\alpha}{\chi (2k+\alpha)}( \mathbf{y}_k - \mathbf{x}_{k+1}) - (\mathbf{F}(\mathbf{y}_k) - \mathbf{F}(\mathbf{y}_{k-1}) - \zeta_k)
\end{align*}
$$

where $g : \mathbb{R}^n \to \mathbb{R}$ is a scalar convex (possibly non-smooth) function, while $\mathbf{F} : \mathbb{R}^n \to \mathbb{R}^n$ is the VI mapping. The convergence of the CFOGDA algorithm is guaranteed for Lipschitz monotone operators, with Lipschitz constant $L < +\infty$, when $\chi \in \left(0,\frac{1}{4L}\right)$ and $\alpha > 2$.

[^10]: Sedlmayer, M., Nguyen, D. K., & Bot, R. I. (2023, July). A fast optimistic method for monotone variational inequalities. In International Conference on Machine Learning (pp. 30406-30438). PMLR.

::: monviso.VI.cfogda

**Parameters**:

- {{ xk }}
- {{ x1k }} 
- {{ y1k }}
- {{ zk }}
- {{ k }}
- {{ step_size }}
- {{ alpha }}
- {{ cvxpy_params }}

**Returns**:

- {{ xk1 }}
- {{ yk }}
- {{ zk1 }}

### Golden ratio algorithm
Given a constant step-size $\chi > 0$ and initial vectors $\mathbf{x}_0,\mathbf{y}_0 \in \mathbb{R}^n$, the basic $k$-th iterate the golden ratio algorithm (GRAAL) is the following [^11]:

$$ 
\begin{align*}
    \mathbf{y}_{k+1} &= \frac{(\phi - 1)\mathbf{x}_k + \phi\mathbf{y}_k}{\phi} \\
    \mathbf{x}_{k+1} &= \text{prox}_{g,\mathcal{S}}(\mathbf{y}_{k+1} - \chi \mathbf{F}(\mathbf{x}_k))
\end{align*}
$$

The convergence of GRAAL algorithm is guaranteed for Lipschitz monotone operators, with Lipschitz constants $L < +\infty$, when $\chi \in \left(0,\frac{\varphi}{2L}\right]$ and $\phi \in (1,\varphi]$, where $\varphi = \frac{1+\sqrt{5}}{2}$ is the golden ratio.

[^11]: Malitsky, Y. (2020). Golden ratio algorithms for variational inequalities. Mathematical Programming, 184(1), 383-410.

::: monviso.VI.graal

**Parameters**:

- {{ xk }}
- {{ yk }}
- {{ step_size }}
- {{ phi }}
- {{ cvxpy_params }}

**Returns**:

- {{ xk1 }}
- {{ yk1 }}

### Adaptive golden ratio algorithm
The Adaptive Golden Ratio Algorithm (aGRAAL) algorithm is a variation of the Golden Ratio Algorithm ([monviso.VI.graal][]), with adaptive step size. 
Following [^12], let $\theta_0 = 1$, $\rho = 1/\phi + 1/\phi^2$, where $\phi \in (0,\varphi]$ and $\varphi = \frac{1+\sqrt{5}}{2}$ is the golden ratio. 
Moreover, let $\bar{\chi} \gg 0$ be a constant (arbitrarily large) step-size. 
Given the initial terms $\mathbf{x}_0,\mathbf{x}_1 \in \mathbb{R}^n$, $\mathbf{y}_0 = \mathbf{x}_1$, and $\chi_0 > 0$, the $k$-th iterate for aGRAAL is the following:
 
$$
\begin{align*} 
\chi_k &= \min\left\{\rho\chi_{k-1},
      \frac{\phi\theta_k \|\mathbf{x}_k
      -\mathbf{x}_{k-1}\|^2}{4\chi_{k-1}\|\mathbf{F}(\mathbf{x}_k)
      -\mathbf{F}(\mathbf{x}_{k-1})\|^2}, \bar{\chi}\right\} \\
\mathbf{x}_{k+1}, \mathbf{y}_{k+1} &= \texttt{graal}(\mathbf{x}_k, \mathbf{y}_k, \chi_k, \phi) \\
\theta_{k+1} &= \phi\frac{\chi_k}{\chi_{k-1}} 
\end{align*}
$$

The convergence guarantees discussed for GRAAL also hold for aGRAAL. 

[^12]: Malitsky, Y. (2020). Golden ratio algorithms for variational inequalities. Mathematical Programming, 184(1), 383-410.

::: monviso.VI.agraal

**Parameters**:

- {{ xk }}
- {{ x1k }}
- {{ yk }}
- {{ s1k }}
- {{ tk }}
- {{ step_size_large }}
- {{ phi }}
- {{ cvxpy_params }}

**Returns**:

- {{ xk1 }}
- {{ yk1 }} 
- {{ sk }} 
- {{ tk1 }}

### Hybrid golden ratio algorithm I 
The HGRAAL-1 algorithm[^13] is a variation of the Adaptive Golden Ratio Algorithm ([monviso.VI.agraal][]). 
Let $\theta_0 = 1$, $\rho = 1/\phi + 1/\phi^2$, where $\phi \in (0,\varphi]$ and $\varphi = \frac{1+\sqrt{5}}{2}$ is the golden ratio. 
The residual at point $\mathbf{x}_k$ is given by $J : \mathbb{R}^n \to \mathbb{R}$, defined as follows:

$$ J(\mathbf{x}_k) = \|\mathbf{x}_k - \text{prox}_{g,\mathcal{S}} (\mathbf{x}_k - \mathbf{F}(\mathbf{x}_k))\| $$

Moreover, let $\bar{\chi} \gg 0$ be a constant (arbitrarily large) step-size. 
Given the initial terms $\mathbf{x}_0,\mathbf{x}_1 \in\mathbb{R}^n$, $\mathbf{y}_0 = \mathbf{x}_1$, and $\chi_0 > 0$, the $k$-th iterate for HGRAAL-1 is the following:

$$ 
\begin{align*}
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
    \mathbf{x}_{k+1} &= \text{prox}_{g,\mathcal{S}}(\mathbf{y}_{k+1} - \chi_k 
        \mathbf{F}(\mathbf{x}_k)) \\
    \theta_{k+1} &= \phi\frac{\chi_k}{\chi_{k-1}} 
\end{align*}
$$

[^13]: Rahimi Baghbadorani, R., Mohajerin Esfahani, P., & Grammatico, S. (2024). A hybrid algorithm for monotone variational inequalities. 
(Manuscript submitted for publication).

::: monviso.VI.hgraal_1

**Parameters**:

- {{ xk }}
- {{ x1k }}
- {{ yk }}
- {{ s1k }}
- {{ tk }}
- {{ ck }}
- {{ step_size_large }}
- {{ phi }}

**Returns**:

- {{ xk1 }} 
- {{ yk1 }}
- {{ sk }}
- {{ tk1 }}
- {{ ck1 }}
