# Options Reference

Options that should be modified by expert users only are marked as *advanced*.

## Termination

We declare an iterate $(x_k, y_k)$ optimal when it satisfies
```math
\lVert \nabla f(x_k) + J(x_k)^T y_k \rVert_{\infty} \leq \epsilon_D
\quad \text{and} \quad
\lVert c(x_k) \rVert_{\infty} \leq \epsilon_P.
```
We define $\epsilon_P$ and $\epsilon_D$ as 
```math
\begin{align*}
  \epsilon_P &= \max(\epsilon_P^a, \epsilon^a) + \max(\epsilon_P^r, \epsilon^r) \lVert c(x_0) \rVert_{\infty}, \\
  \epsilon_D &= \max(\epsilon_D^a, \epsilon^a) + \max(\epsilon_D^r, \epsilon^r) \lVert \nabla f(x_0) + J(x_0)^T y_0 \rVert_{\infty}.
\end{align*}
```
* `atol::T = √eps(T)`: $\epsilon^a$ -- Desired convergence tolerance (absolute).

  > Absolute tolerance for primal and dual residuals. When `T == Float64`, the default value is $\approx 10^{-8}$.

* `rtol::T = √eps(T)`: $\epsilon^r$ -- Desired convergence tolerance (relative).

  > Relative tolerance for primal and dual residuals. When `T == Float64`, the default value is $\approx 10^{-8}$.

* `dual_inf_atol::T = 0`: $\epsilon^a_D$ -- Desired convergence tolerance (absolute) for dual infeasibility.
  
* `dual_inf_rtol::T = 0`: $\epsilon^r_D$ -- Desired convergence tolerance (relative) for dual infeasibility.

* `primal_inf_atol::T = 0`: $\epsilon^a_P$ -- Desired convergence tolerance (absolute) for primal infeasibility.

* `primal_inf_rtol::T = 0`: $\epsilon^r_P$ -- Desired convergence tolerance (relative) for primal infeasibility.

* `max_eval::Int = -1`: Maximum number of objective function evaluation.

  > The solver stops when the number of objective function evaluations exceeds `max_eval`. A negative value means unlimited.

* `max_time::Float64 = 30.0`: Maximum number of CPU seconds.

  > The solver stops when the number of CPU seconds exceeds `max_time`.

* `outer_loop_max_iter::Int = 10000`: Maximum number of *outer-loop* iterations.

  >

* `inner_loop_max_iter::Int = 10000`: Maximum number of *inner-loop* iterations.

  >

## Output

TODO

## NLP

TODO

## Initialization

TODO

## Penalty Parameter Update

TODO

## Quadratic Regularization

TODO

## Linear Solver

TODO

## Hessian Approximation