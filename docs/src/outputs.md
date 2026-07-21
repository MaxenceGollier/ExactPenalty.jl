# Outputs

`L2Penalty` (and the lower-level `solve!` calls) return a `GenericExecutionStats` object, as defined in [SolverCore.jl](https://github.com/JuliaSmoothOptimizers/SolverCore.jl). This section describes the fields of that object as populated by `ExactPenalty.jl`, as well as the console output produced when `print_level > 0`.

```{julia}
julia> stats = L2Penalty(nlp)
```

## The `GenericExecutionStats` Object

* `stats.status::Symbol`: exit status of the algorithm. See [Status](#status) below for the list of possible values.

* `stats.solution::V`: the final iterate $x_k$.

* `stats.objective::T`: the value of $f(x_k)$ at the final iterate.

* `stats.primal_feas::T`: the primal feasibility measure $\lVert c(x_k) \rVert_{\infty}$.

* `stats.dual_feas::T`: the dual feasibility measure $\lVert \nabla f(x_k) + J(x_k)^T y_k \rVert_{\infty}$.

* `stats.multipliers::V`: the vector of Lagrange multiplier estimates $y_k$ at the final iterate.

* `stats.iter::Int`: the number of *outer* (penalty-loop) iterations performed.

* `stats.elapsed_time::Float64`: total elapsed (CPU) time, in seconds.

* `stats.solver_specific::Dict{Symbol,T}`: a dictionary of algorithm-specific quantities, described next.

### `solver_specific` Entries

!!! note "Terminology"
    See the [terminology](options.md#Terminology) section of the options page for an explanation of the *outer*, *R2N*, and *Moré-Sorensen* loops.

* `:n_fact::Int`: the cumulative number of matrix factorizations performed by the linear solver over the whole solve.

* `:tau::T`: the final value of the penalty parameter $\tau_k$.

* `:sigma::T`: the final value of the R2N regularization parameter $\sigma_k$.

* `:rho::T`: the ratio of actual to predicted decrease $\rho_k$ at the last accepted R2N step.

* `:primal_ktol::T`: the primal feasibility tolerance imposed on the last-solved penalized subproblem.

* `:dual_ktol::T`: the dual feasibility tolerance imposed on the last-solved penalized subproblem.

## Status

`stats.status` can take one of the following values:


|
 Status 
|
 Meaning 
|
|
:------:
|
:--------
|
|
`:first_order`
|
 A first-order stationary point was found within tolerance. 
|
|
`:infeasible`
|
 The problem was declared locally infeasible: see 
`infeasible_tol`
 and 
`infeasible_iter`
 in the 
[
options
](
options.md
)
. 
|
|
`:unbounded`
|
 The objective appears to be unbounded below. 
|
|
`:not_desc`
|
 The Moré–Sorensen subsolver could not certify a descent step; see 
`ms_accept_descent`
. 
|
|
`:small_step`
|
 The computed step was numerically insignificant; see 
`r2n_tiny_step_tol`
. 
|
|
`:max_iter`
|
 The iteration limit 
`max_iter`
 was reached. 
|
|
`:max_time`
|
 The time limit 
`max_time`
 was reached. 
|
|
`:max_eval`
|
 The objective evaluation limit 
`max_eval`
 was reached. 
|
|
`:exception`
|
 An internal exception occurred (e.g., the Moré–Sorensen regularization parameter exceeded 
`ms_σmax`
). 
|
|
`:unknown`
|
 The algorithm has not (yet) satisfied any stopping criterion; this should not appear in a returned 
`stats`
 object. 
|

!!! warning "Infeasibility is a Heuristic"
    A `:infeasible` status means the algorithm *detected* apparent local infeasibility using the heuristic described by `infeasible_tol`; it is not a certificate of infeasibility.

## Console Output

Setting the keyword argument `print_level` to a value $\geq 1$ turns on console logging. Because the solver is organized as nested loops (see [terminology](options.md#Terminology)), `print_level` also controls *how deep* the logging goes:


|
`print_level`
|
 Loops logged 
|
|
:--------------:
|
:-------------
|
|
`0`
|
 none (default) 
|
|
`1`
|
 outer (penalty) loop 
|
|
`2`
|
 outer + R2N loop 
|
|
`3`
|
 outer + R2N + Moré–Sorensen loop 
|

The frequency (in iterations) at which each loop prints a line is controlled independently by `verbose`, `r2n_verbose`, and `ms_verbose`.

### Outer-Loop Header

Iter sIter Objective pfeas dfeas τ ptol dtol ‖x‖


* `Iter`: outer iteration counter.
* `sIter`: number of R2N (inner) iterations performed to solve the current penalized subproblem.
* `Objective`: current value of $f(x_k)$.
* `pfeas`: primal feasibility $\lVert c(x_k) \rVert_{\infty}$.
* `dfeas`: dual feasibility $\lVert \nabla f(x_k) + J(x_k)^Ty_k \rVert_{\infty}$.
* `τ`: current penalty parameter.
* `ptol`: primal feasibility tolerance for the current subproblem.
* `dtol`: dual feasibility tolerance for the current subproblem.
* `‖x‖`: norm of the current iterate.

### Inner (R2N)-Loop Header

| Iter sIter Objective pfeas dfeas σ ρ ‖x‖ ‖s‖


* `Iter`: R2N iteration counter (within the current penalized subproblem).
* `sIter`: number of Moré–Sorensen iterations used to compute the current step.
* `Objective`: current value of the penalized objective $f(x) + \tau_k \lVert c(x) \rVert_2$.
* `pfeas`, `dfeas`: as above, but with respect to the current penalized subproblem.
* `σ`: current R2N regularization parameter.
* `ρ`: ratio of actual to predicted decrease for the last step.
* `‖x‖`: norm of the current inner iterate.
* `‖s‖`: norm of the last computed step.

A typical excerpt of console output (with `print_level = 2`) looks like:
...

!!! tip "Verifying Your Linear Solver Choice"
    If you set the `linear_solver` [option](options.md#Linear-Solver) and want to confirm it is actually being used, run with `print_level ≥ 1`: the introduction message printed at the start of the solve reports the linear solver library in use.