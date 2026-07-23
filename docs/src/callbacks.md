# Callbacks
 
`ExactPenalty.jl` lets you hook into the solve at each iteration through a
*callback* function. This is useful to log custom information, record
history, plot progress in real time, or implement your own stopping
criterion.
 
!!! note "Terminology"
    See the [terminology](options.md#Terminology) section of the options
    page for an explanation of the *outer* (penalty), *R2N*, and
    *Moré–Sorensen* loops.
 
## The Outer-Loop Callback
 
The expected signature is
```julia
callback(nlp, solver, stats)
```
and its return value is ignored. The callback is called once before the
first iteration and once after every subsequent iteration, right before the
stopping criteria are checked, so any modification you make to `solver` or
`stats` takes effect immediately.
 
```julia
using ExactPenalty
 
function my_callback(nlp, solver, stats)
  push!(objectives, stats.objective)
end
 
objectives = Float64[]
stats = L2Penalty(nlp, callback = my_callback)
```
 
### Stopping the Algorithm Early
 
Setting
```julia
stats.status = :user
```
inside the callback will cause the algorithm to stop immediately after the
callback returns. Use this to implement custom stopping criteria (e.g., a
target objective value, a wall-clock budget managed externally, or an
interactive "stop" signal).
 
```julia
function stop_early(nlp, solver, stats)
  if stats.objective < 1e-3
    stats.status = :user
  end
end
```
 
### What You Can Access and Modify
 
All the information relevant to the current state of the algorithm is
available through `nlp`, `solver`, and `stats`. In particular:
 
* `solver.x`: the current outer iterate $x_k$. You may modify it, though
  doing so will affect subsequent iterations, so use with care.
* `solver.subsolver`: a `PenaltyR2NSolver` structure holding the
  state of the R2N subsolver used to (approximately) minimize the current
  penalized subproblem $f(x) + \tau_k \|c(x)\|_2$. See the [inner-loop
  callback](#The-Inner-Loop-Callback) section below for how to hook into
  this subsolver directly, and the [options](options.md#Terminology) page
  for the overall solver structure.
* `stats`: the [`GenericExecutionStats`](https://github.com/JuliaSmoothOptimizers/SolverCore.jl)
  object that will eventually be returned by `L2Penalty`. Notably:
  * `stats.iter`: the current outer-iteration counter;
  * `stats.objective`: the current value of $f(x_k)$;
  * `stats.primal_feas`, `stats.dual_feas`: the current primal and dual
    feasibility measures;
  * `stats.status`: the current status of the algorithm. It should remain
    `:unknown` unless a stopping criterion has been attained; setting it to
    any other value (typically `:user`, see above) will stop the algorithm;
  * `stats.elapsed_time`: the elapsed (CPU) time, in seconds, since the
    start of the solve.
## The Inner-Loop (R2N) Callback
 
If you need finer-grained control, you can also pass a callback directly to
the R2N subsolver through the `sub_callback` keyword argument of
`L2Penalty`. It has exactly the same structure as the outer-loop callback
described above, but is called once per *R2N* (inner-loop) iteration, i.e.,
once per step computed while minimizing a single penalized subproblem.
 
```julia
function my_sub_callback(penalized_nlp, subsolver, substats)
  @info "R2N iter $(substats.iter): objective = $(substats.objective)"
end
 
stats = L2Penalty(nlp, sub_callback = my_sub_callback)
```
 
Inside `my_sub_callback`, the arguments correspond to:
 
* `penalized_nlp`: the current `L2PenalizedProblem` (or its shifted
  variant), representing $f(x) + \tau_k \|c(x)\|_2$ for the current value
  of $\tau_k$;
* `subsolver`: the `PenaltyR2NSolver` itself, giving access to
  `subsolver.xk` (the current inner iterate), `subsolver.y` (the current
  multiplier estimate), and `subsolver.subsolver` (the underlying
  `MoreSorensenSolver` used to compute each step);
* `substats`: the `GenericExecutionStats` for the R2N loop, with the same
  fields as `stats` above (`iter`, `objective`, `primal_feas`, `dual_feas`,
  `status`, `elapsed_time`), plus solver-specific entries such as
  `substats.solver_specific[:sigma]` (the current quadratic regularization
  parameter) and `substats.solver_specific[:rho]` (the ratio of actual to
  predicted decrease for the last step).
As with the outer-loop callback, setting `substats.status = :user` stops
the R2N loop early (and control then returns to the outer loop, which will
proceed to update the penalty parameter as usual).
 
## Example: Recording Convergence History
 
Combining both callbacks lets you build a detailed convergence history
across all levels of the algorithm:
 
```julia
using ExactPenalty
 
outer_history = Float64[]
inner_history = Float64[]
 
function outer_cb(nlp, solver, stats)
  push!(outer_history, stats.objective)
end
 
function inner_cb(penalized_nlp, subsolver, substats)
  push!(inner_history, substats.objective)
end
 
stats = L2Penalty(nlp, callback = outer_cb, sub_callback = inner_cb)
```