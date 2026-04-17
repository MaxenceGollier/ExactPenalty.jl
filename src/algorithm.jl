export L2Penalty, L2PenaltySolver, solve!

import SolverCore.solve!

mutable struct L2PenaltySolver{
  T<:Real,
  V<:AbstractVector{T},
  S<:AbstractOptimizationSolver,
  PB<:AbstractRegularizedNLPModel,
} <: AbstractOptimizationSolver
  x::V
  y::V
  dual_res::V
  s::V
  s0::V
  ∇fk::V
  temp_b::V
  subsolver::S
  subpb::PB
  substats::GenericExecutionStats{T,V,V,T}
end

function L2PenaltySolver(nlp::AbstractNLPModel{T,V}; subsolver = PenaltyR2NSolver) where {T,V}
  x0 = nlp.meta.x0
  x, s, s0 = similar(x0), similar(x0), similar(x0)
  temp_b = similar(x0, nlp.meta.ncon)
  dual_res = similar(x0)
  y = similar(x0, nlp.meta.ncon)
  ∇fk = similar(x0)

  penalty_subproblem = L2PenalizedProblem(nlp) # f(x) + τ‖c(x)‖₂
  substats = GenericExecutionStats(penalty_subproblem, solver_specific = Dict{Symbol, T}())
  solver = subsolver(penalty_subproblem)

  set_solver_specific!(substats, :primal_ktol, T(0))
  set_solver_specific!(substats, :dual_ktol, T(0))

  return L2PenaltySolver(x, y, dual_res, s, s0, ∇fk, temp_b, solver, penalty_subproblem, substats)
end

function SolverCore.reset!(solver::L2PenaltySolver)
  SolverCore.reset!(solver.subsolver)
end

"""
    L2Penalty(nlp; kwargs…)

An exact ℓ₂-penalty method for the problem

    min f(x) 	s.t c(x) = 0

where f: ℝⁿ → ℝ and c: ℝⁿ → ℝᵐ respectively have a Lipschitz-continuous gradient and Jacobian.

At each iteration k, an iterate is computed as 

    xₖ ∈ argmin f(x) + τₖ‖c(x)‖₂

where τₖ is some penalty parameter.
This nonsmooth problem is solved using `R2` (see `R2` for more information) with the first order model ψ(s;x) = τₖ‖c(x) + J(x)s‖₂

For advanced usage, first define a solver "L2PenaltySolver" to preallocate the memory used in the algorithm, and then call `solve!`:

    solver = L2PenaltySolver(nlp)
    solve!(solver, nlp)

    stats = ExactPenaltyExecutionStats(nlp)
    solver = L2PenaltySolver(nlp)
    solve!(solver, nlp, stats)

# Arguments
* `nlp::AbstractNLPModel{T, V}`: the problem to solve, see `RegularizedProblems.jl`, `NLPModels.jl`.

# Keyword arguments 
- `x::V = nlp.meta.x0`: the initial guess;
- `atol::T = √eps(T)`: absolute tolerance;
- `rtol::T = √eps(T)`: relative tolerance;
- `sub_atol::T = zero(T)`: absolute tolerance given to the subsolver;
- `sub_rtol::T = T(1e-2)`: relative tolerance given to the subsolver;
- `infeasible_tol = T(1e-2)`: tolerance used to decide whether the problem is infeasible or not √θₖ/‖c(xₖ)‖₂ < infeasible_tol, the problem is declared infeasible.
- `max_eval::Int = -1`: maximum number of evaluation of the objective function (negative number means unlimited);
- `sub_max_eval::Int = -1`: maximum number of evaluation for the subsolver (negative number means unlimited);
- `max_time::Float64 = 30.0`: maximum time limit in seconds;
- `max_iter::Int = 10000`: maximum number of iterations;
- `sub_max_iter::Int = 10000`: maximum number of iterations for the subsolver;
- `max_decreas_iter::Int = 10`: maximum number of iteration where ‖c(xₖ)‖₂ does not decrease before calling the problem locally infeasible;
- `verbose::Int = 0`: if > 0, display iteration details every `verbose` iteration;
- `sub_verbose::Int = 0`: if > 0, display subsolver iteration details every `verbose` iteration;
- `τ::T = T(100)`: initial penalty parameter;
- `β1::T = T(1)`: minimal penalty parameter increase,
- `β3::T = 1/τ`: initial regularization parameter σ₀ = β3/τₖ at each iteration;
- `β4::T = eps(T)`: minimal regularization parameter σ for `R2`;
- `primal_feasibility_mode::Symbol = :kkt`: describes how the primal feasibility is computed during the outer iterations. 
                                            With `:kkt`, the primal feasibility is the infinity norm of the residual ‖c(xₖ)‖∞.
                                            With `:decrease`, the primal feasibility is computed as a model decrease of the feasibility problem.
- `dual_feasibility_mode::Symbol = :kkt`: describes how the dual feasibility is computed during the outer and inner iterations. 
                                          With `:kkt`, the dual feasibility is the infinity norm of the residual ‖∇fₖ + Jₖᵀyₖ‖∞, where yₖ 
                                          is resulting from the computation of the Cauchy point of the subproblem.  
                                          With `:decrease`, the dual feasibility is computed as a model decrease with respect to the Cauchy point. 

other 'kwargs' are passed to `R2` (see `R2` for more information).

The algorithm stops either when `√θₖ < atol + rtol*√θ₀ ` or `θₖ < 0` and `√(-θₖ) < neg_tol` where θₖ := ‖c(xₖ)‖₂ - ‖c(xₖ) + J(xₖ)sₖ‖₂, and √θₖ is a stationarity measure.

# Output
The value returned is a `GenericExecutionStats`, see `SolverCore.jl`.

# Callback
The callback is called at each iteration.
The expected signature of the callback is `callback(nlp, solver, stats)`, and its output is ignored.
Changing any of the input arguments will affect the subsequent iterations.
In particular, setting `stats.status = :user` will stop the algorithm.
All relevant information should be available in `nlp` and `solver`.
Notably, you can access, and modify, the following:
- `solver.x`: current iterate;
- `solver.subsolver`: a `PenaltyR2Solver` structure holding relevant information on the subsolver state, see `R2` for more information;
- `stats`: structure holding the output of the algorithm (`GenericExecutionStats`), which contains, among other things:
  - `stats.iter`: current iteration counter;
  - `stats.objective`: current objective function value;
  - `stats.status`: current status of the algorithm. Should be `:unknown` unless the algorithm has attained a stopping criterion. Changing this to anything will stop the algorithm, but you should use `:user` to properly indicate the intention.
  - `stats.elapsed_time`: elapsed time in seconds.
You can also use the `sub_callback` keyword argument which has exactly the same structure and in sent to `R2`.
"""
function L2Penalty(
  nlp::AbstractNLPModel{T,V};
  subsolver = PenaltyR2Solver,
  kwargs...,
) where {T<:Real,V}
  if !equality_constrained(nlp)
    error("L2Penalty: This algorithm only works for equality contrained problems.")
  end
  solver = L2PenaltySolver(nlp, subsolver = subsolver)
  stats = ExactPenaltyExecutionStats(nlp)
  solve!(solver, nlp, stats; kwargs...)
  return stats
end

function SolverCore.solve!(
  solver::L2PenaltySolver{T,V},
  nlp::AbstractNLPModel{T,V},
  stats::GenericExecutionStats{T,V,V};
  callback = (args...) -> nothing,
  x::V = nlp.meta.x0,
  atol::T = √eps(T),
  rtol::T = √eps(T),
  sub_rtol = 1e-2,
  sub_atol = zero(T),
  infeasible_tol = T(1e-2),
  max_iter::Int = 100,
  sub_max_iter::Int = 100,
  max_time::T = T(30.0),
  max_eval::Int = -1,
  sub_max_eval::Int = -1,
  max_decreas_iter::Int = 10,
  verbose::Int = 0,
  sub_verbose::Int = 0,
  τ::T = T(100),
  β1::T = T(1),
  β3::T = 1e-4/τ,
  β4::T = eps(T),
  primal_feasibility_mode::Symbol = :kkt,
  dual_feasibility_mode::Symbol = :kkt,
) where {T,V}
  reset!(stats)

  @assert (primal_feasibility_mode == :decrease || primal_feasibility_mode == :kkt)
  @assert (dual_feasibility_mode == :decrease || dual_feasibility_mode == :kkt)

  # Retrieve workspace
  penalty_pb = solver.subpb # f(x) + τ‖c(x)‖₂
  mk = solver.subsolver.subpb
  φ, ψ = mk.model, mk.h

  x = solver.x .= x

  shift!(mk, x)
  fx = obj(nlp, x)
  hx = norm(ψ.b)

  if verbose > 0
    @info log_header(
      [:iter, :sub_iter, :fx, :pr_feas, :pr_feas_k, :du_feas, :du_feas_k, :tau, :normx],
      [Int, Int, Float64, Float64, Float64, Float64, Float64, Float64, Float64],
      hdr_override = Dict{Symbol,String}(   # TODO: Add this as constant dict elsewhere
        :iter => "outer",
        :sub_iter => "inner",
        :fx => "f(x)",
        :pr_feas => "pr_feas",
        :pr_feas_k => "pεₖ",
        :du_feas => "du_feas",
        :du_feas_k => "dεₖ",
        :tau => "τ",
        :normx => "‖x‖",
      ),
      colsep = 1,
    )
  end

  set_iter!(stats, 0)
  rem_eval = max_eval
  start_time = time()
  set_time!(stats, 0.0)
  set_objective!(stats, fx)

  ## Compute Feasibility

  primal_feas_computer! =
    primal_feasibility_mode == :decrease ? decr_primal_feas! : kkt_primal_feas!
  primal_feas = primal_feas_computer!(solver)

  set_solver_specific!(solver.substats, :smooth_obj, obj(nlp, x))
  fx = solver.substats.solver_specific[:smooth_obj]
  solver.∇fk .= φ.data.c
  compute_least_square_multipliers!(solver)

  τ = max(norm(solver.y, 1), T(1))
  set_penalty!(mk, τ)
  νsub = 1/max(β4, β3*τ)

  dual_feas_computer! =
    dual_feasibility_mode == :decrease ? decr_dual_feas! : kkt_dual_feas!
  dual_feas = least_square_dual_feas!(solver)

  primal_tol = atol + rtol * primal_feas
  dual_tol = atol + rtol * dual_feas

  primal_ktol = one(primal_tol)
  dual_ktol = min(one(dual_tol), max(sub_rtol * dual_feas + sub_atol, dual_tol))

  set_solver_specific!(solver.substats, :primal_ktol, primal_ktol)
  set_solver_specific!(solver.substats, :dual_ktol, dual_ktol)

  solved = dual_feas ≤ dual_tol && primal_feas ≤ primal_tol

  infeasible = false
  not_desc = false
  n_iter_since_decrease = 0

  set_status!(
    stats,
    get_status(
      nlp,
      elapsed_time = stats.elapsed_time,
      n_iter_since_decrease = n_iter_since_decrease,
      iter = stats.iter,
      optimal = solved,
      infeasible = infeasible,
      not_desc = not_desc,
      max_eval = max_eval,
      max_time = max_time,
      max_iter = max_iter,
      max_decreas_iter = max_decreas_iter,
    ),
  )

  callback(nlp, solver, stats)

  done = stats.status != :unknown

  while !done

    solve!(
      solver.subsolver,
      solver.subpb,
      solver.substats;
      x = x,
      atol = dual_ktol,
      rtol = T(0),
      verbose = sub_verbose,
      max_iter = sub_max_iter,
      max_time = max_time - stats.elapsed_time,
      max_eval = min(rem_eval, sub_max_eval),
      σmin = β4,
      σk = max(β4, β3*τ),
      is_shifted = true
    )

    if solver.substats.status == :unbounded
      τ *= 10
      set_penalty!(mk, τ)
      νsub = 1/max(β4, β3*τ)
      solver.subsolver.∇fk .= solver.∇fk
      set_solver_specific!(solver.substats, :smooth_obj, fx)
      continue
    end

    if solver.substats.status == :not_desc
      not_desc = true
    end

    x .= solver.substats.solution
    fx = solver.substats.solver_specific[:smooth_obj]
    hx_prev = copy(hx)
    hx = solver.substats.solver_specific[:nonsmooth_obj]/τ
    solver.∇fk .= φ.data.c
    update_constraint_multipliers!(solver)

    ## Compute feasibility 

    primal_feas = primal_feas_computer!(solver)
    dual_feas = dual_feas_computer!(solver)

    ## Log status
    verbose > 0 &&
      stats.iter % verbose == 0 &&
      @info log_row(
        Any[
          stats.iter,
          solver.substats.iter,
          fx,
          primal_feas,
          primal_ktol,
          dual_feas,
          dual_ktol,
          τ,
          norm(x),
        ],
        colsep = 1,
      )

    if primal_feas > primal_ktol || (dual_ktol ≤ dual_tol && primal_feas > primal_tol)
      # Update penalty parameter
      compute_least_square_multipliers!(solver)
      τ = max(τ + β1, norm(solver.y, 1))
      set_penalty!(mk, τ)

      # Initialize regularization parameter
      νsub = 1/max(β4, β3*τ)

      # Reset tolerance
      #dual_feas = dual_feas_computer!(solver)
      #dual_ktol = max(sub_rtol*dual_feas + sub_atol, dual_tol)
      #set_solver_specific!(solver.substats, :dual_ktol, dual_ktol)
    else
      # Tighten tolerances
      primal_ktol = max(sub_rtol*primal_feas + sub_atol, primal_tol)
      dual_ktol = max(sub_rtol*dual_feas + sub_atol, dual_tol)
      set_solver_specific!(solver.substats, :primal_ktol, primal_ktol)
      set_solver_specific!(solver.substats, :dual_ktol, dual_ktol)

      # Initialize regularization parameter
      νsub = 1/solver.substats.solver_specific[:sigma]
    end

    # Check whether the primal feasibility has decreased. If not, increase the penalty parameter more aggressively.
    if primal_feas > primal_ktol && hx_prev ≥ hx
      n_iter_since_decrease += 1
      β1 *= 10
    else
      n_iter_since_decrease = 0
    end

    solved = dual_feas ≤ dual_tol && primal_feas ≤ primal_tol

    θ = primal_feasibility_mode == :decrease ? primal_feas^2 : compute_θ!(solver)
    infeasible = sqrt(max(θ, 0))/hx < infeasible_tol && hx > primal_tol

    set_iter!(stats, stats.iter + 1)
    rem_eval = max_eval - neval_obj(nlp)
    set_time!(stats, time() - start_time)
    set_objective!(stats, fx)
    set_residuals!(stats, primal_feas, dual_feas)
    set_constraint_multipliers!(stats, solver.y)

    set_status!(
      stats,
      get_status(
        nlp,
        elapsed_time = stats.elapsed_time,
        n_iter_since_decrease = n_iter_since_decrease,
        iter = stats.iter,
        optimal = solved,
        infeasible = infeasible,
        not_desc = not_desc,
        max_eval = max_eval,
        max_time = max_time,
        max_iter = max_iter,
        max_decreas_iter = max_decreas_iter,
      ),
    )

    callback(nlp, solver, stats)

    done = stats.status != :unknown
  end

  set_solution!(stats, x)
  return stats
end

function get_status(
  nlp::M;
  elapsed_time = 0.0,
  iter = 0,
  optimal = false,
  infeasible = false,
  not_desc = false,
  n_iter_since_decrease = 0,
  max_eval = Inf,
  max_time = Inf,
  max_iter = Inf,
  max_decreas_iter = Inf,
) where {M<:AbstractNLPModel}
  if infeasible
    :infeasible
  elseif optimal
    :first_order
  elseif not_desc
    :not_desc
  elseif iter > max_iter
    :max_iter
  elseif elapsed_time > max_time
    :max_time
  elseif neval_obj(nlp) > max_eval && max_eval > -1
    :max_eval
  elseif n_iter_since_decrease ≥ max_decreas_iter
    :infeasible
  else
    :unknown
  end
end