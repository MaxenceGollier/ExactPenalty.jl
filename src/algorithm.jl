export L2Penalty, L2PenaltySolver, solve!

import SolverCore.solve!

mutable struct L2PenaltySolver{
  T <: Real,
  V <: AbstractVector{T},
  S <: AbstractOptimizationSolver,
  PB <: AbstractRegularizedNLPModel,
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
  substats::GenericExecutionStats{T, V, V, T}
end

function L2PenaltySolver(nlp::AbstractNLPModel{T, V}; subsolver = R2Solver) where {T, V}
  x0 = nlp.meta.x0
  x = similar(x0)
  s = similar(x0)
  y = similar(x0, nlp.meta.ncon)
  temp_b = similar(y)
  dual_res = similar(x0)
  s0 = zero(x0)
  ∇fk = similar(x0)

  # Allocating variables for the ShiftedProximalOperator structure
  (rows, cols) = jac_structure(nlp)
  vals = similar(rows, eltype(x0))
  A = SparseMatrixCOO(nlp.meta.ncon, nlp.meta.nvar, rows, cols, vals)
  b = similar(x0, eltype(x0), nlp.meta.ncon)

  # Allocate sub_h = ||c(x)|| to solve min f(x) + τ||c(x)||
  store_previous_jacobian = isa(nlp, QuasiNewtonModel) ? true : false
  sub_h =
    CompositeNormL2(one(T), (c, x) -> cons!(nlp, x, c), (j, x) -> jac_coord!(nlp, x, j.vals), A, b, store_previous_jacobian = store_previous_jacobian)
  subnlp = RegularizedNLPModel(nlp, sub_h)
  substats = RegularizedExecutionStats(subnlp)

  subpb = L2PenalizedProblem(nlp, sub_h, substats.multipliers)
  set_solver_specific!(substats, :ktol, T(0))

  if subsolver == R2NSolver
    if isa(nlp, QuasiNewtonModel)
      solver = subsolver(subpb, subsolver = TRMoreSorensenLinOpSolver)
    else
      solver = subsolver(subpb, subsolver = TRMoreSorensenLinOpSolver)
    end
  else
    solver = subsolver(subpb)
  end

  return L2PenaltySolver(x, y, dual_res, s, s0, ∇fk, temp_b, solver, subpb, substats)
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
- `solver.subsolver`: a `R2Solver` structure holding relevant information on the subsolver state, see `R2` for more information;
- `stats`: structure holding the output of the algorithm (`GenericExecutionStats`), which contains, among other things:
  - `stats.iter`: current iteration counter;
  - `stats.objective`: current objective function value;
  - `stats.status`: current status of the algorithm. Should be `:unknown` unless the algorithm has attained a stopping criterion. Changing this to anything will stop the algorithm, but you should use `:user` to properly indicate the intention.
  - `stats.elapsed_time`: elapsed time in seconds.
You can also use the `sub_callback` keyword argument which has exactly the same structure and in sent to `R2`.
"""
function L2Penalty(
  nlp::AbstractNLPModel{T, V};
  subsolver = R2Solver,
  kwargs...
) where {T <: Real, V}
  if !equality_constrained(nlp)
    error("L2Penalty: This algorithm only works for equality contrained problems.")
  end
  solver = L2PenaltySolver(nlp, subsolver = subsolver)
  stats = ExactPenaltyExecutionStats(nlp)
  solve!(solver, nlp, stats; kwargs...)
  return stats
end

function SolverCore.solve!(
  solver::L2PenaltySolver{T, V},
  nlp::AbstractNLPModel{T, V},
  stats::GenericExecutionStats{T, V, V};
  callback = (args...) -> nothing,
  x::V = nlp.meta.x0,
  atol::T = √eps(T),
  rtol::T = √eps(T),
  sub_rtol = 1e-2,
  sub_atol = zero(T),
  infeasible_tol = T(1e-2),
  max_iter::Int = 10000,
  sub_max_iter::Int = 10000,
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
) where {T, V}
  reset!(stats)
  reset!(solver)
  reset!(solver.substats)
  reset!(solver.subsolver)
  reset_data!(nlp)

  isa(solver.subsolver, R2NSolver) && (solver.subsolver.v0 .= (isodd.(eachindex(solver.subsolver.v0)) .* -2 .+ 1) ./ sqrt(length(solver.subsolver.v0))) # FIXME
  #This should be done in RegularizedOptimization, when calling reset!(::R2NSolver)

  @assert (primal_feasibility_mode == :decrease || primal_feasibility_mode == :kkt)
  @assert (dual_feasibility_mode == :decrease || dual_feasibility_mode == :kkt)

  # Retrieve workspace
  sub_h = solver.subpb.h
  ψ = solver.subsolver.ψ


  x = solver.x .= x
  shift!(ψ, x)
  fx = obj(nlp, x) #TODO: this call is redundant with the first evaluation of the objective function of R2N. We can remove this and rely on the lines in the while loop below.
  hx = norm(ψ.b)

  if verbose > 0
    @info log_header(
      [:iter, :sub_iter, :fx, :pr_feas, :du_feas, :epsk, :tau, :normx],
      [Int, Int, Float64, Float64, Float64, Float64, Float64, Float64, Float64],
      hdr_override = Dict{Symbol, String}(   # TODO: Add this as constant dict elsewhere
        :iter => "outer",
        :sub_iter => "inner",
        :fx => "f(x)",
        :pr_feas => "pr_feas",
        :du_feas => "du_feas",
        :epsk => "ϵₖ",
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

  primal_feas_computer! = primal_feasibility_mode == :decrease ? decr_primal_feas! : kkt_primal_feas!
  primal_feas = primal_feas_computer!(solver)

  set_solver_specific!(solver.substats, :smooth_obj, obj(nlp, x))
  fx = solver.substats.solver_specific[:smooth_obj]
  grad!(nlp, x, solver.subsolver.∇fk)
  solver.∇fk .= solver.subsolver.∇fk
  compute_least_square_multipliers!(solver)

  τ = max(norm(solver.y, 1), T(1))
  sub_h.h = NormL2(τ)
  ψ.h = NormL2(τ)
  νsub = 1/max(β4, β3*τ)

  dual_feas_computer! = dual_feasibility_mode == :decrease ? decr_dual_feas! : kkt_dual_feas!
  dual_feas = dual_feas_computer!(solver)

  feas = max(primal_feas, dual_feas) 

  atol += rtol * feas # make stopping test absolute and relative

  ktol = max(sub_rtol*dual_feas + sub_atol, atol) # Keep ϵ₀ ≥ ϵ
  set_solver_specific!(solver.substats, :ktol, ktol)
  
  solved = feas ≤ atol
  infeasible = false
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
        max_eval = max_eval,
        max_time = max_time,
        max_iter = max_iter,
        max_decreas_iter = max_decreas_iter,
      ),
    )
  
  callback(nlp, solver, stats)

  done = stats.status != :unknown

  while !done
    if isa(solver.subsolver, R2Solver)
      solve!(
        solver.subsolver,
        solver.subpb,
        solver.substats;
        callback = (args...) -> subsolver_callback(args...; feasibility_mode = dual_feasibility_mode),
        x = x,
        atol = ktol,
        rtol = T(0),
        neg_tol = T(0),
        verbose = sub_verbose,
        max_iter = sub_max_iter,
        max_time = max_time - stats.elapsed_time,
        max_eval = min(rem_eval, sub_max_eval),
        σmin = β4,
        ν = νsub,
        compute_obj = false,
        compute_grad = false
      )
    elseif isa(nlp, QuasiNewtonModel)
      solve!(
        solver.subsolver,
        solver.subpb,
        solver.substats;
        callback = (args...) -> subsolver_callback(args...; feasibility_mode = dual_feasibility_mode),
        qn_update_y! = _qn_lag_update_y!,
        qn_copy! = _qn_lag_copy!,
        x = x,
        atol = ktol,
        rtol = T(0),
        neg_tol = T(0),
        verbose = sub_verbose,
        max_iter = sub_max_iter,
        max_time = max_time - stats.elapsed_time,
        max_eval = min(rem_eval, sub_max_eval),
        σmin = β4,
        σk = 1/νsub,
        compute_obj = false,
        compute_grad = false
      )
    else 
      solve!(
        solver.subsolver,
        solver.subpb,
        solver.substats;
        callback = (args...) -> subsolver_callback(args...; feasibility_mode = dual_feasibility_mode),
        x = x,
        atol = ktol,
        rtol = T(0),
        neg_tol = T(0),
        verbose = sub_verbose,
        max_iter = sub_max_iter,
        max_time = max_time - stats.elapsed_time,
        max_eval = min(rem_eval, sub_max_eval),
        σmin = β4,
        σk = 1/νsub,
        compute_obj = false,
        compute_grad = false
      )
    end

    if solver.substats.status == :unbounded
      τ *= 10
      sub_h.h = NormL2(τ)
      ψ.h = NormL2(τ)
      νsub = 1/max(β4, β3*τ)
      solver.subsolver.∇fk .= solver.∇fk
      set_solver_specific!(solver.substats, :smooth_obj, fx)
      continue
    end

    x .= solver.substats.solution
    fx = solver.substats.solver_specific[:smooth_obj]
    hx_prev = copy(hx)
    hx = solver.substats.solver_specific[:nonsmooth_obj]/τ
    solver.∇fk .= solver.subsolver.∇fk
    update_constraint_multipliers!(solver)

    ## Compute feasibility 

    primal_feas = primal_feas_computer!(solver)
    dual_feas = dual_feas_computer!(solver)
    feas = max(primal_feas, dual_feas)

    ## Log status
    verbose > 0 &&
      stats.iter % verbose == 0 &&
      @info log_row(
        Any[stats.iter, solver.substats.iter, fx, primal_feas, dual_feas, ktol, τ, norm(x)],
        colsep = 1,
      )


    if primal_feas > ktol #FIXME
      compute_least_square_multipliers!(solver)
      τ₊ = max(τ + β1, norm(solver.y, 1))

      ## Extrapolation technique
      if isa(solver.subsolver, R2NSolver) 
        extrapolate!(x, solver, τ₊, τ)
        fx_new = obj(nlp, x)
        solver.temp_b .= cons(nlp, x) # FIXME
        if fx_new + τ₊*norm(solver.temp_b) < fx + τ₊*hx
          set_solver_specific!(solver.substats, :smooth_obj, fx_new)
          fx = fx_new
          grad!(nlp, x, solver.subsolver.∇fk)
        else
          x .= solver.substats.solution
        end
      end

      τ = τ₊
      sub_h.h = NormL2(τ)
      ψ.h = NormL2(τ)
      νsub = 1/max(β4, β3*τ)
    else
      n_iter_since_decrease = 0
      ktol = max(sub_rtol*dual_feas + sub_atol, atol)
      set_solver_specific!(solver.substats, :ktol, ktol)
      νsub = 1/solver.substats.solver_specific[:sigma]
    end
    if primal_feas > ktol && hx_prev ≥ hx
      n_iter_since_decrease += 1
      β1 *= 10
    else
      n_iter_since_decrease = 0
    end
      
    solved = feas ≤ atol

    θ = primal_feasibility_mode == :decrease ? primal_feas^2 : compute_θ!(solver)
    infeasible = sqrt(θ)/hx < infeasible_tol && hx > atol
    
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
  n_iter_since_decrease = 0,
  max_eval = Inf,
  max_time = Inf,
  max_iter = Inf,
  max_decreas_iter = Inf,
) where {M <: AbstractNLPModel}
  if infeasible
    :infeasible
  elseif optimal
    :first_order
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

function _qn_lag_update_y!(nlp::AbstractNLPModel{T, V}, solver::R2NSolver{T, G, V}, stats::GenericExecutionStats) where{T, V, G}
  @. solver.y = solver.∇fk - solver.∇fk⁻

  ψ = solver.ψ
  shifted_spmat = ψ.shifted_spmat
  spmat = shifted_spmat.spmat
  spfct = ψ.spfct
  qrm_update_shift_spmat!(shifted_spmat, zero(T))
  spmat.val[1:(spmat.mat.nz - spmat.mat.m)] .= ψ.A.vals
  qrm_spfct_init!(spfct, spmat)
  qrm_set(spfct, "qrm_keeph", 0) # Discard de Q matrix in all subsequent QR factorizations
  qrm_set(spfct, "qrm_rd_eps", eps(T)^(0.4)) # If a diagonal element of the R-factor is less than eps(R)^(0.4), we consider that A is rank defficient.

  mul!(ψ.g, ψ.A, solver.∇fk, one(T), zero(T))
  qrm_analyse!(spmat, spfct; transp = 't')
  qrm_factorize!(spmat, spfct, transp = 't')

  # Check full-rankness
  full_row_rank = (qrm_get(spfct, "qrm_rd_num") == 0)

  if full_row_rank
    qrm_solve!(spfct, ψ.g, ψ.p, transp = 't')
    qrm_solve!(spfct, ψ.p, ψ.q, transp = 'n')
    qrm_refine!(spmat, spfct, ψ.q, ψ.g, ψ.dq, ψ.p)
  else
    α = eps(T)^(0.9)
    qrm_golub_riley!(
      ψ.shifted_spmat,
      spfct,
      ψ.p,
      ψ.g,
      ψ.dp,
      ψ.q,
      ψ.dq,
      transp = 't',
      α = α,
      tol = eps(T)^(0.75),
    )
  end
  mul!(solver.y, solver.ψ.A', ψ.q, -one(T), one(T)) # y = y + J(x)^T λ 
  mul!(solver.y, solver.ψ.A_prev', ψ.q, one(T), one(T)) # y = y - J(x)_prev^T λ
end

function _qn_lag_copy!(nlp::AbstractNLPModel{T, V}, solver::R2NSolver{T, G, V}, stats::GenericExecutionStats) where{T, V,  G}
  solver.∇fk⁻ .= solver.∇fk
  solver.ψ.A_prev.vals .= solver.ψ.A.vals
end