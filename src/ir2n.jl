export PenaltyR2N, PenaltyR2NSolver, solve!

import SolverCore.solve!

mutable struct PenaltyR2NSolver{
  T <: Real,
  V <: AbstractVector{T},
  ST <: AbstractOptimizationSolver,
  PB <: AbstractRegularizedNLPModel,
} <: AbstractOptimizationSolver
  xk::V
  y::V
  xkn::V
  s::V
  m_fh_hist::V
  subsolver::ST
  subpb::PB
  substats::GenericExecutionStats{T, V, V, T}
end

function PenaltyR2NSolver(
  penalty_nlp::AbstractPenalizedProblem{T, V};
  subsolver = MoreSorensenSolver,
  m_monotone::Int = 1,
) where {T, V}
  x0 = penalty_nlp.model.meta.x0

  xk = similar(x0)
  ∇fk = similar(x0)
  y = similar(x0, get_ncon(penalty_nlp))
  xkn = similar(x0)
  s = similar(x0)

  m_fh_hist = fill(T(-Inf), m_monotone - 1)

  subpb = shifted(penalty_nlp, xk, ∇f = ∇fk)
  substats = GenericExecutionStats(subpb, solver_specific = Dict{Symbol, T}())
  subsolver = subsolver(subpb)

  return PenaltyR2NSolver{T, V, typeof(subsolver), typeof(subpb)}(
    xk,
    y,
    xkn,
    s,
    m_fh_hist,
    subsolver,
    subpb,
    substats,
  )
end

function SolverCore.reset!(solver::PenaltyR2NSolver)
  reset!(solver.subpb)
end

SolverCore.reset!(solver::PenaltyR2NSolver, model) = SolverCore.reset!(solver)

function SolverCore.solve!(
  solver::PenaltyR2NSolver{T, V},
  reg_nlp::AbstractRegularizedNLPModel{T, V},
  stats::GenericExecutionStats{T, V};
  callback = (args...) -> nothing,
  x::V = reg_nlp.model.meta.x0,
  atol::T = √eps(T),
  rtol::T = √eps(T),
  verbose::Int = 0,
  max_iter::Int = 1000,
  max_time::Float64 = 30.0,
  max_eval::Int = -1,
  σk::T = eps(T)^(1 / 5),
  σmin::T = eps(T),
  η1::T = √√eps(T),
  η2::T = T(0.1),
  γ::T = T(3),
  is_shifted::Bool = false
) where {T, V}
  reset!(stats)

  # Retrieve workspace
  nlp, h = reg_nlp.model, reg_nlp.h
  mk = solver.subpb
  φ, ψ = mk.model, mk.h

  xk = solver.xk .= x

  # Make sure ψ has the correct shift 
  !is_shifted && shift!(mk, xk, compute_grad = compute_grad)

  ∇fk = solver.subpb.model.data.c
  xkn = solver.xkn
  s, y = solver.s, solver.y
  m_fh_hist = solver.m_fh_hist .= T(-Inf)

  m_monotone = length(m_fh_hist) + 1

  if verbose > 0
    @info log_header(
      [:outer, :inner, :fx, :hx, :xi, :ρ, :σ, :normx, :norms, :arrow],
      [Int, Int, T, T, T, T, T, T, T, Char],
      hdr_override = Dict{Symbol, String}(
        :fx => "f(x)",
        :hx => "h(x)",
        :xi => "du_feas",
        :normx => "‖x‖",
        :norms => "‖s‖",
        :arrow => "PenaltyR2N",
      ),
      colsep = 1,
    )
  end

  local ξ1::T
  local ρk::T = zero(T)

  # initialize parameters
  hk = @views h(xk)
  fk = !is_shifted ? obj(nlp, xk) : stats.solver_specific[:smooth_obj]

  set_iter!(stats, 0)
  start_time = time()
  set_time!(stats, 0.0)
  set_objective!(stats, fk + hk)
  set_solver_specific!(stats, :smooth_obj, fk)
  set_solver_specific!(stats, :nonsmooth_obj, hk)
  set_solver_specific!(stats, :sigma, σk)
  m_monotone > 1 && (m_fh_hist[stats.iter % (m_monotone - 1) + 1] = fk + hk)

  solved = false

  set_status!(
    stats,
    get_status(
      reg_nlp,
      elapsed_time = stats.elapsed_time,
      iter = stats.iter,
      optimal = solved,
      max_eval = max_eval,
      max_time = max_time,
      max_iter = max_iter,
    ),
  )

  callback(reg_nlp, solver, stats)

  done = stats.status != :unknown

  while !done

    solver.subpb.model.data.σ = σk

    solve!(
      solver.subsolver,
      solver.subpb,
      solver.substats;
    )

    get_primal_dual_sol!(s, y, solver.subsolver)

    xkn .= xk .+ s
    fkn, hkn = obj(nlp, xkn), h(xkn)
    mks = dot(∇fk, s) + ψ(s)

    fhmax = m_monotone > 1 ? maximum(m_fh_hist) : fk + hk
    Δobj = fhmax - (fkn + hkn) + max(1, abs(fk + hk)) * 10 * eps()
    Δmod = fhmax - (fk + mks) + max(1, abs(fhmax)) * 10 * eps()

    ρk = Δobj / Δmod

    # Check stopping criteria
    σk = solver.subpb.model.data.σ 
    dual_res = Symmetric(solver.subpb.model.data.H, :L) * s + σk * s
    set_dual_residual!(stats, norm(dual_res, Inf))
    solved = stats.dual_feas ≤ atol
    stats.iter == 0 && (atol += stats.dual_feas * rtol)

    # Check boundedness
    unbounded = fk < - 1 / eps(T) 

    verbose > 0 &&
      stats.iter % verbose == 0 &&
      @info log_row(
        Any[
          stats.iter,
          solver.substats.iter,
          fk,
          hk,
          stats.dual_feas,
          ρk,
          σk,
          norm(xk),
          norm(s),
          (η2 ≤ ρk < Inf) ? '↘' : (ρk < η1 ? '↗' : '='),
        ],
        colsep = 1,
      )

    if η1 ≤ ρk < Inf
      xk .= xkn

      #update functions
      fk, hk = fkn, hkn

      shift!(mk, xk, y = y)

    end

    if η2 ≤ ρk < Inf
      σk = max(σk / γ, σmin)
    end

    if ρk < η1 || ρk == Inf
      σk = σk * γ
    end

    m_monotone > 1 && (m_fh_hist[stats.iter % (m_monotone - 1) + 1] = fk + hk)

    set_objective!(stats, fk + hk)
    set_solver_specific!(stats, :smooth_obj, fk)
    set_solver_specific!(stats, :nonsmooth_obj, hk)
    set_solver_specific!(stats, :sigma, σk)
    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time() - start_time)

    set_status!(
      stats,
      get_status(
        reg_nlp,
        elapsed_time = stats.elapsed_time,
        iter = stats.iter,
        optimal = solved,
        unbounded = unbounded,
        max_eval = max_eval,
        max_time = max_time,
        max_iter = max_iter,
      ),
    )

    callback(reg_nlp, solver, stats)

    done = stats.status != :unknown
  end

  if verbose > 0 && stats.status == :first_order
    @info log_row(
      Any[stats.iter, 0, fk, hk, stats.dual_feas, ρk, σk, norm(xk), norm(s), ""],
      colsep = 1,
    )
    @info "PenaltyR2N: terminating with √(ξ1/ν) = $(stats.dual_feas)"
  end

  set_solution!(stats, xk)
  set_residuals!(stats, zero(eltype(xk)), stats.dual_feas)
  return stats
end