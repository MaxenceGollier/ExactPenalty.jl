function second_order_correction!(
  solver::PenaltyR2NSolver{T,V},
  reg_nlp::AbstractRegularizedNLPModel{T,V},
  stats::GenericExecutionStats{T,V};
  callback = (args...) -> nothing,
  verbose::Int = 0,
  max_iter::Int = 10,
  max_time::Float64 = 30.0,
  max_eval::Int = -1,
  η1::T = √√eps(T),
  η2::T = T(0.1),
  γ::T = T(3),
  ρk::T = zero(T),
  min_ratio::T = T(1e-1),
) where {T,V}

  # Retrieve workspace
  nlp, h = reg_nlp.model, reg_nlp.h
  n = nlp.meta.nvar
  mk = solver.subpb
  φ, ψ = mk.model, mk.h
  xk, y, xkn, s = solver.xk, solver.y, solver.xkn, solver.s
  ∇fk = φ.data.c
  m_fh_hist = solver.m_fh_hist
  fk, hk = stats.solver_specific[:smooth_obj], stats.solver_specific[:nonsmooth_obj]

  m_monotone = length(m_fh_hist) + 1

  # Logging
  if verbose > 0 && verbose % stats.iter == 0
    @info log_row(
      Any[
        stats.iter,
        solver.substats.iter,
        T(0),
        T(0),
        T(0),
        ρk,
        T(0),
        norm(xk),
        norm(s),
        '=',
        "soc"
      ],
      colsep = 1,
    )
  end

  k = 0
  fkn, hkn = zero(T), zero(T)

  while ρk < η1 && k < max_iter

    second_order_correction!(
      solver.subsolver,
      solver.subpb,
      solver.substats
    )
    
    s_soc = @view solver.subsolver.workspace.x[1:n]
    
    xkn .+= s_soc
    s .+= s_soc

    fkn, hkn = obj(nlp, xkn), h(xkn)
    mks = dot(∇fk, s) + ψ(s)

    fhmax = m_monotone > 1 ? maximum(m_fh_hist) : fk + hk
    Δobj = fhmax - (fkn + hkn) + max(1, abs(fk + hk)) * 10 * eps()
    Δmod = fhmax - (fk + mks) + max(1, abs(fhmax)) * 10 * eps()

    ρkn = Δobj / Δmod
    if ρk / ρkn > min_ratio && ρkn < η1
      break
    else
      ρk = ρkn
    end
    k = k + 1
  end

  # Step acceptance
  if η1 ≤ ρk < Inf
    xk .= xkn

    #update functions
    fk, hk = fkn, hkn
    set_solver_specific!(stats, :smooth_obj, fk)
    set_solver_specific!(stats, :nonsmooth_obj, hk)

    shift!(mk, xk, y = y)
  end

  if η2 ≤ ρk < Inf
    stats.solver_specific[:sigma] = stats.solver_specific[:sigma] / γ
  end

  if ρk < η1 || ρk == Inf
    stats.solver_specific[:sigma] = stats.solver_specific[:sigma] * γ
  end
end