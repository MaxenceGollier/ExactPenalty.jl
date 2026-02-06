function subsolver_callback(nlp, solver::S, stats; feasibility_mode = :kkt) where{S <: Union{R2NSolver, R2Solver}}
  if feasibility_mode == :kkt 
    kkt_stopping_callback(nlp, solver, stats)
  elseif feasibility_mode == :decrease
    decr_stopping_callback(nlp, solver, stats)
  end
end

function kkt_stopping_callback(nlp, solver::S, stats) where{S <: Union{R2NSolver, R2Solver}}
  σ = isa(solver, R2NSolver) ? stats.solver_specific[:sigma_cauchy] : stats.solver_specific[:sigma]
  s = isa(solver, R2NSolver) ? solver.s1 : solver.s

  if norm(s) < eps(eltype(s)) && stats.iter > 1
    stats.status = :small_step
  end

  if stats.objective < -1e16
    stats.status = :unbounded
  end

  ξ1 = solver.ψ.h.lambda*norm(solver.ψ.b) - solver.ψ(s) - dot(s, solver.∇fk)
  
  if ξ1 < 0
    stats.status = :not_desc
  end

  if isa(solver, R2NSolver)
    if solver.substats.status == :not_desc
      stats.status = :not_desc
    end
  end

  ktol = stats.solver_specific[:ktol]

  set_dual_residual!(stats, norm(s, Inf)*σ)
  stats.multipliers .= solver.ψ.q .*( -σ )
  stats.dual_feas ≤ ktol && (stats.status = :user)
end

function decr_stopping_callback(nlp, solver::S, stats) where{S <: Union{R2NSolver, R2Solver}}
  σ = isa(solver, R2NSolver) ? stats.solver_specific[:sigma_cauchy] : stats.solver_specific[:sigma]
  s = isa(solver, R2NSolver) ? solver.s1 : solver.s

  ktol = stats.solver_specific[:ktol]

  ξ1 = solver.ψ.h.lambda*norm(solver.ψ.b) - solver.ψ(s) - dot(s, solver.∇fk)

  if ξ1 < 0
    stats.status = :not_desc
  end

  if norm(s) < eps(eltype(s)) && stats.iter > 1
    stats.status = :small_step
  end

  if stats.objective < -1e16
    stats.status = :unbounded
  end

  if isa(solver, R2NSolver)
    if solver.substats.status == :not_desc
      stats.status = :not_desc
    end
  end

  set_dual_residual!(stats, sqrt(σ*ξ1))
  stats.multipliers .= solver.ψ.q .*( -σ )
  stats.dual_feas ≤ ktol && (stats.status = :user)
end