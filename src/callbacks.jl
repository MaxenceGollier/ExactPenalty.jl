function subsolver_callback(nlp, solver::S, stats; feasibility_mode = :kkt) where{S <: Union{R2NSolver, R2Solver}}
  if feasibility_mode == :kkt 
    kkt_stopping_callback(nlp, solver, stats)
  elseif feasibility_mode == :prox
    prox_stopping_callback(nlp, solver, stats)
  end
end
function kkt_stopping_callback(nlp, solver::S, stats) where{S <: Union{R2NSolver, R2Solver}}
  σ = isa(solver, R2NSolver) ? stats.solver_specific[:sigma_cauchy] : stats.solver_specific[:sigma]
  s = isa(solver, R2NSolver) ? solver.s1 : solver.s

  if norm(s) < eps(eltype(s)) && stats.iter > 1
    stats.status = :small_step
  end

  ktol = stats.solver_specific[:ktol]

  set_dual_residual!(stats, norm(s, Inf)*σ)
  stats.dual_feas ≤ ktol && (stats.status = :user)
end

function prox_stopping_callback(nlp, solver::S, stats) where{S <: Union{R2NSolver, R2Solver}}
  σ = isa(solver, R2NSolver) ? stats.solver_specific[:sigma_cauchy] : stats.solver_specific[:sigma]
  s = isa(solver, R2NSolver) ? solver.s1 : solver.s

  ktol = stats.solver_specific[:ktol]

  ξ1 = solver.ψ.h.lambda*norm(solver.ψ.b) - solver.ψ(s) - dot(s, solver.∇fk)

  if ξ1 < 0
    stats.status = :not_desc
    return
  end
  if norm(s) < eps(eltype(s)) && stats.iter > 1
    stats.status = :small_step
  end

  set_dual_residual!(stats, sqrt(σ*ξ1))
  stats.dual_feas ≤ ktol && (stats.status = :user)
end