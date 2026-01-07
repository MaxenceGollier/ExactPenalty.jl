function prox_primal_feas!(solver::L2PenaltySolver{T}) where{T}
  norm_cx = solver.ψ.h(solver.ψ.b)  
  prox!(solver.s, solver.ψ, solver.s0, T(1))
  θ = norm_cx - solver.ψ(solver.s)

  sqrt_θ = θ ≥ 0 ? sqrt(θ) : sqrt(-θ)
  θ < 0 &&
    sqrt_θ ≥ neg_tol &&
    error("L2Penalty: prox-gradient step should produce a decrease but θ = $(θ)")
  return sqrt_θ
end

function kkt_primal_feas!(solver::L2PenaltySolver{T}) where{T}
  return norm(solver.ψ.b, Inf) 
end

function compute_multipliers!(solver::L2PenaltySolver{T}) where{T}
  s = isa(solver.subsolver, R2NSolver) ? solver.subsolver.s1 : solver.subsolver.s
  sigma_symbol = isa(solver.subsolver, R2NSolver) ? :sigma_cauchy : :sigma
  ψ = solver.subsolver.ψ

  lambda_temp = ψ.h.lambda

  # if τ = Inf, assuming full row rankness of A, then the solution of the prox is given by
  # y = (AAᵀ)^{-1}(-A∇f) -> corresponds to the least-square multipliers.
  ψ.h = NormL2(Inf)
  temp = copy(ψ.b) #FIXME
  ψ.b .= 0 
  @. solver.subsolver.mν∇fk = - solver.subsolver.∇fk
  prox!(s, ψ, solver.subsolver.mν∇fk, T(1))

  # Reset old value
  ψ.h = NormL2(lambda_temp)
  ψ.b .= temp

  set_solver_specific!(solver.substats, sigma_symbol, T(1))
  @. solver.y = - ψ.q
  solver.subsolver.subsolver.x1[length(s)+1:end] .= ψ.q
end

function prox_dual_feas!(solver::L2PenaltySolver{T}) where{T}
  σ = isa(solver.subsolver, R2NSolver) ? solver.substats.solver_specific[:sigma_cauchy] : solver.substats.solver_specific[:sigma]
  s = isa(solver.subsolver, R2NSolver) ? solver.subsolver.s1 : solver.subsolver.s

  norm_cx = solver.subsolver.ψ.h(solver.subsolver.ψ.b)
  shifted_norm_cx = solver.subsolver.ψ(s)

  ξ1 = norm_cx - shifted_norm_cx - dot(solver.subsolver.∇fk, s)
  sqrt_ξ1_σ = ξ1 ≥ 0 ? sqrt(ξ1 * σ) : sqrt(-ξ1 * σ)
  (ξ1 < 0 ) && error("L2Penalty: prox-gradient step should produce a decrease but ξ1 = $(ξ1)")
  return sqrt_ξ1_σ
end

function kkt_dual_feas!(solver::L2PenaltySolver{T}) where{T}
  σ = isa(solver.subsolver, R2NSolver) ? solver.substats.solver_specific[:sigma_cauchy] : solver.substats.solver_specific[:sigma]
  s = isa(solver.subsolver, R2NSolver) ? solver.subsolver.s1 : solver.subsolver.s
  return norm(s, Inf)*σ
end