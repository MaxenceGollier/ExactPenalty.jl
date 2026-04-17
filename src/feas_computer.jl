function compute_θ!(solver::L2PenaltySolver{T}) where {T}
  ## Computes a model decrease for the feasbility problem minₓ ‖c(x)‖₂
  ψ = solver.subsolver.subpb.h
  norm_cx = ψ.h(ψ.b)
  prox!(solver.s, ψ, solver.s0, 1/ψ.h.lambda)
  θ = (norm_cx - ψ(solver.s))/ψ.h.lambda
  return θ
end

function decr_primal_feas!(solver::L2PenaltySolver{T}) where {T}
  θ = compute_θ!(solver)
  θ < 0 && error("L2Penalty: prox-gradient step should produce a decrease but θ = $(θ)")

  sqrt_θ = θ ≥ 0 ? sqrt(θ) : sqrt(-θ)
  return sqrt_θ
end

function kkt_primal_feas!(solver::L2PenaltySolver{T}) where {T}
  return norm(solver.subsolver.subpb.h.b, Inf)
end

function compute_least_square_multipliers!(solver::L2PenaltySolver{T}) where {T}
  s = solver.subsolver.s
  sigma_symbol = isa(solver.subsolver, PenaltyR2NSolver) ? :sigma_cauchy : :sigma
  ψ = solver.subsolver.subpb.h

  lambda_temp = ψ.h.lambda

  # if τ = Inf, assuming full row rankness of A, then the solution of the prox is given by
  # y = (AAᵀ)^{-1}(-A∇f) -> corresponds to the least-square multipliers.
  ψ.h = NormL2(Inf)
  solver.temp_b .= ψ.b
  ψ.b .= 0
  solver.subsolver.subpb.model.data.c *= -1
  prox!(s, ψ, solver.subsolver.subpb.model.data.c, T(1))
  solver.subsolver.subpb.model.data.c *= -1

  # Reset old value
  ψ.h = NormL2(lambda_temp)
  ψ.b .= solver.temp_b

  set_solver_specific!(solver.substats, sigma_symbol, T(1))
  @. solver.y = - ψ.q
end

function update_constraint_multipliers!(solver::L2PenaltySolver{T}) where {T}
  σ =
    isa(solver.subsolver, PenaltyR2NSolver) ? solver.substats.solver_specific[:sigma_cauchy] :
    solver.substats.solver_specific[:sigma]
  @. solver.y = solver.subsolver.subpb.h.q * σ
end

function decr_dual_feas!(solver::L2PenaltySolver{T}) where {T}
  σ =
    isa(solver.subsolver, PenaltyR2NSolver) ? solver.substats.solver_specific[:sigma_cauchy] :
    solver.substats.solver_specific[:sigma]
  s = isa(solver.subsolver, PenaltyR2NSolver) ? solver.subsolver.s1 : solver.subsolver.s

  norm_cx = solver.subsolver.subpb.h.h(solver.subsolver.subpb.h.b)
  shifted_norm_cx = solver.subsolver.subpb.h(s)

  ξ1 = norm_cx - shifted_norm_cx - dot(solver.subsolver.∇fk, s)
  sqrt_ξ1_σ = ξ1 ≥ 0 ? sqrt(ξ1 * σ) : sqrt(-ξ1 * σ)
  (ξ1 < 0) &&
    error("L2Penalty: prox-gradient step should produce a decrease but ξ1 = $(ξ1)")
  return sqrt_ξ1_σ
end

function kkt_dual_feas!(solver::L2PenaltySolver{T}) where {T}
  σ = solver.subsolver.subpb.model.data.σ
  s = solver.subsolver.s

  solver.dual_res .= s .* σ
  mul!(solver.dual_res, Symmetric(solver.subsolver.subpb.model.data.H, :L), s, one(T), one(T))

  return norm(solver.dual_res, Inf)
end

function least_square_dual_feas!(solver::L2PenaltySolver{T}) where {T}
  dual_res, y = solver.dual_res, solver.y
  g, J = solver.subsolver.subpb.model.data.c, solver.subsolver.subpb.h.A #FIXME
  
  dual_res .= g
  mul!(dual_res, J', y, one(T), -one(T))
  return norm(dual_res, Inf)
end