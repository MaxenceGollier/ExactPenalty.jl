function compute_θ!(solver::L2PenaltySolver{T}) where {T}
  ## Computes a model decrease for the feasbility problem minₓ ‖c(x)‖₂
  ψ = get_ψ(solver)
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
  return norm(get_ψ(solver).b, Inf)
end

function compute_least_square_multipliers!(solver::L2PenaltySolver{T}) where {T}
  s = get_cauchy_step(solver)
  sigma_symbol = isa(solver.subsolver, R2NSolver) ? :sigma_cauchy : :sigma
  ψ = get_ψ(solver)

  lambda_temp = ψ.h.lambda

  # if τ = Inf, assuming full row rankness of A, then the solution of the prox is given by
  # y = (AAᵀ)^{-1}(-A∇f) -> corresponds to the least-square multipliers.
  ψ.h = NormL2(Inf)
  solver.temp_b .= ψ.b
  ψ.b .= 0
  @. solver.subsolver.mν∇fk = - solver.subsolver.∇fk
  prox!(s, ψ, solver.subsolver.mν∇fk, T(1))

  # Reset old value
  ψ.h = NormL2(lambda_temp)
  ψ.b .= solver.temp_b

  set_solver_specific!(solver.substats, sigma_symbol, T(1))
  @. solver.y = ψ.q
end

function update_constraint_multipliers!(solver::L2PenaltySolver{T}) where {T}
  σ = get_cauchy_sigma(solver)
  ψ = get_ψ(solver)
  @. solver.y = ψ.q * σ
end

function decr_dual_feas!(solver::L2PenaltySolver{T}) where {T}
  σ = get_cauchy_sigma(solver)
  s = get_cauchy_step(solver)
  ψ = get_ψ(solver)

  norm_cx = ψ.h(ψ.b)
  shifted_norm_cx = ψ(s)

  ξ1 = norm_cx - shifted_norm_cx - dot(solver.subsolver.∇fk, s)
  sqrt_ξ1_σ = ξ1 ≥ 0 ? sqrt(ξ1 * σ) : sqrt(-ξ1 * σ)
  (ξ1 < 0) &&
    error("L2Penalty: prox-gradient step should produce a decrease but ξ1 = $(ξ1)")
  return sqrt_ξ1_σ
end

function kkt_dual_feas!(solver::L2PenaltySolver{T}) where {T}
  σ = get_cauchy_sigma(solver)
  s = get_cauchy_step(solver)
  return norm(s, Inf)*σ
end
