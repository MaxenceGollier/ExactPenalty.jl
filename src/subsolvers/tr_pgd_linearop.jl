export TRProjectedGDLinOpSolver, solve!

mutable struct TRProjectedGDLinOpSolver{T <: Real, V <: AbstractVector{T}, L, W} <: AbstractPenalizedProblemSolver
  u1::V
  u2::V
  x1::V
  x2::V
  y::V
  ∇q::V
  g::V
  H::L
  workspace::W
end

function TRProjectedGDLinOpSolver(reg_nlp::AbstractRegularizedNLPModel{T, V}; ) where {T, V}
  x0 = reg_nlp.model.meta.x0
  n = reg_nlp.model.meta.nvar
  m = length(reg_nlp.h.b)
  u1 = similar(x0, n)
  u2 = zeros(eltype(x0), m)
  g = similar(x0, m)
  x1 = similar(u1)
  x2 = similar(u1)
  ∇q = similar(g)
  y = zeros(eltype(g), m)

  # Create the linear operator
  function mul_H!(res, v, α, β)
    if β == 0
      apply_H!(res, v, reg_nlp.h.A, reg_nlp.model.B, reg_nlp.model.σ, x1, x2)
      res .*= α
    else
      apply_H!(u2, v, reg_nlp.h.A, reg_nlp.model.B, reg_nlp.model.σ, x1, x2)
      res .= α .* u2 .+ β .* res
    end
  end

  # H = AQ^{1}A^T
  H = LinearOperator(T, m, m, true, true, mul_H!)

  # g is the RHS
  g = zeros(T, m)
  cg_workspace = CgWorkspace(H, g)

  return TRProjectedGDLinOpSolver(u1, u2, x1, x2, y, ∇q, g, H, cg_workspace)
end

function SolverCore.solve!( #TODO add verbose and kwargs
  solver::TRProjectedGDLinOpSolver{T, V},
  reg_nlp::AbstractRegularizedNLPModel{T, V},
  stats::GenericExecutionStats{T, V, V};
  x = reg_nlp.model.meta.x0,
  σk = T(1),
  atol = eps(T)^(0.6),
  max_time = T(30),
  max_iter = 10000,
) where {T <: Real, V <: AbstractVector{T}}
  set_iter!(stats, 0)

  x1, x2 = solver.x1, solver.x2
  u1, u2 = solver.u1, solver.u2

  θ = svdvals(Matrix(reg_nlp.h.A))[1]^2/(minimum(svdvals(Matrix(reg_nlp.model.B+reg_nlp.model.σ*I(reg_nlp.model.meta.nvar)))))

  α = 1 / θ

  g = solver.g
  Δ = reg_nlp.h.h.lambda

  # Compute RHS AQ^{1}∇f - c
  g .= reg_nlp.h.b
  solve_shifted_system!(u1, reg_nlp.model.B, reg_nlp.model.∇f, reg_nlp.model.σ)
  mul!(g, reg_nlp.h.A, u1, one(T), -one(T))

  # Apply projected GD

  ∇q = solver.∇q
  y = solver.y
  while stats.iter <= max_iter
    #stats.iter % 100 == 0 && println("Trust region radius = $Δ, ||y|| = $(norm(y)),  ||∇q|| = $(norm(∇q))")
    #println("q(y) = $(dot(y, solver.H*y - g))")
  
    # Compute gradient
    ∇q .= g 
    mul!(∇q, solver.H, y, one(T), -one(T))

    @. y = y - α*∇q 

    # Project y
    norm_y = norm(y)
    if norm_y > Δ
      #println("Projection")
      @. y = (Δ/norm_y)*y
    end

    set_iter!(stats, stats.iter + 1)
  end

  # Compute solution of primal problem u = Q^{-1}(-∇f + A^T y)
  u1 .= reg_nlp.model.∇f
  mul!(u1, reg_nlp.h.A', y, one(T), -one(T))
  solve_shifted_system!(stats.solution, reg_nlp.model.B, u1, reg_nlp.model.σ)
end