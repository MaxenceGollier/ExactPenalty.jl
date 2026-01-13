export TRTruncatedCGLinOpSolver, solve!

mutable struct TRTruncatedCGLinOpSolver{T <: Real, V <: AbstractVector{T}, L, W} <: AbstractPenalizedProblemSolver
  u1::V
  u2::V
  x1::V
  x2::V
  g::V
  H::L
  workspace::W
end

function TRTruncatedCGLinOpSolver(reg_nlp::AbstractRegularizedNLPModel{T, V}; ) where {T, V}
  x0 = reg_nlp.model.meta.x0
  n = reg_nlp.model.meta.nvar
  m = length(reg_nlp.h.b)
  u1 = similar(x0, n)
  u2 = zeros(eltype(x0), m)
  g = similar(x0, m)
  x1 = similar(u1)
  x2 = similar(u1)


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

  return TRTruncatedCGLinOpSolver(u1, u2, x1, x2, g, H, cg_workspace)
end

function SolverCore.solve!( #TODO add verbose and kwargs
  solver::TRTruncatedCGLinOpSolver{T, V},
  reg_nlp::AbstractRegularizedNLPModel{T, V},
  stats::GenericExecutionStats{T, V, V};
  x = reg_nlp.model.meta.x0,
  σk = T(1),
  atol = eps(T)^(0.6),
  max_time = T(30),
  max_iter = 100,
) where {T <: Real, V <: AbstractVector{T}}

  x1, x2 = solver.x1, solver.x2
  u1, u2 = solver.u1, solver.u2

  g = solver.g
  Δ = reg_nlp.h.h.lambda

  #Compute RHS AQ^{1}∇f - c
  g .= reg_nlp.h.b
  solve_shifted_system!(u1, reg_nlp.model.B, reg_nlp.model.∇f, reg_nlp.model.σ)
  mul!(g, reg_nlp.h.A, u1, one(T), -one(T))

  # Apply truncated CG
  cg!(solver.workspace, solver.H, solver.g, radius = Δ)

  # Compute solution of primal problem u = Q^{-1}(-∇f + A^T y)
  u1 .= reg_nlp.model.∇f
  mul!(u1, reg_nlp.h.A', solver.workspace.x, one(T), -one(T))
  solve_shifted_system!(stats.solution, reg_nlp.model.B, u1, reg_nlp.model.σ)

  if Δ*norm(reg_nlp.h.b) - obj(reg_nlp.model, stats.solution; skip_sigma = true) - reg_nlp.h(stats.solution)  < 0
    set_solution!(stats, x)
  end
end

function apply_H!(z::V, y::V, A::M1, Q::M2, σ::T, x1::V, x2::V) where{V, M1, M2, T}
  mul!(x1, A', y)
  solve_shifted_system!(x2, Q, x1, σ)
  mul!(z, A, x2)
end