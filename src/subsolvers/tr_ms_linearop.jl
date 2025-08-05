export TRMoreSorensenLinOpSolver
import Base.show

mutable struct K2{T <:  Real, M1, M2} <: AbstractLinearOperator{T} #TODO move elsewhere etc.
  n::Int
  m::Int
  nrow::Int
  ncol::Int
  α::T
  σ::T
  A::M1
  B::M2
end

function LinearAlgebra.mul!(y::AbstractVector{T}, H::K2{T}, x::AbstractVector{T}, α::T, β::T) where T
    n, m = H.n, H.m
    @views mul!(y[1:n], H.B, x[1:n], α, β)
    @views @. y[1:n] += (α * H.σ) * x[1:n]
    @views mul!(y[1:n], H.A', x[n+1:end], α, one(T))
    @views mul!(y[n+1:end], H.A, x[1:n], α, β)
    @views @. y[n+1:end] -= (α * H.α) * x[n+1:end]

    return y
end

function Base.show(io::IO, op::K2)
  s = "K2 Linear operator\n"
  print(io, s)
end

mutable struct TRMoreSorensenLinOpSolver{T <: Real, V <: AbstractVector{T}, L <: AbstractLinearOperator, W <: KrylovWorkspace} <: AbstractPenalizedProblemSolver
  u1::V
  u2::V
  x1::V
  x2::V
  H::L
  workspace::W
end

function TRMoreSorensenLinOpSolver(reg_nlp::AbstractRegularizedNLPModel{T, V}; ) where {T, V}
  x0 = reg_nlp.model.meta.x0
  n = reg_nlp.model.meta.nvar
  m = length(reg_nlp.h.b)
  u1 = similar(x0, n+m)
  u2 = zeros(eltype(x0), n+m)
  x1 = similar(u1)
  x2 = similar(u1)


  H = K2(n, m, n+m, n+m, zero(T), reg_nlp.model.σ, reg_nlp.h.A, reg_nlp.model.B)
  krylov_workspace = MinresQlpWorkspace(H, u1)                   #TODO : allow to switch between Krylov solvers with krylov_workspace

  return TRMoreSorensenLinOpSolver(u1, u2, x1, x2, H, krylov_workspace)
end

function SolverCore.solve!( #TODO add verbose and kwargs
  solver::TRMoreSorensenLinOpSolver{T, V},
  reg_nlp::AbstractRegularizedNLPModel{T, V},
  stats::GenericExecutionStats{T, V, V};
  x = reg_nlp.model.meta.x0,
  σk = T(1),
  atol = eps(T)^(0.5),
  max_time = T(30),
  max_iter = 100,
) where {T <: Real, V <: AbstractVector{T}}
  start_time = time()
  set_time!(stats, 0.0)
  set_iter!(stats, 0)

  n = reg_nlp.model.meta.nvar
  m = length(reg_nlp.h.b)
  Δ = reg_nlp.h.h.lambda

  u1 = solver.u1
  u2 = solver.u2
  x1 = solver.x1
  x2 = solver.x2
  H = solver.H
  krylov_workspace = solver.workspace

  # Create problem
  @. u1[1:n] = -reg_nlp.model.∇f
  @. u1[(n + 1):(n + m)] = -reg_nlp.h.b

  H.α = zero(T)
  αmin = eps(T)^(0.5)
  θ = 0.8
  Q = 
  atol = eps(T)^0.3

  #FIXME : Do I need to update H.Q, H.A or are they automatically referenced ?
  H.σ = reg_nlp.model.σ
  
  minres_qlp!(krylov_workspace, H, u1, atol = eps(T)^0.8, rtol = eps(T)^0.8, Artol  = eps(T)^0.7)
  x1 .= krylov_workspace.x
  stats_krylov = krylov_workspace.stats

  if norm(@view x1[n+1:n+m]) <= Δ && stats_krylov.inconsistent && !(stats_krylov.status =="condition number seems too large for this machine")
		set_solution!(stats, x1[1:n])
    if reg_nlp.h.lambda*norm(reg_nlp.h.b) - obj(reg_nlp, x1[1:n]) < 0
      set_solution!(stats, x)
      isa(reg_nlp.model.B, AbstractQuasiNewtonOperator) && reset!(reg_nlp.model.B)
    end
	  return
  end

	if stats_krylov.inconsistent == true || stats_krylov.status =="condition number seems too large for this machine"
		H.α = αmin
		minres_qlp!(krylov_workspace, H, u1)
    x1 .= krylov_workspace.x
	end
 
  @views @. u2[(n + 1):(n + m)] = -x1[(n + 1):(n + m)]
  minres_qlp!(krylov_workspace, H, u2, atol = eps(T)^0.7, rtol = eps(T)^0.7)
  x2 .= krylov_workspace.x

  norm_x1 = norm(@view x1[(n + 1):(n + m)])
  while abs(norm_x1 - Δ) > atol && stats.iter < max_iter && stats.elapsed_time < max_time
    @views α₊ = H.α + norm_x1^2/dot(x1[(n + 1):(n + m)], x2[(n + 1):(n + m)])*(norm_x1/Δ - 1)

    H.α = α₊ ≤ 0 ? θ*H.α : α₊
    H.α = H.α ≤ αmin ? αmin : H.α

    minres_qlp!(krylov_workspace, H, u1, atol = eps(T)^0.7, rtol = eps(T)^0.7)
    x1 .= krylov_workspace.x
    norm_x1 = norm(x1)

    @views @. u2[(n + 1):(n + m)] = -x1[(n + 1):(n + m)]
    minres_qlp!(krylov_workspace, H, u1, atol = eps(T)^0.7, rtol = eps(T)^0.7)
    x2 .= krylov_workspace.x

    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time()-start_time)
    H.α == αmin && break
  end

  (stats.iter >= max_iter && isa(reg_nlp.model, AbstractQuasiNewtonOperator)) && reset!(reg_nlp.model.B)
  set_solution!(stats, @view x1[1:n])

  if Δ*norm(reg_nlp.h.b) - obj(reg_nlp, @view x1[1:n]) < 0 || any(isnan, x1)
    set_solution!(stats, x)
    isa(reg_nlp.model.B, AbstractQuasiNewtonOperator) && reset!(reg_nlp.model.B)
  end
end
