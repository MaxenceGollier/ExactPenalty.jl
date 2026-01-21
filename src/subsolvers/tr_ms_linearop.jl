export TRMoreSorensenLinOpSolver
import Base.show

mutable struct OpK2{T <:  Real, M1, M2 <:AbstractLinearOperator} <: AbstractLinearOperator{T} #TODO move elsewhere etc.
  n::Int
  m::Int
  nrow::Int
  ncol::Int
  α::T
  σ::T
  A::M1
  B::M2
end

function LinearAlgebra.mul!(y::AbstractVector{T}, H::OpK2{T}, x::AbstractVector{T}, α::T, β::T) where T
    n, m = H.n, H.m
    @views mul!(y[1:n], H.B, x[1:n], α, β)
    @views @. y[1:n] += (α * H.σ) * x[1:n]
    @views mul!(y[1:n], H.A', x[n+1:end], α, one(T))
    @views mul!(y[n+1:end], H.A, x[1:n], α, β)
    @views @. y[n+1:end] -= (α * H.α) * x[n+1:end]

    return y
end

function Base.show(io::IO, op::OpK2)
  s = "K2 Linear operator\n"
  print(io, s)
end

function K2(n::Int, m::Int, nrow::Int, ncol::Int, α::T, σ::T, A::M1, B::M2) where{T, M1, M2}
  if M2 <: AbstractLinearOperator
    return OpK2(n, m, nrow, ncol, α, σ, A, B)
  elseif M2 <: AbstractMatrix
    # For some reason, doing this in one line results in a SparseMatrixCSC instead of SparseMatrixCOO...
    H1 = [B+σ*I(n) coo_spzeros(T, n, m);]
    H2 = [A (-one(T))*I]
    return [H1; H2] 
  end
end

mutable struct TRMoreSorensenLinOpSolver{T <: Real, V <: AbstractVector{T}, L <: Union{AbstractLinearOperator, AbstractMatrix}, W} <: AbstractPenalizedProblemSolver
  u1::V
  u2::V
  x1::V
  x2::V
  H::L
  workspace::W
end

function TRMoreSorensenLinOpSolver(reg_nlp::AbstractRegularizedNLPModel{T, V}; solver = :minres_qlp) where {T, V}
  x0 = reg_nlp.model.meta.x0
  n = reg_nlp.model.meta.nvar
  m = length(reg_nlp.h.b)
  u1 = similar(x0, n+m)
  u2 = zeros(eltype(x0), n+m)
  x1 = zeros(eltype(x0), n+m)
  x2 = zeros(eltype(x0), n+m)

  isa(reg_nlp.model.B, AbstractMatrix) && (solver = :ma57) #FIXME

  H = K2(n, m, n+m, n+m, zero(T), reg_nlp.model.σ, reg_nlp.h.A, reg_nlp.model.B)
  workspace = construct_workspace(H, u1; solver = solver)

  return TRMoreSorensenLinOpSolver(u1, u2, x1, x2, H, workspace)
end

function SolverCore.solve!( #TODO add verbose and kwargs
  solver::TRMoreSorensenLinOpSolver{T, V},
  reg_nlp::AbstractRegularizedNLPModel{T, V},
  stats::GenericExecutionStats{T, V, V};
  x = reg_nlp.model.meta.x0,
  atol = eps(T)^(0.3),
  max_time = T(30),
  max_iter = 10,
) where {T <: Real, V <: AbstractVector{T}}
  start_time = time()
  set_time!(stats, 0.0)
  set_iter!(stats, 0)

  n = reg_nlp.model.meta.nvar
  m = length(reg_nlp.h.b)
  Δ = reg_nlp.h.h.lambda

  u1, u2, x1, x2 = solver.u1, solver.u2, solver.x1, solver.x2
  H = solver.H
  solver_workspace = solver.workspace

  # Create problem
  @. u1[1:n] = -reg_nlp.model.∇f
  @. u1[(n + 1):(n + m)] = -reg_nlp.h.b

  α = zero(T)
  update_workspace!(solver_workspace, H, reg_nlp.model.B, reg_nlp.h.A, reg_nlp.model.σ, α)

  αmin = eps(T)^(0.5)
  θ = 0.8
  system_tolerances = (atol = eps(T)^0.8, rtol = eps(T)^0.8)

  # [ H + σI Aᵀ][x] = -[∇f]
  # [   A    0 ][y] = -[c] 
  solve_system!(solver_workspace, H, u1; system_tolerances...)
  get_solution!(x1, solver_workspace)
  status = get_status(solver_workspace)

  if norm(@view x1[n+1:n+m]) <= Δ && status == :success
		set_solution!(stats, @view x1[1:n])
    if reg_nlp.h.h.lambda*norm(reg_nlp.h.b) - obj(reg_nlp, @view x1[1:n]) < 0
      # FIXME: just throw "not_desc" in this case, and let R2N do its thing...
      set_solution!(stats, x)
      isa(reg_nlp.model.B, AbstractQuasiNewtonOperator) && LinearOperators.reset!(reg_nlp.model.B)
    end
	  return
  end

	if status == :failed
    α = αmin
		update_workspace!(solver_workspace, H, m, αmin)
		solve_system!(solver_workspace, H, u1; system_tolerances...)
    get_solution!(x1, solver_workspace)
	end
 
  # [ H + σI Aᵀ][x'] = -[0]
  # [   A    0 ][y'] = -[x] 
  @views @. u2[(n + 1):(n + m)] = -x1[(n + 1):(n + m)]
  solve_system!(solver_workspace, H, u2; system_tolerances...)
  get_solution!(x2, solver_workspace)

  norm_x1 = norm(@view x1[(n + 1):(n + m)])

  while abs(norm_x1 - Δ) > atol && stats.iter < max_iter && stats.elapsed_time < max_time
    println(abs(norm_x1 - Δ))
    # α = α + (‖y‖/Δ - 1)*‖y‖²/(yᵀy')
    @views α₊ = α + norm_x1^2/dot(x1[(n + 1):(n + m)], x2[(n + 1):(n + m)])*(norm_x1/Δ - 1)
    
    α = α₊ ≤ 0 ? max(θ*α, αmin) : α₊
    update_workspace!(solver_workspace, H, m, α)

    # [ H + σI  Aᵀ ][x] = -[∇f]
    # [   A    -αI ][y] = -[c] 
    solve_system!(solver_workspace, H, u1; system_tolerances...)
    get_solution!(x1, solver_workspace)
    norm_x1 = norm(@view x1[(n + 1):(n + m)])

    # [ H + σI  Aᵀ ][x'] = -[0]
    # [   A    -αI ][y'] = -[x]
    @views @. u2[(n + 1):(n + m)] = -x1[(n + 1):(n + m)]
    solve_system!(solver_workspace, H, u2; system_tolerances...)
    get_solution!(x2, solver_workspace)

    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time()-start_time)
    α == αmin && break
  end

  (stats.iter >= max_iter && isa(reg_nlp.model, AbstractQuasiNewtonOperator)) && reset!(reg_nlp.model.B)
  # FIXME: just throw "max_iter" and let R2N do its thing...
  set_solution!(stats, @view x1[1:n])
  if Δ*norm(reg_nlp.h.b) - obj(reg_nlp, @view x1[1:n]) < 0 || any(isnan, x1) # FIXME: just throw "not_desc" in this case, and let R2N do its thing...
    set_solution!(stats, x)
    isa(reg_nlp.model.B, AbstractQuasiNewtonOperator) && LinearOperators.reset!(reg_nlp.model.B)
  end
end
