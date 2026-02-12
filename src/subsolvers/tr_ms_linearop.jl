export TRMoreSorensenLinOpSolver
import Base.show

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

  H = K2(n, m, n+m, n+m, zero(T), reg_nlp.model.σ, reg_nlp.h.A, reg_nlp.model.B)

  solver = isa(H, AbstractLinearOperator) ? :minres_qlp : :ldlt
  workspace = construct_workspace(H, u1, n, m; solver = solver)

  return TRMoreSorensenLinOpSolver(u1, u2, x1, x2, H, workspace)
end

function SolverCore.solve!( #TODO add verbose and kwargs
  solver::TRMoreSorensenLinOpSolver{T, V},
  reg_nlp::AbstractRegularizedNLPModel{T, V},
  stats::GenericExecutionStats{T, V, V};
  x = reg_nlp.model.meta.x0,
  σk = T(1),
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
  solver_workspace = solver.workspace

  # Create problem
  @. u1[1:n] = -reg_nlp.model.∇f
  @. u1[(n + 1):(n + m)] = -reg_nlp.h.b

  α = zero(T)
  update_workspace!(solver_workspace, reg_nlp.model.B, reg_nlp.h.A, reg_nlp.model.σ, α)

  αmin = eps(T)^(0.5)
  θ = 0.8

  # [ H + σI Aᵀ][x] = -[∇f]
  # [   A    0 ][y] = -[c] 
  solve_system!(solver_workspace, u1)
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
		update_workspace!(solver_workspace, αmin)
		solve_system!(solver_workspace, u1)
    get_solution!(x1, solver_workspace)
	end
 
  # [ H + σI Aᵀ][x'] = -[0]
  # [   A    0 ][y'] = -[x] 
  @views @. u2[(n + 1):(n + m)] = -x1[(n + 1):(n + m)]
  solve_system!(solver_workspace, u2)
  get_solution!(x2, solver_workspace)

  norm_x1 = norm(@view x1[(n + 1):(n + m)])

  while abs(norm_x1 - Δ) > atol && stats.iter < max_iter && stats.elapsed_time < max_time
    # α = α + (‖y‖/Δ - 1)*‖y‖²/(yᵀy')
    @views α₊ = α + norm_x1^2/dot(x1[(n + 1):(n + m)], x2[(n + 1):(n + m)])*(norm_x1/Δ - 1)
    
    α = α₊ ≤ 0 ? max(θ*α, αmin) : α₊
    update_workspace!(solver_workspace, α)

    # [ H + σI  Aᵀ ][x] = -[∇f]
    # [   A    -αI ][y] = -[c] 
    solve_system!(solver_workspace, u1)
    get_solution!(x1, solver_workspace)
    norm_x1 = norm(@view x1[(n + 1):(n + m)])

    # [ H + σI  Aᵀ ][x'] = -[0]
    # [   A    -αI ][y'] = -[x]
    @views @. u2[(n + 1):(n + m)] = -x1[(n + 1):(n + m)]
    solve_system!(solver_workspace, u2)
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
