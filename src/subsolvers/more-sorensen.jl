export MoreSorensenSolver
import Base.show

mutable struct MoreSorensenSolver{
  T<:Real,
  V<:AbstractVector{T},
  L<:Union{AbstractLinearOperator,AbstractMatrix},
  W,
} <: AbstractPenalizedProblemSolver
  u1::V
  u2::V
  x1::V
  x2::V
  H::L
  workspace::W
end

function MoreSorensenSolver(
  reg_nlp::AbstractRegularizedNLPModel{T,V};
  solver = :minres_qlp,
) where {T,V}
  x0 = reg_nlp.model.meta.x0
  n = reg_nlp.model.meta.nvar
  m = length(reg_nlp.h.b)
  u1 = similar(x0, n+m)
  u2 = zeros(eltype(x0), n+m)
  x1 = zeros(eltype(x0), n+m)
  x2 = zeros(eltype(x0), n+m)

  H = K2(n, m, n+m, n+m, zero(T), reg_nlp.model.data.σ, reg_nlp.h.A, reg_nlp.model.data.H)

  solver = isa(H, AbstractLinearOperator) ? :minres_qlp : :ldlt
  workspace = construct_workspace(H, u1, n, m; solver = solver)

  return MoreSorensenSolver(u1, u2, x1, x2, H, workspace)
end

function SolverCore.solve!( #TODO add verbose and kwargs
  solver::MoreSorensenSolver{T,V},
  reg_nlp::ShiftedL2PenalizedProblem{T, V, M, H, P},
  stats::GenericExecutionStats{T,V,V};
  x = reg_nlp.model.meta.x0,
  σk = T(1),
  atol = eps(T)^(0.6),
  max_time = T(30),
  max_iter = 10,
  σmax = 1 / eps(T)
) where {T, V, M, H, P}
  start_time = time()
  set_time!(stats, 0.0)
  set_iter!(stats, 0)

  n = reg_nlp.model.meta.nvar
  m = length(reg_nlp.h.b)
  Δ = reg_nlp.h.h.lambda

  u1, u2, x1, x2 = solver.u1, solver.u2, solver.x1, solver.x2
  solver_workspace = solver.workspace

  # Create problem
  @. u1[1:n] = -reg_nlp.model.data.c
  @. u1[(n+1):(n+m)] = -reg_nlp.h.b

  α = zero(T)
  update_workspace!(
    solver_workspace,
    reg_nlp.model.data.H,
    reg_nlp.h.A,
    reg_nlp.model.data.σ,
    α,
  )

  αmin = isa(reg_nlp.model.data.H, CompactBFGS) ? eps(T)^(0.6) : eps(T)^(0.8)
  θ = T(0.8)
  μ = T(10)

  # [ H + σI Aᵀ][x] = -[∇f]
  # [   A    0 ][y] = -[c] 
  solve_system!(solver_workspace, u1)
  get_solution!(x1, solver_workspace)
  npos, nzero, nneg = get_inertia(solver_workspace)
  status = get_status(solver_workspace)
  
  # Get correct inertia
  # If the factorization/solver failed, it in indicates we should add a minimal regularization too.
  if nneg < m || status == :failed
    α = αmin
    set_dual_inertia!(solver_workspace, αmin)
    solve_system!(solver_workspace, u1)
    get_solution!(x1, solver_workspace)
    npos, nzero, nneg = get_inertia(solver_workspace)
    status = get_status(solver_workspace)
  end

  while npos < n && reg_nlp.model.data.σ <= σmax

    reg_nlp.model.data.σ *= μ
    set_primal_inertia!(solver_workspace, reg_nlp.model.data.σ)

    # [ H + σI Aᵀ][x] = -[∇f]
    # [   A    0 ][y] = -[c] 
    solve_system!(solver_workspace, u1)
    get_solution!(x1, solver_workspace)
    npos, nzero, nneg = get_inertia(solver_workspace)
  end

  if reg_nlp.model.data.σ >= σmax 
    set_status!(stats, :exception) 
    return
  end

  if norm(@view x1[(n+1):(n+m)]) <= Δ
    set_solution!(stats, @view x1[1:n])
    set_status!(stats, :first_order)
  
    not_desc = !check_descent(reg_nlp, @view x1[1:n])
    not_desc && set_status!(stats, :not_desc)

    return
  end

  # [ H + σI Aᵀ][x'] = -[0]
  # [   A    0 ][y'] = -[x] 
  @views @. u2[(n+1):(n+m)] = -x1[(n+1):(n+m)]
  solve_system!(solver_workspace, u2)
  get_solution!(x2, solver_workspace)

  norm_x1 = norm(@view x1[(n+1):(n+m)])

  while abs(norm_x1 - Δ) > atol && stats.iter < max_iter && stats.elapsed_time < max_time
    # α = α + (‖y‖/Δ - 1)*‖y‖²/(yᵀy')
    @views α₊ = α + norm_x1^2/dot(x1[(n+1):(n+m)], x2[(n+1):(n+m)])*(norm_x1/Δ - 1)

    α = α₊ ≤ 0 ? max(θ*α, αmin) : α₊
    set_dual_inertia!(solver_workspace, α)

    # [ H + σI  Aᵀ ][x] = -[∇f]
    # [   A    -αI ][y] = -[c] 
    solve_system!(solver_workspace, u1)
    get_solution!(x1, solver_workspace)
    norm_x1 = norm(@view x1[(n+1):(n+m)])

    # Check whether the matrix still has the correct inertia. (We may have failed to detect earlier)
    npos, nzero, nneg = get_inertia(solver_workspace)
    if npos < n
      reg_nlp.model.data.σ *= μ
      if reg_nlp.model.data.σ >= σmax 
        set_status!(stats, :exception) 
        return
      end
      solve!(solver, reg_nlp, stats)
    end

    # [ H + σI  Aᵀ ][x'] = -[0]
    # [   A    -αI ][y'] = -[x]
    @views @. u2[(n+1):(n+m)] = -x1[(n+1):(n+m)]
    solve_system!(solver_workspace, u2)
    get_solution!(x2, solver_workspace)

    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time()-start_time)
    α == αmin && break
  end

  set_solution!(stats, @view x1[1:n])
  set_status!(stats, :first_order)

  stats.iter >= max_iter && set_status!(stats, :max_iter)
  stats.elapsed_time >= max_time && set_status!(stats, :max_time)
  !check_descent(reg_nlp, @view x1[1:n]) && set_status!(stats, :not_desc)
  if !check_descent(reg_nlp, @view x1[1:n])
    reg_nlp.model.data.σ *= μ
    if reg_nlp.model.data.σ >= σmax 
      set_status!(stats, :not_desc) 
      return
    end
    solve!(solver, reg_nlp, stats)
  end
end

function SolverCore.solve!(
  solver::MoreSorensenSolver{T,V},
  reg_nlp::ShiftedL2PenalizedProblem{T, V, M, H, P},
  stats::GenericExecutionStats{T,V,V};
  x = reg_nlp.model.meta.x0,
  σk = T(1),
  atol = eps(T)^(0.6),
  max_time = T(30),
  max_iter = 10,
  σmax = 1 / eps(T)
) where {T, V, M, H, O <: NullHessianModel, P <: L2PenalizedProblem{T, V, O}}

  n = reg_nlp.model.meta.nvar
  ψ = reg_nlp.h
  u1, x1 = solver.u1, solver.x1

  ν = 1 / reg_nlp.model.data.σ
  @. u1[1:n] = - ν * reg_nlp.model.data.c

  @views prox!(x1[1:n], ψ, u1[1:n], ν, max_iter = max_iter, max_time = max_time, atol = atol)
  
  @. x1[(n+1):end] = - ψ.q / ν
  set_solution!(stats, @view x1[1:n])
  set_status!(stats, :first_order)
  !check_descent(reg_nlp, @view x1[1:n]) && set_status!(stats, :not_desc)
end

function get_primal_dual_sol!(s, y, solver::MoreSorensenSolver)
  n = length(s)
  s .= @view solver.x1[1:n]
  y .= @view solver.x1[(n+1):end]
end
