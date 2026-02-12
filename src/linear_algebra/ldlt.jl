struct PenaltyLDLTWorkspace{WP <: LDLFactorization, K2 <: AbstractMatrix, V <: AbstractVector}
  M::WP
  H::K2
  x::V
  n::Int
  m::Int
end

function construct_ldlt_workspace(H::M, u1::V, n, m) where{T, V <: AbstractVector{T}, M <: Symmetric{T, SparseMatrixCSC{T, Int}}}
  S = ldl_analyze(H)
  return PenaltyLDLTWorkspace(S, H, similar(u1), n, m)
end

function update_workspace!(solver_workspace::PenaltyLDLTWorkspace, B, A, σ, α)
  n, m = solver_workspace.n, solver_workspace.m
  H = solver_workspace.H.data

  @views H[1:n, 1:n] .= B.data
  @inbounds for i in 1:n
    H[i,i] += σ
  end

  @views H[1:n, n+1:n+m] .= A'

  @inbounds for i = 1:m
    H[n+i, n+i] = -α
  end
  solver_workspace.M.__factorized = false
end

function update_workspace!(solver_workspace::PenaltyLDLTWorkspace, α)
  n, m = solver_workspace.n, solver_workspace.m
  H = solver_workspace.H.data
  @inbounds for i = 1:m
    H[n+i, n+i] = -α
  end
  solver_workspace.M.__factorized = false
end

function solve_system!(workspace::PenaltyLDLTWorkspace, u::V) where{V <: AbstractVector}
  factorized(workspace.M) || ldl_factorize!(workspace.H, workspace.M)
  ldiv!(workspace.x, workspace.M, u)
end

function get_solution!(x::V, workspace::PenaltyLDLTWorkspace) where{V <: AbstractVector}
  x .= workspace.x
end

function get_status(workspace::PenaltyLDLTWorkspace)
  return :success #FIXME
end