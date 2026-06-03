mutable struct PenaltyMUMPSWorkspace{
  WP<:Mumps,
  K2<:AbstractMatrix,
  V<:AbstractVector,
  T<:Real,
}
  M::WP
  H::K2
  x::V
  σ::T
  n::Int
  m::Int
  status::Symbol
  factorized::Bool
end

function get_H(
  solver_workspace::PenaltyMUMPSWorkspace{WP,K2},
) where {T,M,WP,K2<:Symmetric{T,M}}
  return solver_workspace.H.data
end

function get_H(solver_workspace::PenaltyMUMPSWorkspace{WP,K2}) where {WP,K2<:CompactBFGSK2}
  return solver_workspace.H.H.data
end

function construct_mumps_workspace(
  H::M,
  u1::V,
  n,
  m,
) where {T,V<:AbstractVector{T},M}
  # Set params : TODO
  cntl = T == Float64 ? default_cntl64 : default_cntl32
  MPI.Init()
  S = Mumps{T}(mumps_symmetric, default_icntl, cntl)
  return PenaltyMUMPSWorkspace(
    S,
    H,
    similar(u1),
    zero(T),
    n,
    m,
    :uninitialized,
    false,
  )
end

function update_workspace!(
  solver_workspace::PenaltyMUMPSWorkspace,
  B::M,
  A,
  σ,
  α,
) where {M<:SparseMatrixCOO}
  n, m = solver_workspace.n, solver_workspace.m
  H = get_H(solver_workspace)

  @inbounds for i = 1:n
    H[i, i] = σ
  end

  @inbounds for i = 1:length(B.vals)
    if B.cols[i] == B.rows[i]
      H[B.cols[i], B.rows[i]] += B.vals[i]
    else
      H[B.cols[i], B.rows[i]] = B.vals[i]
    end
  end

  @inbounds for i = 1:length(A.vals)
    H[A.cols[i], n+A.rows[i]] = A.vals[i]
  end

  @inbounds for i = 1:m
    H[n+i, n+i] = -α
  end

  solver_workspace.σ = σ
  solver_workspace.factorized = false
end

function update_workspace!(
  solver_workspace::PenaltyMUMPSWorkspace,
  B::M,
  A,
  σ,
  α,
) where {M<:CompactBFGS}
  n, m = solver_workspace.n, solver_workspace.m
  H = get_H(solver_workspace)

  @inbounds for i = 1:n
    H[i, i] = σ + B.ξ
  end

  @inbounds for i = 1:length(A.vals)
    H[A.cols[i], n+A.rows[i]] = A.vals[i]
  end

  @inbounds for i = 1:m
    H[n+i, n+i] = -α
  end
  solver_workspace.σ = σ
  solver_workspace.factorized = false
end

function set_dual_inertia!(solver_workspace::PenaltyMUMPSWorkspace, α)
  n, m = solver_workspace.n, solver_workspace.m
  H = get_H(solver_workspace)
  @inbounds for i = 1:m
    H[n+i, n+i] = -α
  end
  solver_workspace.factorized = false
end

function set_primal_inertia!(solver_workspace::PenaltyMUMPSWorkspace, σ)
  n, m = solver_workspace.n, solver_workspace.m
  H = get_H(solver_workspace)
  σ_prev = solver_workspace.σ
  @inbounds for i = 1:n
    H[i, i] += σ - σ_prev
  end
  solver_workspace.σ = σ
  solver_workspace.factorized = false
end

function solve_system!(
  workspace::PenaltyMUMPSWorkspace{WP,K2},
  u::V,
) where {V<:AbstractVector,WP,K2}
  workspace.status = :success
  mumps, H = workspace.M, workspace.H

  if !workspace.factorized
    associate_matrix!(mumps, H)
    factorize!(mumps)
  end

  # Check factorization status : TODO

  associate_rhs!(mumps, u)
  MUMPS.solve!(mumps)
  MUMPS.get_sol!(workspace.x, mumps)

  # Check solve status : TODO
  return
end

function get_solution!(x::V, workspace::PenaltyMUMPSWorkspace) where {V<:AbstractVector}
  x .= workspace.x
end

function get_status(workspace::PenaltyMUMPSWorkspace)
  return workspace.status
end

function get_inertia(workspace::PenaltyMUMPSWorkspace)
  m, n = workspace.m, workspace.n
  return n, 0, m # TODO
end