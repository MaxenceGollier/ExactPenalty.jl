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
) where {T,V<:AbstractVector{T},M <: Symmetric}
  # Set params : TODO
  cntl = T == Float64 ? default_cntl64 : default_cntl32
  icntl = default_icntl

  ## Set Parameters

  # Deactivate Logging
  icntl[2], icntl[3], icntl[4] = 0, 0, 0

  # Deactivate permutation/scaling
  icntl[6], icntl[7] = 0, 0

  # Max number of iterative refinement steps
  icntl[10] = 10
  
  MPI.Init()
  S = Mumps{T}(mumps_symmetric, icntl, cntl)

  # Associate the row, cols and vals of the mumps structure with those of H.
  irn, jcn, a = H.data.rows, H.data.cols, H.data.vals
  S.irn, S.jcn, S.a = pointer.((irn, jcn, a))
  S.n = m+n
  S.nnz = length(irn)
  S._irn_gc_haven = irn
  S._jcn_gc_haven = jcn
  S._a_gc_haven = a

  # Associate the size and number of the right hand side
  x = similar(u1)
  S.lrhs = n + m
  S.nrhs = 1
  S.rhs = pointer(x)
  S._y_gc_haven = x

  return PenaltyMUMPSWorkspace(
    S,
    H,
    x,
    zero(T),
    n,
    m,
    :uninitialized,
    false,
  )
end

function construct_mumps_workspace(
  H::M,
  u1::V,
  n,
  m,
) where {T,V<:AbstractVector{T},M <: CompactBFGSK2}
  # Set params : TODO
  cntl = T == Float64 ? default_cntl64 : default_cntl32
  icntl = default_icntl

  ## Set Parameters

  # Deactivate Logging
  icntl[2], icntl[3], icntl[4] = 0, 0, 0

  # Deactivate permutation/scaling
  icntl[6], icntl[7] = 0, 0

  # Max number of iterative refinement steps
  icntl[10] = 10
  
  MPI.Init()
  S = Mumps{T}(mumps_symmetric, icntl, cntl)

  # Associate the row, cols and vals of the mumps structure with those of H.
  irn, jcn, a = H.H.data.rows, H.H.data.cols, H.H.data.vals
  S.irn, S.jcn, S.a = pointer.((irn, jcn, a))
  S.n = m+n
  S.nnz = length(irn)
  S._irn_gc_haven = irn
  S._jcn_gc_haven = jcn
  S._a_gc_haven = a

  # Associate the size and number of the right hand side
  x = similar(u1)
  S.lrhs = n + m
  S.nrhs = 1
  S.rhs = pointer(x)
  S._y_gc_haven = x

  return PenaltyMUMPSWorkspace(
    S,
    H,
    x,
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
  nnz_B, nnz_A = length(B.vals), length(A.vals)

  H = get_H(solver_workspace)

  H.vals[1:nnz_B] .= B.vals
  H.vals[(nnz_B + 1):(nnz_B + nnz_A)] .= A.vals
  H.vals[(nnz_B + nnz_A + 1):(nnz_B + nnz_A + n)] .= σ
  H.vals[(nnz_B + nnz_A + n + 1):(nnz_B + nnz_A + n + m)] .= -α

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
  nnz_A = length(A.vals)

  H = get_H(solver_workspace)

  H.vals[1:nnz_A] .= A.vals
  H.vals[(nnz_A + 1):(nnz_A + n)] .= σ + B.ξ
  H.vals[(nnz_A + n + 1):(nnz_A + n + m)] .= -α

  solver_workspace.σ = σ
  solver_workspace.factorized = false
end

function set_dual_inertia!(solver_workspace::PenaltyMUMPSWorkspace, α)
  n, m = solver_workspace.n, solver_workspace.m
  H = get_H(solver_workspace)
  H.vals[end-m+1:end] .= -α
  solver_workspace.factorized = false
end

function set_primal_inertia!(solver_workspace::PenaltyMUMPSWorkspace, σ)
  n, m = solver_workspace.n, solver_workspace.m
  H = get_H(solver_workspace)
  H.vals[end-m-n+1:end-m] .= σ
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
    job = mumps.job
    mumps.job = MUMPS.INITIALIZE
    factorize!(mumps)

    k, max_iter = 0, 5
    # MUMPS Documentation - infog(1) = -9
    # The main internal real/complex workarray S is too small. If INFO(2) is positive, then the number
    # of entries that are missing in S at the moment when the error is raised is available in INFO(2).
    # If INFO(2) is negative, then its absolute value should be multiplied by 1 million. If an error –9
    # occurs, the user should increase the value of ICNTL(14) before calling the factorization (JOB=
    # 2) again, except if LWK USER is provided LWK USER should be increased.
    while mumps.infog[1] == -9 && k < max_iter
      MUMPS.set_icntl!(mumps, 14, mumps.icntl[14] * 2)
      mumps.job = MUMPS.FACTOR
      factorize!(mumps)
      k = k + 1
    end
    workspace.factorized = true
  end

  # Check factorization status : TODO

  workspace.x .= u
  MUMPS.mumps_solve!(workspace.x, mumps; rhs_changed = true)
  #mumps.infog[15] > 0 && error("done") # It seems that mumps doesn't use the itref.. infog15 is the num of itref steps.
  # Check solve status : TODO
  return
end

function get_solution!(x::V, workspace::PenaltyMUMPSWorkspace) where {V<:AbstractVector}
  x .= workspace.x
end

function get_status(workspace::PenaltyMUMPSWorkspace)
  return workspace.status
end

function get_inertia(workspace::PenaltyMUMPSWorkspace{WP,K2}) where{WP,K2}

  n, m = workspace.n, workspace.m
  (npos, nzero, nneg) = (0, 0, 0)

  nneg = workspace.M.infog[12]
  rank = n + m - workspace.M.infog[28]
  nzero = n + m - rank
  npos = n + m - nzero - nneg
  
  return npos, nzero, nneg
end