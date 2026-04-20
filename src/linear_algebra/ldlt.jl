mutable struct PenaltyLDLTWorkspace{WP <: LDLFactorization, K2 <: AbstractMatrix, V <: AbstractVector, T <: Real}
  M::WP
  H::K2
  x::V
  dx::V
  r::V
  σ::T
  n::Int
  m::Int
  status::Symbol
end

function construct_ldlt_workspace(H::M, u1::V, n, m) where{T, V <: AbstractVector{T}, M <: Union{SparseMatrixCSC{T, Int}}}
  S = ldl_analyze(H)
  return PenaltyLDLTWorkspace(S, H, similar(u1), similar(u1), similar(u1), zero(T), n, m, :uninitialized)
end

function update_workspace!(solver_workspace::PenaltyLDLTWorkspace, B, A, σ, α)
  n, m = solver_workspace.n, solver_workspace.m
  H = solver_workspace.H

  @views H[1:n, 1:n] .= B'
  @inbounds for i in 1:n
    H[i,i] += σ
  end

  @views H[1:n, n+1:n+m] .= A'

  @inbounds for i = 1:m
    H[n+i, n+i] = -α
  end
  solver_workspace.σ = σ
  solver_workspace.M.__factorized = false
end

function set_dual_inertia!(solver_workspace::PenaltyLDLTWorkspace, α)
  n, m = solver_workspace.n, solver_workspace.m
  H = solver_workspace.H
  @inbounds for i = 1:m
    H[n+i, n+i] = -α
  end
  solver_workspace.M.__factorized = false
end

function set_primal_inertia!(solver_workspace::PenaltyLDLTWorkspace, σ)
  n, m = solver_workspace.n, solver_workspace.m
  H = solver_workspace.H
  σ_prev = solver_workspace.σ
  @inbounds for i in 1:n
    H[i,i] += σ - σ_prev
  end
  solver_workspace.σ = σ
  solver_workspace.M.__factorized = false
end

# Given Ax ≈ b, refine the solution by solving AΔx = b - Ax and updating x += Δx
function refine!(workspace::PenaltyLDLTWorkspace, u::V; max_iter::Int = 50, tol::T = eps(T)) where{T, V <: AbstractVector{T}}

  # Compute the residual r = u - H*x
  r, H, x = workspace.r, workspace.H, workspace.x
  dx = workspace.dx
  solved = false
  k = 0
  while k < max_iter && !solved
    r .= u
    mul!(r, Symmetric(H), workspace.x, -one(T), one(T)) # r = u - H*x
    ldiv!(dx, workspace.M, r) # H*dx = r
    x .+= dx
    k = k + 1
    solved = norm(dx) < tol*norm(x)
  end
  # Solve H*Δx = r

  ldiv!(dx, workspace.M, r)
  x .+= dx
  return workspace.x
end

function solve_system!(workspace::PenaltyLDLTWorkspace, u::V) where{V <: AbstractVector}
  factorized(workspace.M) || ldl_factorize!(workspace.H, workspace.M)
  if !factorized(workspace.M)
    workspace.status = :failed
    return
  end
  workspace.status = :success
  ldiv!(workspace.x, workspace.M, u)
  refine!(workspace, u)
end

function get_solution!(x::V, workspace::PenaltyLDLTWorkspace) where{V <: AbstractVector}
  x .= workspace.x
end

function get_status(workspace::PenaltyLDLTWorkspace)
  return workspace.status
end

function get_inertia(workspace::PenaltyLDLTWorkspace)
  LDL = workspace.M

  n = LDL.n
  (npos, nzero, nneg) = (0, 0, 0)

  D = LDL.d
  for i=1:n
    d = D[i]
    if real(d) > 0
      npos += 1
    elseif real(d) == 0
      nzero += 1
    else
      nneg += 1
    end
  end

  return npos, nzero, nneg
end