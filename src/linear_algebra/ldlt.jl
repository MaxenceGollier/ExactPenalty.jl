mutable struct PenaltyLDLTWorkspace{
  WP<:LDLFactorization,
  K2<:AbstractMatrix,
  V<:AbstractVector,
  T<:Real,
}
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

function get_H(
  solver_workspace::PenaltyLDLTWorkspace{WP,K2},
) where {T,WP,K2<:Symmetric{T,SparseMatrixCSC{T,Int}}}
  return solver_workspace.H.data
end

function get_H(solver_workspace::PenaltyLDLTWorkspace{WP,K2}) where {WP,K2<:CompactBFGSK2}
  return solver_workspace.H.H.data
end

function construct_ldlt_workspace(
  H::M,
  u1::V,
  n,
  m,
) where {T,V<:AbstractVector{T},M<:Symmetric{T,SparseMatrixCSC{T,Int}}}
  S = ldl_analyze(H)
  return PenaltyLDLTWorkspace(
    S,
    H,
    similar(u1),
    similar(u1),
    similar(u1),
    zero(T),
    n,
    m,
    :uninitialized,
  )
end

function construct_ldlt_workspace(
  H::M,
  u1::V,
  n,
  m,
) where {T,V<:AbstractVector{T},M<:CompactBFGSK2}
  S = ldl_analyze(H.H)
  return PenaltyLDLTWorkspace(
    S,
    H,
    similar(u1),
    similar(u1),
    similar(u1),
    zero(T),
    n,
    m,
    :uninitialized,
  )
end

function update_workspace!(solver_workspace::PenaltyLDLTWorkspace, B::M, A, σ, α) where {M}
  n, m = solver_workspace.n, solver_workspace.m
  H = get_H(solver_workspace)

  @views H[1:n, 1:n] .= B'
  @inbounds for i = 1:n
    H[i, i] += σ
  end

  @views H[1:n, (n+1):(n+m)] .= A'

  @inbounds for i = 1:m
    H[n+i, n+i] = -α
  end
  solver_workspace.σ = σ
  solver_workspace.M.__factorized = false
end

function update_workspace!(
  solver_workspace::PenaltyLDLTWorkspace,
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

  @views H[1:n, (n+1):(n+m)] .= A'

  @inbounds for i = 1:m
    H[n+i, n+i] = -α
  end
  solver_workspace.σ = σ
  solver_workspace.M.__factorized = false
end

function set_dual_inertia!(solver_workspace::PenaltyLDLTWorkspace, α)
  n, m = solver_workspace.n, solver_workspace.m
  H = get_H(solver_workspace)
  @inbounds for i = 1:m
    H[n+i, n+i] = -α
  end
  solver_workspace.M.__factorized = false
end

function set_primal_inertia!(solver_workspace::PenaltyLDLTWorkspace, σ)
  n, m = solver_workspace.n, solver_workspace.m
  H = get_H(solver_workspace)
  σ_prev = solver_workspace.σ
  @inbounds for i = 1:n
    H[i, i] += σ - σ_prev
  end
  solver_workspace.σ = σ
  solver_workspace.M.__factorized = false
end

# Given Ax ≈ b, refine the solution by solving AΔx = b - Ax and updating x += Δx
function refine!(
  workspace::PenaltyLDLTWorkspace,
  u::V;
  max_iter::Int = 5,
  tol::T = eps(T),
) where {T,V<:AbstractVector{T}}

  # Compute the residual r = u - H*x
  r, H, x = workspace.r, workspace.H, workspace.x
  n, m = workspace.n, workspace.m
  dx = workspace.dx
  solved = false
  k = 0
  while k < max_iter && !solved
    r .= u

    # https://github.com/JuliaSparse/SparseArrays.jl/issues/685: mul!(r, H, x, -one(T), one(T)) is somehow extremely slow.
    mul!(r, H.data, x, -one(T), one(T)) # r = u - H*x
    mul!(r, H.data', x, -one(T), one(T))
    @inbounds for i = 1:(n+m)
      r[i] += H.data[i, i]*x[i]
    end

    ldiv!(dx, workspace.M, r) # H*dx = r
    x .+= dx
    k = k + 1
    solved = norm(dx) < tol*norm(x)
  end
  return workspace.x
end

function solve_system!(
  workspace::PenaltyLDLTWorkspace{WP,K2},
  u::V,
) where {V<:AbstractVector,WP,K2}
  workspace.status = :success

  factorized(workspace.M) || ldl_factorize!(workspace.H, workspace.M)
  if !factorized(workspace.M)
    workspace.status = :failed
    return
  end

  ldiv!(workspace.x, workspace.M, u)
  if any(isnan, workspace.x)
    workspace.status = :failed
    return
  end

  refine!(workspace, u)
  if any(isnan, workspace.x)
    workspace.status = :failed
    return
  end
end

function solve_system!(
  workspace::PenaltyLDLTWorkspace{WP,K2},
  u::V,
) where {V<:AbstractVector,WP,K2<:CompactBFGSK2}
  workspace.status = :success
  H = workspace.H
  B = workspace.H.B
  n, m = workspace.n, workspace.m
  p = B._mem
  x1, x2, x3, y1, y2 = H.x1, H.x2, H.x3, H.y1, H.y2
  Z1, Z2 = H.Z1, H.Z2

  # Step 0: Write (#TODO: we can use easily use QRMumps instead of LDLFactorization here...)
  # [B  Aᵀ] = [σI+ξI  Aᵀ] + [-U V]([U V])ᵀ
  # [A -αI] = [A     -αI] + [ 0 0]([0 0])
  # Hence,
  # [B  Aᵀ] = [σI+ξI  Aᵀ] + EFᵀ
  # [A -αI] = [A     -αI] + EFᵀ

  # Step 1: Factorize
  # [σI+ξI  Aᵀ]
  # [A     -αI]
  factorized(workspace.M) || ldl_factorize!(workspace.H.H, workspace.M)
  if !factorized(workspace.M)
    workspace.status = :failed
    return
  end

  # Step 2: Compute # TODO: allow for iterative refinement
  # [x₁] = [σI+ξI  Aᵀ]⁻¹[u]
  # [x₁] = [A     -αI]  [u]
  ldiv!(x1, workspace.M, u)
  if any(isnan, workspace.x)
    workspace.status = :failed
    return
  end

  # Step 3: Compute
  # y₁ = Fᵀx₁ = [Uᵀx₁(1:n)]
  # y₁ = Fᵀx₁ = [Vᵀx₁(1:n)]
  @views mul!(y1[1:p], B.Uk', x1[1:n])
  @views mul!(y1[(p+1):end], B.Vk', x1[1:n])


  # Step 4: Assemble Schur complement (I + Fᵀ [σI+ξI  Aᵀ]⁻¹ E )
  #                                   (       [A     -αI]     )
  # Step 4.1: Compute 
  # Z₁ = [σI+ξI  Aᵀ]⁻¹ E = [σI+ξI  Aᵀ]⁻¹[-U V]
  # Z₁ = [A     -αI]   E = [A     -αI]  [ 0 0]
  Z1 .= 0

  @views Z1[1:n, 1:p] .= B.Uk .* (-1)
  @views Z1[1:n, (p+1):end] .= B.Vk
  ldiv!(workspace.M, Z1)

  # Step 4.2: Compute 
  # Z₂ = FᵀZ₁ = UᵀZ₁[1:n]
  # Z₂ = FᵀZ₁ = VᵀZ₁[1:n]
  @views mul!(Z2[1:p, :], B.Uk', Z1[1:n, :])
  @views mul!(Z2[(p+1):end, :], B.Vk', Z1[1:n, :])

  # Step 4.3: Compute 
  # Z₂ = I + Z₂
  for i = 1:(2*p)
    Z2[i, i] += 1
  end

  # Step 5: Solve
  # (I + Fᵀ [σI+ξI  Aᵀ]⁻¹ E )⁻¹[y₁]
  # (       [A     -αI]     )  [y₁]
  # using Julia LinearALgebra's lu!
  F = lu!(Z2, check = false) # FIXME ?
  ldiv!(y2, F, y1)
  if any(isnan, y2)
    workspace.status = :failed
    return
  end

  # Step 6: Compute
  # x₂ = E[y₂] = [-U V][y₂] = [-Uy₂ + Vy₂]
  # x₂ = E[y₂] = [ 0 0][y₂] = [0]
  @views mul!(x2[1:n], B.Vk, y2[1:p])
  @views mul!(x2[1:n], B.Uk, y2[(p+1):end], -one(eltype(y2)), one(eltype(y2)))

  # Step 7: Solve
  # [x₃] = [σI+ξI  Aᵀ]⁻¹[x₂]
  # [x₃] = [A     -αI]  [x₂]
  ldiv!(x3, workspace.M, x2)

  # Step 8:
  # [B  Aᵀ]⁻¹[u] = x₁ - x₃ 
  # [A -αI]  [u] = x₁ - x₃
  workspace.x .= x1 .- x3
end

function get_solution!(x::V, workspace::PenaltyLDLTWorkspace) where {V<:AbstractVector}
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
  for i = 1:n
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
