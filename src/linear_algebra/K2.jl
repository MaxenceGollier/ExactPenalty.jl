mutable struct OpK2{T<:Real,M1,M2<:AbstractLinearOperator} <: AbstractLinearOperator{T}
  n::Int
  m::Int
  nrow::Int
  ncol::Int
  α::T
  σ::T
  A::M1
  B::M2
end

mutable struct CompactBFGSK2{T <: Real, V, M1 <: CompactBFGS, M2 <: AbstractMatrix{T}, M3 <: AbstractMatrix{T}} <: AbstractMatrix{T}
  B::M1
  H::M2
  Z1::M3
  Z2::M3
  x1::V
  x2::V
  x3::V
  y1::V
  y2::V
end

function CscK2(n::Int, m::Int, nrow::Int, ncol::Int, α::T, σ::T, A::M1, B::M2) where{T, M1 <:SparseMatrixCOO, M2<:SparseMatrixCSC}

  I, J, V = Vector{Int}(), Vector{Int}(), Vector{T}()

  # Step 1: Add the transpose of B to the K2 matrix.
  # Produces
  # [ Bᵀ 0 ]
  # [ 0  0 ]
  #
  # Note: LDLFactorizations requires an upper triangular view;
  # Meanwhile, NLPModels produces a lower triangular view...
  for j in 1:n
    for k in B.colptr[j]:(B.colptr[j+1]-1)
        i = B.rowval[k]
        push!(I, j)
        push!(J, i)
        push!(V, B.nzval[k])
    end
  end

  # Step 2: Add the transpose of A to the K2 matrix.
  # Produces 
  # [ Bᵀ Aᵀ ]
  # [ 0  0  ]
  #
  for k in 1:nnz(A)
    push!(I, A.cols[k])          # transpose: row becomes column
    push!(J, A.rows[k] + n)
    push!(V, A.vals[k])
  end

  # Step 3: Construct the sparse CSC matrix
  H = sparse(I, J, V, n+m, n+m)
  
  # Step 4: Initialize inertia corrections
  # Produces 
  # [ Bᵀ+σI Aᵀ ]
  # [ 0    -αI ]
  α_temp = iszero(α) ? eps(T) : α
  σ_temp = iszero(σ) ? eps(T) : σ

  @inbounds for i in 1:n
    H[i,i] += σ_temp
  end

  @inbounds for i in n+1:n+m
    H[i,i] -= α_temp
  end
  
  return Symmetric(H)
end

function CscK2(n::Int, m::Int, nrow::Int, ncol::Int, α::T, σ::T, A::M1, B::M2) where{T, M1 <:SparseMatrixCOO, M2<:CompactBFGS}

  I, J, V = Vector{Int}(), Vector{Int}(), Vector{T}()

  # Step 1: Add the transpose of A to the K2 matrix.
  # Produces 
  # [ 0  Aᵀ ]
  # [ 0  0  ]
  #
  for k in 1:nnz(A)
    push!(I, A.cols[k])          # transpose: row becomes column
    push!(J, A.rows[k] + n)
    push!(V, A.vals[k])
  end

  # Step 3: Construct the sparse CSC matrix
  H = sparse(I, J, V, n+m, n+m)
  
  # Step 4: Initialize inertia corrections
  # Produces 
  # [ ξI + σI  Aᵀ ]
  # [ 0       -αI ]
  α_temp = iszero(α) ? eps(T) : α
  σ_temp = iszero(σ) ? eps(T) : σ

  @inbounds for i in 1:n
    H[i,i] += B.ξ + σ_temp
  end

  @inbounds for i in n+1:n+m
    H[i,i] -= α_temp
  end

  return CompactBFGSK2(
    B, 
    Symmetric(H), 
    zeros(T, n + m, 2*B._mem),
    zeros(T, 2*B._mem, 2*B._mem),
    zeros(T, n+m), 
    zeros(T, n+m), 
    zeros(T, n+m), 
    zeros(T, 2*B._mem),
    zeros(T, 2*B._mem)
  )
end

function CooK2(n::Int, m::Int, nrow::Int, ncol::Int, α::T, σ::T, A::M1, B::M2) where{T, M1 <:SparseMatrixCOO, M2<:SparseMatrixCOO}

  I, J, V = Vector{Int}(), Vector{Int}(), Vector{T}()

  # Step 1: Add the transpose of B to the K2 matrix.
  # Produces
  # [ Bᵀ 0 ]
  # [ 0  0 ]
  #
  # Note: LDLFactorizations requires an upper triangular view;
  # Meanwhile, NLPModels produces a lower triangular view...
  append!(I, B.cols)
  append!(J, B.rows)
  append!(V, B.vals)

  # Step 2: Add the transpose of A to the K2 matrix.
  # Produces 
  # [ Bᵀ Aᵀ ]
  # [ 0  0  ]
  #
  for k in 1:nnz(A)
    push!(I, A.cols[k])          # transpose: row becomes column
    push!(J, A.rows[k] + n)
    push!(V, A.vals[k])
  end

  # Step 3: Construct the sparse CSC matrix
  H = sparse(I, J, V, n+m, n+m)
  
  # Step 4: Initialize inertia corrections
  # Produces 
  # [ Bᵀ+σI Aᵀ ]
  # [ 0    -αI ]
  α_temp = iszero(α) ? eps(T) : α
  σ_temp = iszero(σ) ? eps(T) : σ

  @inbounds for i in 1:n
    H[i,i] += σ_temp
  end

  @inbounds for i in n+1:n+m
    H[i,i] -= α_temp
  end
  
  return Symmetric(H)
end

function LinearAlgebra.mul!(
  y::AbstractVector{T},
  H::OpK2{T},
  x::AbstractVector{T},
  α::T,
  β::T,
) where {T}
  n, m = H.n, H.m
  @views mul!(y[1:n], H.B, x[1:n], α, β)
  @views @. y[1:n] += (α * H.σ) * x[1:n]
  @views mul!(y[1:n], H.A', x[(n+1):end], α, one(T))
  @views mul!(y[(n+1):end], H.A, x[1:n], α, β)
  @views @. y[(n+1):end] -= (α * H.α) * x[(n+1):end]

  return y
end

function Base.show(io::IO, op::OpK2)
  s = "K2 Linear operator\n"
  print(io, s)
end

function K2(n::Int, m::Int, nrow::Int, ncol::Int, α::T, σ::T, A::M1, B::M2) where {T,M1,M2}
  if M2 <: AbstractLinearOperator
    return OpK2(n, m, nrow, ncol, α, σ, A, B)
  elseif M2 <: SparseMatrixCOO
    return CooK2(n, m, nrow, ncol, α, σ, A, B)
  else
    return CscK2(n, m, nrow, ncol, α, σ, A, B)
  end
end
