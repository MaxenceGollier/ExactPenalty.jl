mutable struct OpK2{T <:  Real, M1, M2 <:AbstractLinearOperator} <: AbstractLinearOperator{T}
  n::Int
  m::Int
  nrow::Int
  ncol::Int
  α::T
  σ::T
  A::M1
  B::M2
end

mutable struct CooK2{T <: Real, I <: Integer} <: AbstractSparseMatrixCOO{T, I}
  m::Int
  n::Int
  nnz_B::Int
  nnz_A::Int
  rows::Vector{Ti}
  cols::Vector{Ti}
  vals::Vector{Tv}
end

function set_B!(M::CooK2, B::M2) where{M2 <: AbstractSparseMatrixCOO}
  nnz_B = M.nnz_B
  @views M.vals[1:nnz_B] .= B.vals
end

function set_A!(M::CooK2, A::M1) where{M1 <: AbstractSparseMatrixCOO}
  nnz_B = M.nnz_B
  @views M.vals[nnz_B+1:end] .= A.vals
end

function set_σ!(M::CooK2, σ::T) where{T}
  nnz_B = M.nnz_B
  nnz_A = M.nnz_A
  @views M.vals[nnz_B+nnz_A+1:nnz_B+nnz_A+M.n] .= σ
end

function set_α!(M::CooK2, α::T) where{T}
  nnz_B = M.nnz_B
  nnz_A = M.nnz_A
  @views M.vals[nnz_B+nnz_A+M.n+1:end] .= -α
end

function CooK2(n::Int, m::Int, nrow::Int, ncol::Int, α::T, σ::T, A::M1, B::M2) where{T, M1 <: AbstractSparseMatrixCOO, M2 <: AbstractSparseMatrixCOO}
  
  nnz_B = length(B.rows)
  nnz_A = length(A.rows)
  rows = zeros(Int, nnz_B + nnz_A + n + m)
  cols = zeros(Int, nnz_B + nnz_A + n + m)
  vals = zeros(T, nnz_B + nnz_A + n + m)
      

  @views rows[1:nnz_B] .= B.rows
  @views cols[1:nnz_B] .= B.cols
  @views vals[1:nnz_B] .= B.vals

  @views rows[nnz_B+1:nnz_B+nnz_A] .= A.rows .+ n
  @views cols[nnz_B+1:nnz_B+nnz_A] .= A.cols
  @views vals[nnz_B+1:nnz_B+nnz_A] .= A.vals

  @views rows[nnz_B+nnz_A+1:end] .= 1:(n+m)
  @views cols[nnz_B+nnz_A+1:end] .= 1:(n+m)
  @views vals[nnz_B+nnz_A+1:nnz_B+nnz_A+n] .= σ
  @views vals[nnz_B+nnz_A+n+1:end] .= -α

  return CooK2(n, m, nnz_B, nnz_A, rows, cols, vals) 
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
  elseif M2 <: AbstractSparseMatrixCOO && M1 <: AbstractSparseMatrixCOO
    return CooK2(n, m, nrow, ncol, α, σ, A, B) 
  else
    error("K2: currently only supports the case where A and B are either both sparse COO or B is a linear operator.")
  end
end