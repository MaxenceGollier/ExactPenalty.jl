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