export CompactBFGSModel

mutable struct CompactBFGS{T,V<:AbstractVector{T},MT<:AbstractMatrix{T}} <:
               AbstractMatrix{T}
  const scaling::Bool
  ξ::T
  Sk::MT # n x p
  Yk::MT # n x p
  Lk::MT # p x p
  _Dkinvsq::V  # p
  _DLk::MT # p x p
  Mk::MT # p x p
  Uk::MT # n x p
  Vk::MT # n x p
  _y::V  # p
  _mem::Int
  _insert::Int
  _nskip::Int
end

function CompactBFGS(
  T::Type,
  n::I;
  mem::I = 5,
  scaling::Bool = true,
  damped::Bool = false,
  σ₂::Float64 = 0.99,
  σ₃::Float64 = 10.0,
) where {I<:Integer}
  return CompactBFGS{T,Vector{T},Matrix{T}}(
    scaling,
    one(T),
    zeros(T, n, mem),
    zeros(T, n, mem),
    zeros(T, mem, mem),
    zeros(T, mem),
    zeros(T, mem, mem),
    zeros(T, mem, mem),
    zeros(T, n, mem),
    zeros(T, n, mem),
    zeros(T, mem),
    mem,
    1,
    0,
  )
end

# There is no efficient way to compute the number of nonzeros of this approximation.
# The meta of the compact BFGSModel will have 0 for the nnzh which is fine.
SparseArrays.nnz(::CompactBFGS) = 0
LinearAlgebra.Symmetric(op::CompactBFGS, ::Symbol) = op

# TODO: test this: compare with LinearOperators.jl
function LinearAlgebra.mul!(
  x::AbstractVector,
  op::CompactBFGS,
  y::AbstractVector,
  α::Real,
  β::Real,
)
  x .*= β                                               # x = βx
  x .+= (α*op.ξ) .* y                                   # x = βx + α * (ξI)y

  k = min(op._insert - 1, op._mem)

  @views mul!(op._y[1:k], op.Uk[:, 1:k]', y)            # _y = Uₖᵀy
  @views mul!(x, op.Uk[:, 1:k], op._y[1:k], -α, one(α)) #  x = βx + α * (ξI - UₖUₖᵀ)y
  @views mul!(op._y[1:k], op.Vk[:, 1:k]', y)            # _y = Vₖᵀy
  @views mul!(x, op.Vk[:, 1:k], op._y[1:k], α, one(α))  #  x = βx + α * (ξI - UₖUₖᵀ + VₖVₖᵀ)y
end

function NLPModels.reset!(op::CompactBFGS)
  op.ξ = 1
  op.Sk .= 0
  op.Yk .= 0
  op.Lk .= 0
  op.Mk .= 0
  op.Uk .= 0
  op.Vk .= 0
  op._Dkinvsq .= 0
  op._DLk .= 0
  op._y .= 0

  op._insert = 1
  op._nskip = 0

end
mutable struct CompactBFGSModel{
  T,
  S,
  M<:AbstractNLPModel{T,S},
  Meta<:AbstractNLPModelMeta{T,S},
  Op<:CompactBFGS{T},
} <: QuasiNewtonModel{T,S}
  meta::Meta
  model::M
  op::Op
end

function Base.push!(op::CompactBFGS{T,V,MT}, s::V, y::V) where {T,V,MT}
  k, mem = op._insert, op._mem
  Sk, Yk, Lk, Dkinvsq, _DLk, Mk, Uk, Vk =
    op.Sk, op.Yk, op.Lk, op._Dkinvsq, op._DLk, op.Mk, op.Uk, op.Vk
  ξ = op.ξ

  sy = dot(s, y)

  if sy <= eps(T) 
    op._nskip = op._nskip + 1
    if op._nskip > 2
      NLPModels.reset!(op)
    end
    return
  else
    op._nskip = 0
  end

  if op.scaling
    ξ = op.ξ = dot(y, y) / sy
  end

  # Shift
  if k > mem
    @inbounds for i = 1:(mem-1)
      copyto!(view(Sk, :, i), view(Sk, :, i+1))
      copyto!(view(Yk, :, i), view(Yk, :, i+1))
      Dkinvsq[i] = Dkinvsq[i+1]

      for j = 1:(i-1)
        Lk[i, j] = Lk[i+1, j+1]
      end
    end
    k = mem
  end

  copyto!(view(Sk, :, k), s)
  copyto!(view(Yk, :, k), y)

  @inbounds for j = 1:(k-1)
    Lk[k, j] = dot(s, view(Yk, :, j))
  end
  Dkinvsq[k] = 1 / sqrt(sy)

  # Compute compact representation Bₖ = σₖI - UₖUₖᵀ + VₖVₖᵀ.
  # Source: https://github.com/MadNLP/MadNLP.jl/blob/v0.9.1/src/quasi_newton.jl#L366

  # Step 1: compute Mₖ
  mul!(_DLk, Diagonal(Dkinvsq), Lk')
  mul!(Mk, _DLk', _DLk)                                    # Mₖ = Lₖ Dₖ⁻¹ Lₖᵀ
  mul!(Mk, Sk', Sk, ξ, one(T))                             # Mₖ = ξ Sₖᵀ Sₖ + Lₖ Dₖ⁻¹ Lₖᵀ

  # Step 2: factorize Mₖ
  cholesky!(Symmetric(@view Mk[1:k, 1:k]))                 # Mₖ = Jₖᵀ Jₖ (factorization)

  # Step 3: compute Vₖ
  mul!(Vk, Yk, Diagonal(Dkinvsq))

  # Step 4: compute Uₖ
  mul!(Uk, Vk, _DLk)                                       # Uₖ = Yₖ Dₖ⁻¹ Lₖᵀ
  axpy!(ξ, Sk, Uk)                                         # Uₖ = ξ Sₖ + Yₖ Dₖ⁻¹ Lₖᵀ
  @views rdiv!(Uk[:, 1:k], UpperTriangular(Mk[1:k, 1:k]))  # Uₖ = (ξ Sₖ + Yₖ Dₖ⁻¹ Lₖᵀ) Jₖ⁻¹

  op._insert = min(k + 1, mem + 1)
end

function CompactBFGSModel(nlp::AbstractNLPModel{T,S}; kwargs...) where {T,S}
  op = CompactBFGS(T, nlp.meta.nvar; kwargs...)
  return CompactBFGSModel(nlp.meta, nlp, op)
end

get_model(nlp::CompactBFGSModel) = nlp.model
get_op(nlp::CompactBFGSModel) = nlp.op
@default_counters CompactBFGSModel model
