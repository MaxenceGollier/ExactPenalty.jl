abstract type AbstractShiftedPenalizedProblem{T,S} <: AbstractPenalizedProblem{T,S} end

"""
    shifted_penalty_problem = ShiftedL2PenalizedProblem(nlp::L2PenalizedProblem, x::AbstractVector)

Given a `L2PenalizedProblem` model `nlp` representing the penalized problem

    minimize f(x) + τ‖c(x)‖₂,

construct the shifted L2 penalty problem around the point `x`:

    minimize f(x) + ∇f(x)ᵀ s + 1/2 sᵀ H(x) s + τ‖c(x) + J(x)s‖₂. 

The L2PenalizedProblem is made of the following components:
- `model`: the quadratic model of the original NLP model `nlp` around the point `x`;
- `h`: the shifted penalty term, which is a `ShiftedCompositeNormL2` object, see `ShiftedProximalOperators.jl`.
- `parent`: the original `L2PenalizedProblem` model `nlp`.
"""
mutable struct ShiftedL2PenalizedProblem{
  T,
  S,
  M<:AbstractQuadraticModel{T,S},
  H<:ShiftedCompositeNormL2,
  P<:L2PenalizedProblem,
  SN<:Union{Nothing, S},
} <: AbstractShiftedPenalizedProblem{T,S}
  model::M
  h::H
  parent::P
  meta::NLPModelMeta{T, S}
  _qn_∇f_prev::SN
  _qn_y::SN
  _qn_x_prev::SN
end

function ShiftedL2PenalizedProblem(
  penalty_nlp::L2PenalizedProblem{T, V, M}, 
  x::V;
  ∇f::VN1 = nothing,
) where{T, V, M <: QuasiNewtonModel{T, V}, VN1 <: Union{Nothing, V}}
 
  nlp, h = penalty_nlp.model, penalty_nlp.h

  ∇f = isnothing(∇f) ? grad(nlp, x) : ∇f
  B = hess_op(nlp, x)
  φ = QuadraticModel(∇f, B, x0 = x, regularize = true)

  ψ = shifted(h, x)

  return ShiftedL2PenalizedProblem(φ, ψ, penalty_nlp, penalty_nlp.meta, copy(∇f), similar(∇f), similar(∇f))
end

function ShiftedL2PenalizedProblem(
  penalty_nlp::L2PenalizedProblem{T, V, M}, 
  x::V;
  ∇f::VN1 = nothing,
  y::VN2 = nothing,
) where{T, V, M, VN1 <: Union{Nothing, V}, VN2 <: Union{Nothing, V}}
 
  nlp, h = penalty_nlp.model, penalty_nlp.h

  ∇f = isnothing(∇f) ? grad(nlp, x) : ∇f
  y = isnothing(y) ? zeros(T, nlp.meta.ncon) : y

  B = hess(nlp, x, y)
  φ = QuadraticModel(∇f, B, x0 = x, regularize = true)

  ψ = shifted(h, x)

  return ShiftedL2PenalizedProblem(φ, ψ, penalty_nlp, penalty_nlp.meta, nothing, nothing, nothing)
end

# ShiftedProximalOperators API
function ShiftedProximalOperators.shifted(
  penalty_nlp::L2PenalizedProblem{T, V, M}, 
  x::V;
  ∇f::VN1 = nothing,
) where{T, V, M <: QuasiNewtonModel{T, V}, VN1 <: Union{Nothing, V}}
  return ShiftedL2PenalizedProblem(penalty_nlp, x; ∇f = ∇f)
end

function ShiftedProximalOperators.shifted(
  penalty_nlp::L2PenalizedProblem{T, V, M}, 
  x::V;
  ∇f::VN1 = nothing,
  y::VN2 = nothing,
) where{T, V, M, VN1 <: Union{Nothing, V}, VN2 <: Union{Nothing, V}}
  return ShiftedL2PenalizedProblem(penalty_nlp, x; ∇f = ∇f, y = y)
end

function ShiftedProximalOperators.shift!(
  shifted_penalty_nlp::ShiftedL2PenalizedProblem{T, V, M},
  x::V;
  y::VN = nothing,
) where{T, V, H <: AbstractQuasiNewtonOperator{T}, M <: QuadraticModel{T, V, H}, VN <: Union{Nothing, V}}
  nlp, h = shifted_penalty_nlp.parent.model, shifted_penalty_nlp.parent.h
  φ, ψ = shifted_penalty_nlp.model, shifted_penalty_nlp.h
  qn_y, qn_g_prev, qn_x_prev = shifted_penalty_nlp._qn_y, shifted_penalty_nlp._qn_∇f_prev, shifted_penalty_nlp._qn_x_prev

  qn_s = qn_x_prev
  g, B = φ.data.c, φ.data.H

  grad!(nlp, x, g)

  # Update the approximation.
  @. qn_y = g - qn_g_prev
  if !isnothing(y)
    mul!(qn_y, ψ.A', y, -one(T), one(T)) # y = y + J(x)^T λ 
    mul!(qn_y, ψ.A_prev', y, one(T), one(T)) # y = y - J(x)_prev^T λ
  end

  qn_s .= x .- qn_x_prev

  push!(B, qn_y, qn_s)

  shift!(ψ, x)

  # Copy the gradient and Jacobian.
  qn_g_prev .= g
  ψ.A_prev.vals .= ψ.A.vals
  qn_x_prev .= x

end

function ShiftedProximalOperators.shift!(
  shifted_penalty_nlp::ShiftedL2PenalizedProblem{T, V, M},
  x::V;
  y::VN = nothing,
) where{T, V, M, VN <: Union{Nothing, V}}
  nlp, h = shifted_penalty_nlp.parent.model, shifted_penalty_nlp.parent.h
  φ, ψ = shifted_penalty_nlp.model, shifted_penalty_nlp.h

  g = φ.data.c
  grad!(nlp, x, g)

  if isnothing(y)
    hess_coord!(nlp, x, φ.data.H.nzval)
  else
    hess_coord!(nlp, x, y, φ.data.H.nzval)
  end

  shift!(ψ, x)
end

# Miscellaneous
function set_penalty!(nlp::ShiftedL2PenalizedProblem{T}, τ::T) where{T} 
  nlp.h.h = NormL2(τ)
  nlp.parent.h.h = NormL2(τ)
end

function check_descent(shifted_penalty_nlp::ShiftedL2PenalizedProblem{T}, s::AbstractVector) where{T}
  φ, ψ = shifted_penalty_nlp.model, shifted_penalty_nlp.h

  cx, τ = ψ.b, ψ.h.lambda
  ψ0 = τ * norm(cx) # φ0 = 0
  return ψ0 - obj(φ, s) - ψ(s) >= 0
end