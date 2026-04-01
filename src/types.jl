# f(x) + \tau ||c(x)||
mutable struct L2PenalizedProblem{
  T,
  S,
  M<:AbstractNLPModel{T,S},
  H<:CompositeNormL2,
  V<:AbstractVector,
  I,
} <: AbstractPenalizedProblem{T,S}
  model::M
  h::H
  selected::I
  y::V
end

function L2PenalizedProblem(nlp::AbstractNLPModel{T}, h::CompositeNormL2) where {T}
  if isa(nlp, QuasiNewtonModel)
    return L2PenalizedProblem(nlp, h, 1:nlp.meta.nvar, zeros(T, 0))
  end
  return L2PenalizedProblem(nlp, h, 1:nlp.meta.nvar, zeros(T, nlp.meta.ncon))
end

function L2PenalizedProblem(
  nlp::AbstractNLPModel{T,V},
  h::CompositeNormL2,
  y::V,
) where {T,V}
  return L2PenalizedProblem(nlp, h, 1:nlp.meta.nvar, y)
end

function NLPModels.hess_op(nlp::L2PenalizedProblem, xk::AbstractVector)
  isa(nlp.model, QuasiNewtonModel) && return hess_op(nlp.model, xk)
  return hess_op(nlp.model, xk, nlp.y)
end

### 

mutable struct ShiftedPenalizedProblem{
  T, 
  V, 
  M <: AbstractNLPModel{T, V}, 
  H <: ShiftedProximableFunction, 
  I,
  P <: AbstractRegularizedNLPModel{T, V}
} <: AbstractShiftedProximableNLPModel{T, V}
  model::M
  h::H
  selected::I
  parent::P
  y::V
end

function RegularizedProblems.ShiftedProximableQuadraticNLPModel(
  reg_nlp::L2PenalizedProblem{T, V}, 
  x::V;
  ∇f::VN = nothing,
) where{T, V, VN <: Union{Nothing, AbstractVector{T}}}

  nlp, h, selected = reg_nlp.model, reg_nlp.h, reg_nlp.selected

  # φ(s) + ½ σ ‖s‖²
  isnothing(∇f) && (∇f = grad(nlp, x))
  B = hess_op(nlp, x, reg_nlp.y)
  φ = QuadraticModel(∇f, B, x0 = x, regularize = true)

  # ||c(x) + J(x) s||
  ψ = shifted(h, x)
      
  ShiftedPenalizedProblem(φ, ψ, selected, reg_nlp, reg_nlp.y)
end

function ShiftedProximalOperators.shift!(
  reg_nlp::ShiftedPenalizedProblem{T, V},
  x::V;
  compute_grad::Bool = true
) where{T, V}
  nlp, h = reg_nlp.parent.model, reg_nlp.parent.h
  φ, ψ = reg_nlp.model, reg_nlp.h

  ShiftedProximalOperators.shift!(ψ, x)

  g = φ.data.c
  compute_grad && grad!(nlp, x, g)

  φ.data.H = hess_op(nlp, x, reg_nlp.y)
end