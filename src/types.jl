mutable struct L2PenalizedProblem{T, S, M <: AbstractNLPModel{T, S}, V <: AbstractVector, I} <: AbstractPenalizedProblem{T, S}
  model::M
  h::CompositeNormL2
  selected::I
  y::V
end

function L2PenalizedProblem(nlp::AbstractNLPModel{T}, h::CompositeNormL2) where{T}
  if isa(nlp, QuasiNewtonModel)
    return L2PenalizedProblem(nlp, h, 1:nlp.meta.nvar, zeros(T, 0))
  end
  return L2PenalizedProblem(nlp, h, 1:nlp.meta.nvar, zeros(T, nlp.meta.ncon))
end

function L2PenalizedProblem(nlp::AbstractNLPModel{T, V}, h::CompositeNormL2, y::V) where{T, V}
  return L2PenalizedProblem(nlp, h, 1:nlp.meta.nvar, y)
end

function NLPModels.hess_op(nlp::L2PenalizedProblem, xk::AbstractVector)
  return hess_op(nlp.model, xk, nlp.y)
end