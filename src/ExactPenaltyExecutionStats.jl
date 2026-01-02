export ExactPenaltyExecutionStats

function ExactPenaltyExecutionStats(nlp::AbstractNLPModel{T, V}) where{T, V}
  stats = GenericExecutionStats(nlp, solver_specific = Dict{Symbol, T}())
  set_solver_specific!(stats, :theta, T(Inf))
  return stats
end