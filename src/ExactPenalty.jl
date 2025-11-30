module ExactPenalty

using LinearAlgebra
using NLPModels, NLPModelsModifiers, RegularizedProblems
using Krylov, LinearOperators, SolverCore

abstract type AbstractPenalizedProblemSolver <: AbstractOptimizationSolver end

include("subsolvers/tr_ms_linearop.jl")
include("subsolvers/tr_ms_sparse.jl")

include("algorithm.jl")

end