module ExactPenalty

using LinearAlgebra
using NLPModels, NLPModelsModifiers, RegularizedProblems, SparseMatricesCOO
using HSL, Krylov, LinearOperators, SolverCore

abstract type AbstractPenalizedProblemSolver <: AbstractOptimizationSolver end

include("algorithm.jl")
include("subsolvers/tr_ms_linearop.jl")
include("subsolvers/tr_ms_sparse.jl")

end