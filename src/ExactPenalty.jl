module ExactPenalty

using LinearAlgebra
using NLPModels, NLPModelsModifiers, RegularizedProblems, RegularizedOptimization
using Krylov, LinearOperators, ProximalOperators, QRMumps, ShiftedProximalOperators, SolverCore, SparseMatricesCOO

import SolverCore.reset!

abstract type AbstractPenalizedProblemSolver <: AbstractOptimizationSolver end

include("ExactPenaltyExecutionStats.jl")

include("subsolvers/tr_ms_linearop.jl")
include("subsolvers/tr_ms_sparse.jl")

include("feas_computer.jl")

include("algorithm.jl")

end