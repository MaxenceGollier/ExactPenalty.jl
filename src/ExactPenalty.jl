module ExactPenalty

using LinearAlgebra
using NLPModels, NLPModelsModifiers, RegularizedProblems, RegularizedOptimization
using Krylov, LinearOperators, ProximalOperators, QRMumps, ShiftedProximalOperators, SolverCore, SparseMatricesCOO

import SolverCore.reset!

abstract type AbstractPenalizedProblemSolver <: AbstractOptimizationSolver end
abstract type AbstractPenalizedProblem{T, S} <: AbstractRegularizedNLPModel{T, S} end

include("ExactPenaltyExecutionStats.jl")

include("linear_algebra/K2.jl")
include("linear_algebra/construct_workspace.jl")
include("linear_algebra/krylov.jl")

include("subsolvers/tr_ms_linearop.jl")
include("subsolvers/tr_ms_sparse.jl")

include("algorithm.jl")

include("callbacks.jl")
include("extrapolate.jl")
include("feas_computer.jl")
include("types.jl")

end