module ExactPenalty

using LinearAlgebra, ProximalOperators

using QuadraticModels, ShiftedProximalOperators
using RegularizedProblems

using Krylov,
  LinearOperators,
  NLPModels,
  NLPModelsModifiers,
  QRMumps,
  RegularizedOptimization,
  SolverCore,
  SparseMatricesCOO

import SolverCore.reset!

abstract type AbstractPenalizedProblemSolver <: AbstractOptimizationSolver end
abstract type AbstractPenalizedProblem{T,S} <: AbstractRegularizedNLPModel{T,S} end

include("ExactPenaltyExecutionStats.jl")

include("linear_algebra/K2.jl")
include("linear_algebra/construct_workspace.jl")
include("linear_algebra/krylov.jl")

include("subsolvers/tr_ms_linearop.jl")
include("subsolvers/tr_ms_sparse.jl")

include("algorithm.jl")

include("getters.jl")
include("callbacks.jl")
include("feas_computer.jl")
include("types.jl")

end
