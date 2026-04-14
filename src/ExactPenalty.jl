module ExactPenalty

using LinearAlgebra, SparseArrays
using NLPModels, NLPModelsModifiers, RegularizedProblems
using Krylov,
  LDLFactorizations,
  LinearOperators,
  ProximalOperators,
  QRMumps,
  QuadraticModels,
  ShiftedProximalOperators,
  SolverCore,
  SparseMatricesCOO

import SolverCore.reset!

abstract type AbstractPenalizedProblemSolver <: AbstractOptimizationSolver end

include("ExactPenaltyExecutionStats.jl")

include("linear_algebra/K2.jl")
include("linear_algebra/construct_workspace.jl")
include("linear_algebra/krylov.jl")
include("linear_algebra/ldlt.jl")

include("subsolvers/more-sorensen.jl")

include("types/PenalizedProblem.jl")
include("types/ShiftedPenalizedProblem.jl")

include("ir2n.jl")
include("ir2.jl")

include("algorithm.jl")

include("callbacks.jl")
include("feas_computer.jl")
end
