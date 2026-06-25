using ExactPenalty
using CUTEst,
  LinearOperators,
  NLPModels,
  NLPModelsModifiers,
  QuadraticModels,
  RegularizedProblems,
  ShiftedProximalOperators,
  SolverCore,
  SparseMatricesCOO

using LinearAlgebra, Random, SparseArrays, Test

import ExactPenalty: solve!

Random.seed!(0)

include("allocations-macro.jl")

include("instances/instance-reader.jl")
include("instances/instance-generator.jl")

include("test-quasi-newton.jl")
include("test-subsolvers.jl")

testset "CUTEst-default" begin
  @test isnothing(Base.get_extension(ExactPenalty, :ExactPenaltyMUMPSExt)) # Check that the extension is not loaded.
  include("test/cutest.jl")
end

using MPI, MUMPS
@testset "CUTEst-MUMPS" begin
  @test !isnothing(Base.get_extension(ExactPenalty, :ExactPenaltyMUMPSExt))
  include("test/cutest.jl")
end
