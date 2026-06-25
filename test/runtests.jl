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

for (root, dirs, files) in walkdir(@__DIR__)
  for file in files
    if isnothing(match(r"^test-.*\.jl$", file))
      continue
    end
    title = titlecase(replace(splitext(file[6:end])[1], "-" => " "))
    if title == "Cutest"

      # Test without the MUMPS extension.
      @testset "$title-Default" begin
        @test isnothing(Base.get_extension(ExactPenalty, :ExactPenaltyMUMPSExt)) # Check that the extension is not loaded.
        include(file)
      end

      # Load and test the MUMPS extension.
      using MPI, MUMPS
      @testset "$title-MUMPS" begin
        @test !isnothing(Base.get_extension(ExactPenalty, :ExactPenaltyMUMPSExt))
        include(file)
      end

    else
      @testset "$title" begin
        include(file)
      end
    end
  end
end
