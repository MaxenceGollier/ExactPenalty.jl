using ExactPenalty
using LinearOperators, RegularizedOptimization, RegularizedProblems, ShiftedProximalOperators, SolverCore, SparseMatricesCOO
using LinearAlgebra, Random, Test

include("allocations-macro.jl")

include("instances/instance-reader.jl")
include("instances/instance-generator.jl")

for (root, dirs, files) in walkdir(@__DIR__)
  for file in files
    if isnothing(match(r"^test-.*\.jl$", file))
      continue
    end
    title = titlecase(replace(splitext(file[6:end])[1], "-" => " "))
    @testset "$title" begin
      include(file)
    end
  end
end
