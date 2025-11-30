using CUTEst

problem_names = ["BT1", "MSS1", "SSINE", "VANDAMIUS"]

tol = 1e-3

# Test R2
@testset "L2Penalty - R2" begin
  for name in problem_names
    nlp = CUTEstModel(name)
    stats = L2Penalty(nlp, verbose = 100, atol = tol, rtol = tol)
  end
end