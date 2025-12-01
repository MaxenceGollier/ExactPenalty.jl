using CUTEst, NLPModels, NLPModelsModifiers

problem_names = ["BT1", "MSS1", "SSINE", "VANDANIUMS"]
expected_status = [:first_order, :first_order, :infeasible, :infeasible]

tol = 1e-3

# Test a simple problem
@testset "BT1" begin
  nlp = CUTEstModel("BT1")
  primal_solution = [1, 0]
  dual_solution = [99.5]

  # Test with R2
  @testset "R2" begin
    stats = L2Penalty(nlp, atol = tol, rtol = tol)

    # Test whether the outputs are well defined
    @test stats.status == :first_order
    @test norm(primal_solution - stats.solution) ≤ 10*tol
    @test abs(stats.objective - obj(nlp, primal_solution)) ≤ 10*tol
    @test stats.primal_feas == norm(cons(nlp, stats.solution))
    @test norm(stats.multipliers - dual_solution) ≤ 10*tol
    @test abs(stats.dual_feas - norm(jtprod(nlp, stats.solution, stats.multipliers) - grad(nlp, stats.solution))) ≤ eps(Float64)

    # Test stability and allocations
    solver = L2PenaltySolver(nlp)
    stats_optimized = GenericExecutionStats(nlp)
    @test @wrappedallocs(solve!(solver, nlp, stats_optimized, atol = tol, rtol = tol)) == 0
    
    # Test that the second calling form gives the same output
    @test stats_optimized.status == stats.status
    @test stats_optimized.objective == stats.objective
    @test stats_optimized.primal_feas == stats.primal_feas
    @test stats_optimized.dual_feas == stats.dual_feas
    @test all(stats_optimized.multipliers .== stats.multipliers)
    @test all(stats_optimized.solution .== stats.solution)
    @test stats_optimized.iter == stats.iter
  end
  # Test with R2N
  @testset "R2N (LBFGS)" begin
    stats = L2Penalty(LBFGSModel(nlp), atol = tol, rtol = tol, subsolver = R2NSolver)

    @test stats.status == :first_order
    @test norm(primal_solution - stats.solution) ≤ 10*tol
    @test abs(stats.objective - obj(nlp, primal_solution)) ≤ 10*tol
    @test stats.primal_feas == norm(cons(nlp, stats.solution))
    @test norm(stats.multipliers - dual_solution) ≤ 10*tol
    @test abs(stats.dual_feas - norm(jtprod(nlp, stats.solution, stats.multipliers) - grad(nlp, stats.solution))) ≤ eps(Float64)

    # Test stability and allocations
    solver = L2PenaltySolver(LBFGSModel(nlp), subsolver = R2NSolver)
    stats_optimized = GenericExecutionStats(nlp)
    @test @wrappedallocs(solve!(solver, nlp, stats_optimized, atol = tol, rtol = tol)) == 0
    
    # Test that the second calling form gives the same output
    @test stats_optimized.status == stats.status
    @test stats_optimized.objective == stats.objective
    @test stats_optimized.primal_feas == stats.primal_feas
    @test stats_optimized.dual_feas == stats.dual_feas
    @test all(stats_optimized.multipliers .== stats.multipliers)
    @test all(stats_optimized.solution .== stats.solution)
    @test stats_optimized.iter == stats.iter
  end

  finalize(nlp)
end

# Test an infeasible problem

# Test an ill-conditionned problem
# TODO: Add MSS1
