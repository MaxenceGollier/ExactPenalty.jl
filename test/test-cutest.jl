using CUTEst, NLPModels, NLPModelsModifiers

problem_names = ["BT1", "MSS1", "SSINE", "VANDANIUMS"]
expected_status = [:first_order, :first_order, :infeasible, :infeasible]

tol = 1e-3

function test_problem(name, primal_solution, dual_solution, expected_status)
  nlp = CUTEstModel(name)

  # Test with R2
  @testset "R2" begin
    stats = L2Penalty(nlp, atol = tol, rtol = tol)

    # Test whether the outputs are well defined
    @test stats.status == expected_status
    if expected_status == :first_order
      @test norm(primal_solution - stats.solution) ≤ 10*tol
      @test abs(stats.objective - obj(nlp, primal_solution)) ≤ 10*tol
      @test norm(stats.multipliers - dual_solution) ≤ 10*tol
      @test abs(stats.dual_feas - norm(jtprod(nlp, stats.solution, stats.multipliers) - grad(nlp, stats.solution), Inf)) ≤ eps(Float64)
    end
    @test stats.primal_feas == norm(cons(nlp, stats.solution), Inf)

    # Test stability and allocations
    solver = L2PenaltySolver(nlp)
    stats_optimized = ExactPenaltyExecutionStats(nlp)
    @test @wrappedallocs(solve!(solver, nlp, stats_optimized, atol = 1e-3, rtol = 1e-3)) == 0
    
    # Test that the second calling form gives the same output
    @test stats_optimized.status == stats.status
    @test stats_optimized.objective == stats.objective
    @test stats_optimized.primal_feas == stats.primal_feas
    @test stats_optimized.dual_feas == stats.dual_feas
    @test all(stats_optimized.multipliers .== stats.multipliers)
    @test all(stats_optimized.solution .== stats.solution)
    @test stats_optimized.iter == stats.iter

    stats = L2Penalty(nlp, atol = tol, rtol = tol, primal_feasibility_mode = :decrease, dual_feasibility_mode = :decrease)

    # Test whether the outputs are well defined
    @test stats.status == expected_status
    if expected_status == :first_order
      @test norm(primal_solution - stats.solution) ≤ 10*tol
      @test abs(stats.objective - obj(nlp, primal_solution)) ≤ 10*tol
      @test norm(stats.multipliers - dual_solution) ≤ 10*tol
    end

    # Test stability and allocations
    solver = L2PenaltySolver(nlp)
    stats_optimized = ExactPenaltyExecutionStats(nlp)
    
    @test @wrappedallocs(solve!(solver, nlp, stats_optimized, atol = 1e-3, rtol = 1e-3, primal_feasibility_mode = :decrease, dual_feasibility_mode = :decrease)) == 0

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
    LBFGS_model = LBFGSModel(nlp)
    stats = L2Penalty(LBFGS_model, atol = tol, rtol = tol, subsolver = R2NSolver)

    @test stats.status == expected_status
    if expected_status == :first_order
      @test norm(primal_solution - stats.solution) ≤ 10*tol
      @test abs(stats.objective - obj(nlp, primal_solution)) ≤ 10*tol
      @test norm(stats.multipliers - dual_solution) ≤ 10*tol
      @test abs(stats.dual_feas - norm(jtprod(nlp, stats.solution, stats.multipliers) - grad(nlp, stats.solution), Inf)) ≤ eps(Float64)
    end
    @test stats.primal_feas == norm(cons(nlp, stats.solution), Inf)

    # Test stability and allocations
    solver = L2PenaltySolver(LBFGS_model, subsolver = R2NSolver)
    stats_optimized = ExactPenaltyExecutionStats(LBFGS_model)
    @test @wrappedallocs(solve!(solver, LBFGS_model, stats_optimized, atol = 1e-3, rtol = 1e-3)) == 0
    
    # Test that the second calling form gives the same output
    @test stats_optimized.status == stats.status
    @test stats_optimized.objective == stats.objective
    @test stats_optimized.primal_feas == stats.primal_feas
    @test stats_optimized.dual_feas == stats.dual_feas
    @test all(stats_optimized.multipliers .== stats.multipliers)
    @test all(stats_optimized.solution .== stats.solution)
    @test stats_optimized.iter == stats.iter

    stats = L2Penalty(LBFGS_model, atol = tol, rtol = tol, subsolver = R2NSolver, primal_feasibility_mode = :decrease, dual_feasibility_mode = :decrease)

    @test stats.status == expected_status
    if expected_status == :first_order
      @test norm(primal_solution - stats.solution) ≤ 10*tol
      @test abs(stats.objective - obj(nlp, primal_solution)) ≤ 10*tol
      @test norm(stats.multipliers - dual_solution) ≤ 10*tol
    end

    # Test stability and allocations
    solver = L2PenaltySolver(LBFGS_model, subsolver = R2NSolver)
    stats_optimized = ExactPenaltyExecutionStats(LBFGS_model)
    @test @wrappedallocs(solve!(solver, LBFGS_model, stats_optimized, atol = 1e-3, rtol = 1e-3, primal_feasibility_mode = :decrease, dual_feasibility_mode = :decrease)) == 0
    
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
# Test a simple problem
@testset "BT1" begin
  primal_solution = [1, 0]
  dual_solution = [99.5]

  test_problem("BT1", primal_solution, dual_solution, :first_order)
end

@testset "VANDANIUMS" begin
  test_problem("VANDANIUMS", Float64[], Float64[], :infeasible)
end

# Test an ill-conditionned problem
# TODO: Add MSS1
