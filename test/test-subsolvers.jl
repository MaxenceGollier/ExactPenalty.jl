# Get instances 
instances = "instances/"
instances = filter(isfile, joinpath.(instances, readdir(instances)))

# Get subsolvers
solver_names = ["TRMoreSorensenLinOpSolver"]
solvers = [TRMoreSorensenLinOpSolver]

for (solver_name, solver) in zip(solver_names, solvers)
  @testset "$solver_name" begin
    for instance in instances
      reg_nlp = read_instance(instance, type = Float64, Hessian_modifier = LinearOperator)
      solver = eval(solver)(reg_nlp)
      stats = RegularizedExecutionStats(reg_nlp)
      @test @wrappedallocs(solve!(solver, reg_nlp, stats)) == 0 
    end
  end
end