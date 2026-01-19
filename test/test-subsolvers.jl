# Get instances 
instances = "instances/"
instances = filter(f -> isfile(f) && endswith(f, ".txt"),
                    joinpath.(instances, readdir(instances)))
# Get subsolvers
solver_names = ["TRMoreSorensenLinOpSolver"]
solvers = [TRMoreSorensenLinOpSolver]

# Test on real instances
for (solver_name, solver_constructor) in zip(solver_names, solvers)
  @testset "$solver_name" begin
    @testset "Well-conditionned" begin
      n, m = 10, 2
      small_instance_boundary, solution = generate_instance(n, m, 0.5, Hessian_modifier = LinearOperator)
      solver = eval(solver_constructor)(small_instance_boundary)
      stats = RegularizedExecutionStats(small_instance_boundary)
      @test @wrappedallocs(solve!(solver, small_instance_boundary, stats, atol = 1e-9)) == 0 
      @test norm(solution[:u] - stats.solution) <= 1e-9
      @test norm(solution[:y] - solver.x1[n+1:end]) <= 1e-9
      @test abs(solution[:tau] - norm(solver.x1[n+1:end])) <= 1e-9

      small_instance_interior, solution = generate_instance(n, m, 0.0, Hessian_modifier = LinearOperator)
      solver = eval(solver_constructor)(small_instance_interior)
      stats = RegularizedExecutionStats(small_instance_interior)
      @test @wrappedallocs(solve!(solver, small_instance_interior, stats, atol = 1e-9)) == 0 
      @test norm(solution[:u] - stats.solution) <= 1e-6
      @test norm(solution[:y] - solver.x1[n+1:end]) <= 1e-6
      @test norm(solver.x1[n+1:end]) <= solution[:tau]

      n, m = 100, 20
      medium_instance_boundary, solution = generate_instance(n, m, 0.5, Hessian_modifier = LinearOperator)
      solver = eval(solver_constructor)(medium_instance_boundary)
      stats = RegularizedExecutionStats(medium_instance_boundary)
      @test @wrappedallocs(solve!(solver, medium_instance_boundary, stats, atol = 1e-9)) == 0 
      @test norm(solution[:u] - stats.solution) <= 1e-9
      @test norm(solution[:y] - solver.x1[n+1:end]) <= 1e-9
      @test abs(solution[:tau] - norm(solver.x1[n+1:end])) <= 1e-9

      medium_instance_interior, solution = generate_instance(n, m, 0.0, Hessian_modifier = LinearOperator)
      solver = eval(solver_constructor)(medium_instance_interior)
      stats = RegularizedExecutionStats(medium_instance_interior)
      @test @wrappedallocs(solve!(solver, medium_instance_interior, stats, atol = 1e-9)) == 0 
      @test norm(solution[:u] - stats.solution) <= 1e-6
      @test norm(solution[:y] - solver.x1[n+1:end]) <= 1e-6
      @test norm(solver.x1[n+1:end]) <= solution[:tau]
    end

    @testset "Ill-conditionned" begin
      for instance in instances
        reg_nlp = read_instance(instance, type = Float64, Hessian_modifier = LinearOperator)
        n = reg_nlp.model.meta.nvar
        solver = eval(solver_constructor)(reg_nlp)
        stats = RegularizedExecutionStats(reg_nlp)
        @test @wrappedallocs(solve!(solver, reg_nlp, stats)) == 0 
        instance_name = basename(instance)
        if occursin("boundary", instance_name)
          @test abs(norm(solver.x1[n+1:end]) - reg_nlp.h.h.lambda) <= 1e-6
        end
      end
    end
  end
end