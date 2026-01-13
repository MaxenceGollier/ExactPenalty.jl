using CUTEst, NLPModels, NLPModelsModifiers, ExactPenalty, SolverCore, RegularizedOptimization, LinearOperators

macro wrappedallocs(expr)
  kwargs = [a for a in expr.args if isa(a, Expr)]
  args = [a for a in expr.args if isa(a, Symbol)]

  argnames = [gensym() for a in args]
  kwargs_dict = Dict{Symbol, Any}(a.args[1] => a.args[2] for a in kwargs if a.head == :kw)
  quote
    function g($(argnames...); kwargs_dict...)
      $(Expr(expr.head, argnames..., kwargs...))
      @allocated $(Expr(expr.head, argnames..., kwargs...))
    end
    $(Expr(:call, :g, [esc(a) for a in args]...))
  end
end

problem_names = ["BT1", "MSS1", "SSINE", "VANDANIUMS"]
expected_status = [:first_order, :first_order, :infeasible, :infeasible]

tol = 1e-3

nlp = CUTEstModel("BT1")

LBFGS_model = LBFGSModel(nlp)

#stats = L2Penalty(LBFGS_model, atol = tol, rtol = tol, subsolver = R2NSolver)

solver = L2PenaltySolver(LBFGS_model, subsolver = R2NSolver)
stats_optimized = ExactPenaltyExecutionStats(LBFGS_model)
println(@wrappedallocs solve!(solver, LBFGS_model, stats_optimized, atol = 1e-3, rtol = 1e-3))

#println(stats)
#println(stats_optimized)
finalize(nlp)