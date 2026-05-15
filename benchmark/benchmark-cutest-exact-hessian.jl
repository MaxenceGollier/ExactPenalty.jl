using JLD2

using CUTEst, ExactPenalty, NLPModelsModifiers, SolverBenchmark

nmax = 10000
problem_names = CUTEst.select_sif_problems(
  min_con = 1,
  max_var = nmax,
  only_equ_con = true,
  only_free_var = true,
)
problem_list = (CUTEstModel(name) for name in problem_names)

tol = 1e-6

solvers = Dict(
  :l2penalty_exact =>
    nlp -> L2Penalty(nlp, verbose = 0, atol = tol, rtol = tol),
)

stats = bmark_solvers(solvers, problem_list, skipif = nlp -> nlp.meta.ncon ≥ nlp.meta.nvar)
@save "benchmark/result/stats_exact.jld2" stats
