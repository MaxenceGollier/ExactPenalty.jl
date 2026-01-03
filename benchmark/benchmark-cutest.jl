using CUTEst, ExactPenalty, NLPModelsModifiers, RegularizedOptimization, ShiftedProximalOperators, SolverBenchmark

nmax = 300
problem_names = CUTEst.select_sif_problems(min_con=1, max_var= nmax, only_equ_con=true, only_free_var=true)
problem_list = (CUTEstModel(name) for name in problem_names)

imprecise_tol = 1e-3
precise_tol = 1e-9

solvers = Dict(
  :l2penalty_exact_imprecise =>
    nlp -> L2Penalty(
      nlp,
      verbose   = 0,
      atol      = imprecise_tol,
      rtol      = imprecise_tol,
      subsolver = R2NSolver,
    ),
  :l2penalty_exact_precise =>
    nlp -> L2Penalty(
      nlp,
      verbose   = 0,
      atol      = precise_tol,
      rtol      = precise_tol,
      subsolver = R2NSolver,
    ),
  :l2penalty_lbfgs_imprecise =>
    nlp -> L2Penalty(
      LBFGSModel(nlp),
      verbose   = 0,
      atol      = imprecise_tol,
      rtol      = imprecise_tol,
      subsolver = R2NSolver,
    ),
  :l2penalty_lbfgs_precise =>
    nlp -> L2Penalty(
      LBFGSModel(nlp),
      verbose   = 0,
      atol      = precise_tol,
      rtol      = precise_tol,
      subsolver = R2NSolver,
    ),
  :l2penalty_r2_imprecise =>
    nlp -> L2Penalty(
      nlp,
      verbose   = 0,
      atol      = imprecise_tol,
      rtol      = imprecise_tol,
      subsolver = R2Solver,
    ),
  :l2penalty_r2_precise =>
    nlp -> L2Penalty(
      nlp,
      verbose   = 0,
      atol      = precise_tol,
      rtol      = precise_tol,
      subsolver = R2Solver,
    ),
)

stats = bmark_solvers(solvers, problem_list, skipif= nlp -> nlp.meta.ncon ≥ nlp.meta.nvar)
@save "stats.jld2" stats
