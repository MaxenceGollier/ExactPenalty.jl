using JLD2

using CUTEst, NLPModelsIpopt, SolverBenchmark

nmax = 300
problem_names = CUTEst.select_sif_problems(min_con=1, max_var= nmax, only_equ_con=true, only_free_var=true)
problem_list = (CUTEstModel(name) for name in problem_names)

imprecise_tol = 1e-3
precise_tol = 1e-9

solvers = Dict(
  :ipopt_exact_imprecise =>
    nlp -> ipopt(
      nlp,
      print_level=0,
      tol=imprecise_tol,
      dual_inf_tol=imprecise_tol,
      constr_viol_tol=imprecise_tol,
      compl_inf_tol=Inf,
      acceptable_iter=0,
      s_max = 1e12,
      nlp_scaling_method = "none",
    ),
  :ipopt_exact_precise =>
    nlp -> ipopt(
      nlp,
      print_level=0,
      tol=precise_tol,
      dual_inf_tol=precise_tol,
      constr_viol_tol=precise_tol,
      compl_inf_tol=Inf,
      acceptable_iter=0,
      s_max = 1e12,
      nlp_scaling_method = "none",
    ),
  :ipopt_lbfgs_imprecise =>
    nlp -> ipopt(
      nlp,
      print_level=0,
      tol=imprecise_tol,
      dual_inf_tol=imprecise_tol,
      constr_viol_tol=imprecise_tol,
      compl_inf_tol=Inf,
      acceptable_iter=0,
      s_max = 1e12,
      hessian_approximation="limited-memory",
      nlp_scaling_method = "none",
    ),
  :ipopt_lbfgs_precise =>
    nlp -> ipopt(
      nlp,
      print_level=0,
      tol=precise_tol,
      dual_inf_tol=precise_tol,
      constr_viol_tol=precise_tol,
      compl_inf_tol=Inf,
      acceptable_iter=0,
      s_max = 1e12,
      hessian_approximation="limited-memory",
      nlp_scaling_method = "none",
    )

)

stats = bmark_solvers(solvers, problem_list, skipif= nlp -> nlp.meta.ncon ≥ nlp.meta.nvar)
@save "benchmark/result/stats_ipopt.jld2" stats
