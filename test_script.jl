using LinearAlgebra

using RegularizedOptimization, SolverCore, ShiftedProximalOperators, NLPModelsModifiers
using CUTEst, ExactPenalty, NLPModels, RegularizedProblems

import ShiftedProximalOperators: shifted,  ShiftedProximableFunction, shift!, prox!


nmax = 300
max_time = 300.0 #Each problem has 5 minutes.
problem_names = ["MSS1"] # TODO: add much more problems

problem_names = CUTEst.select_sif_problems(min_con=1, max_var= nmax, only_equ_con=true, only_free_var=true)

using NLPModelsIpopt

tol = 1e-9

for (idx, name) in enumerate(problem_names)
    break
    nlp = CUTEstModel(name)
    if nlp.meta.ncon > nlp.meta.nvar
        continue
    end

    println("SOLVING PROBLEM $name")
    stats = ipopt(nlp, print_level = 0, dual_inf_tol=tol,
      constr_viol_tol=tol,
      compl_inf_tol=Inf,
      acceptable_iter=0,
      s_max = 1e12,
      nlp_scaling_method = "none",
      tol = tol,
      x0 = nlp.meta.x0)
    println("IPOPT COMPUTED WITH THE FOLLOWING COUNTERS $(nlp.counters)")
    ipopt_obj = neval_obj(nlp)
    ipopt_status = stats.status
    NLPModels.reset!(nlp.counters)
    x0 = nlp.meta.x0 
    ∇ = grad(nlp, x0)
    J = jac(nlp, x0)

    y = Matrix(J')\∇

    stats = L2Penalty(
    nlp,
    max_time=max_time,
    max_iter= Int64(typemax(Int32)),
    max_eval=Int64(typemax(Int32)),
    sub_max_iter=100,
    ktol= tol,
    atol=0.0,
    rtol=0.0,
    verbose = 0,
    sub_verbose=0,
    neg_tol=10*sqrt(tol),
    subsolver=R2NSolver,
    #h_type = :Sparse,
    #type = :exact,
    τ=max(norm(y, 1), 1.0),
    callback=(nlp, solver, stats) -> begin
        stats.primal_feas = norm(solver.subsolver.ψ.b, Inf)
        stats.dual_feas = norm(solver.subsolver.s, Inf)*solver.substats.solver_specific[:sigma]
        if stats.primal_feas/stats.solver_specific[:theta] > typeof(stats.primal_feas)(100) && stats.solver_specific[:theta] < tol && stats.primal_feas > tol
            # If θ << c(xk), θ is small but c(xk) is large then the problem might be infeasible, see pb VANDAMIUS. 
            stats.status = :infeasible
        end
        if stats.primal_feas < tol && stats.dual_feas < tol
            stats.status = :user          
        end
        end,
    sub_callback=(nlp, solver, stats) -> begin
        stats.dual_feas = norm(solver.s1, Inf)*stats.solver_specific[:sigma] # s = J(x)^T y + ∇f(x)/σ.
        if norm(solver.ψ.b, Inf) < tol && stats.dual_feas < tol
        stats.status = :user 
        end
        if stats.solver_specific[:smooth_obj] < -1/eps(typeof(stats.solver_specific[:smooth_obj])) # f(xk) < -1e16
        # The subsolver is failing, L2Penalty will increase the penalty parameter accordingly
        solver.xk .= nlp.meta.x0
        stats.status = :failed
        end
        if norm(solver.s) < eps(eltype(solver.s)) && stats.iter > 1
        stats.status = :user
        end
    end
    )
    println("OUR METHOD $(nlp.counters)")
    emoji = ipopt_obj > neval_obj(nlp) ? "✅" : "❌"
    println(ipopt_status)
    println(stats.status)
    emoji_ipopt_stats = ipopt_status == :first_order ? "✅" : "❌"
    emoji_method_stats = stats.status == :user ? "✅" : "❌"
    println("$emoji FOR PROBLEM $name")
    println("NOTE THAT IPOPT HAD STATUS $emoji_ipopt_stats")
    println("NOTE THAT OUR METHOD HAD STATUS $emoji_method_stats")
    println("----------------------------------------------") 

    finalize(nlp)
    #emoji == "❌" && break
end

#error("done")
tol = 1e-9
nlp = CUTEstModel("BT1")


@info "L2Penalty solver with minres_qlp on problem $(nlp.meta.name)"

x0 = nlp.meta.x0 
∇ = grad(nlp, x0)
J = jac(nlp, x0)

y = Matrix(J')\∇
#println(norm(y, 1))
dual_feas = norm(∇ - J'*y, Inf)
#println(dual_feas)
reduction_factor = 1e-2

stats = L2Penalty(
nlp,
max_iter= 4,#Int64(typemax(Int32)),
max_eval= Int64(typemax(Int32)),
ktol= 0.0,#max(reduction_factor*dual_feas, tol), #tol,
atol=0.0,
rtol=0.0,
verbose = 1,
sub_verbose=1,
neg_tol=10*sqrt(tol),
subsolver=R2NSolver,
#h_type = :Sparse,
#type = :exact,
τ=max(norm(y, 1), 1.0),
callback=(nlp, solver, stats) -> begin
    stats.primal_feas = norm(solver.subsolver.ψ.b, Inf)
    stats.dual_feas = norm(solver.subsolver.s1, Inf)*solver.substats.solver_specific[:sigma_cauchy]
    if stats.primal_feas/stats.solver_specific[:theta] > typeof(stats.primal_feas)(100) && stats.solver_specific[:theta] < tol && stats.primal_feas > tol
        # If θ << c(xk), θ is small but c(xk) is large then the problem might be infeasible, see pb VANDAMIUS. 
        stats.status = :infeasible
    end
    if stats.primal_feas < tol && stats.dual_feas < tol
        stats.status = :user          
    end
    end,
sub_callback=(nlp, solver, stats) -> begin
    stats.dual_feas = norm(solver.s1, Inf)*stats.solver_specific[:sigma_cauchy] # s = J(x)^T y + ∇f(x)/σ.
    println("DUAL FEAS $(stats.dual_feas)")
    if stats.iter == 0
        set_solver_specific!(stats, :atol, max(stats.dual_feas*reduction_factor, tol))    
    end
    if stats.dual_feas < stats.solver_specific[:atol]
        stats.status = :user 
    end
    if stats.solver_specific[:smooth_obj] < -1/eps(typeof(stats.solver_specific[:smooth_obj])) # f(xk) < -1e16
    # The subsolver is failing, L2Penalty will increase the penalty parameter accordingly
    solver.xk .= nlp.meta.x0
    stats.status = :failed
    end
    if norm(solver.s) < eps(eltype(solver.s)) && stats.iter > 1
    stats.status = :user
    end
end
)
println(stats)
finalize(nlp)
