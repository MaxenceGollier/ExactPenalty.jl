function get_linear_solver(
  solver::L2PenaltySolver{T, V, S, PB}
) where{T, V, H, WP, MSS <: MoreSorensenSolver{T, V, H, WP},S <: PenaltyR2NSolver{T, V, MSS}, PB}
  if WP <: PenaltyLDLTWorkspace
    return "LDLFactorizations.jl v$(pkgversion(LDLFactorizations))"
  else
    return ""
  end
end

function introduction_message(solver::L2PenaltySolver, nlp::AbstractNLPModel)
  return """

  This is ExactPenalty.jl v$(pkgversion(@__MODULE__)).
  Running with linear solver $(get_linear_solver(solver)).

  $nlp
  """
end

function header_message()
  return log_header(
      [:iter, :fx, :primal_feas, :dual_feas, :lg_tau, :norms, :lg_sigma],
      [Int, Float64, Float64, Float64, Float64, Float64, Float64],
      hdr_override = Dict{Symbol,String}(
        :iter => "iter",
        :fx => "f(x)",
        :primal_feas => "Primal",
        :dual_feas => "Dual",
        :pr_feas_k => "pεₖ",
        :lg_tau => "lg(τ)",
        :norms => "‖s‖",
        :lg_sigma => "lg(σ)",
      ),
      colsep = 2,
    )
end

function log_iteration(
  solver::L2PenaltySolver,
  nlp::AbstractNLPModel,
  stats::GenericExecutionStats,
)
  log = ""
  log *= stats.iter * " "
  log *= @sprintf("%+8.7e", stats.objective) * " "
  log *= @sprintf("%3.2e", stats.primal_feas) * " "
  log *= @sprintf("%3.2e", stats.dual_feas) * " "
  log *= @sprintf("%+2.1f", log10(stats.solver_specific[:tau])) * " "
  log *= @sprintf("%3.2e", norm(solver.substats.solution)) * " "
  log *= @sprintf("%+2.1f", log10(stats.solver_specific[:sigma])) * " "
  return log
end
