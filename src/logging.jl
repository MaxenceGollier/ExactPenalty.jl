function get_linear_solver(
  solver::L2PenaltySolver{T, V, S, PB}
) where{T, V, H, WP, MSS <: MoreSorensenSolver{T, V, H, WP},S <: PenaltyR2NSolver{T, V, MSS}, PB}
  if WP <: PenaltyLDLTWorkspace
    return "LDLFactorizations.jl v$(pkgversion(LDLFactorizations))"
  elseif WP <: AbstractMUMPSWorkspace
    # The MUMPS struct exposes the version number.
    version = solver.subsolver.subsolver.workspace.M.version_number
    version = decode_mumps_version(version)
    return "MUMPS v$(version)"
  else
    return ""
  end
end

function decode_mumps_version(v::NTuple)
    bytes = collect(UInt8.(v))
    i = findfirst(iszero, bytes)
    if i === nothing
        return String(bytes)
    else
        return String(bytes[1:i-1])
    end
end

function introduction_message(solver::L2PenaltySolver, nlp::AbstractNLPModel)
  return """

  This is ExactPenalty.jl v$(pkgversion(@__MODULE__)).
  Running with linear solver $(get_linear_solver(solver)).

  $nlp
  """
end

function introduction_message(solver::PenaltyR2NSolver, nlp::AbstractNLPModel, stats::GenericExecutionStats)
  return @sprintf("
        |  Solving subproblem minₓ f(x) + %-3.2e ‖c(x)‖₂ with tolerance %-3.2e...", 
  stats.solver_specific[:tau], stats.solver_specific[:dual_ktol])
end

const W_ITER  = 6
const W_LARGE = 16
const W_MED   = 12
const W_SMALL = 8

const FMT_OBJ   = "%+-16.7e"
const FMT_MED   = "%-12.2e"
const FMT_SMALL = "%+-8.1f"

function separator(;type = :outer_loop)
  if type == :outer_loop
    return repeat("-", textwidth(header_message(type = type)))
  elseif type == :inner_loop
    return "  | " * repeat("-", textwidth(header_message(type = type))-4)
  end
end

function header_message(; type = :outer_loop)
  if type == :outer_loop
    return @sprintf(
        "%-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s",
        W_ITER,  "Iter",
        W_LARGE, "Objective",
        W_MED,   "pfeas",
        W_MED,   "dfeas",
        W_MED,   "τ",
        W_MED,   "ptol",
        W_MED,   "dtol",
        W_MED,   "‖x‖₂",
    )
  elseif type == :inner_loop
    return @sprintf(
        "  | %-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s",
        W_ITER,  "Iter",
        W_LARGE, "Objective",
        W_MED,   "pfeas",
        W_MED,   "dfeas",
        W_MED,   "σ",
        W_MED,   "ρ",
        W_MED,   "‖x‖₂",
        W_MED,   "‖s‖₂",
    )
  end
end

function log_iteration(solver, nlp, stats; type = :outer_loop)
  if type == :outer_loop
    return @sprintf(
      "%-6d%-+16.7e%-12.2e%-12.2e%-12.2e%-12.2e%-12.2e%-12.2e",
      stats.iter,
      stats.objective,
      stats.primal_feas,
      stats.dual_feas,
      solver.substats.solver_specific[:tau],
      solver.substats.solver_specific[:primal_ktol],
      solver.substats.solver_specific[:dual_ktol],
      norm(solver.x),
    )
  elseif type == :inner_loop
    return @sprintf(
      "  | %-6d%-+16.7e%-12.2e%-12.2e%-12.2e%-+12.2e%-12.2e%-12.2e",
      stats.iter,
      stats.objective,
      stats.primal_feas,
      stats.dual_feas,
      stats.solver_specific[:sigma],
      stats.solver_specific[:rho],
      norm(solver.xk),
      norm(solver.s),
    )
  end
end

function conclusion_message(solver, nlp, stats; type = :outer_loop)
  if type == :outer_loop
    return ""
  elseif type == :inner_loop
    return "  |
        |  Subproblem solved with status $(stats.status), after $(stats.iter) iterations. 
        |  Reached dfeas = $(stats.dual_feas)"
  end
end