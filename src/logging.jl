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

const W_ITER  = 6
const W_LARGE = 16
const W_MED   = 12
const W_SMALL = 8

const FMT_OBJ   = "%+-16.7e"
const FMT_MED   = "%-12.2e"
const FMT_SMALL = "%+-8.1f"

function separator()
    repeat("-", textwidth(header_message()))
end

function header_message()
    @sprintf(
        "%-*s%-*s%-*s%-*s%-*s%-*s%-*s",
        W_ITER,  "Iter",
        W_LARGE, "Objective",
        W_MED,   "Primal_Feas",
        W_MED,   "Dual_Feas",
        W_SMALL, "log(τ)",
        W_MED,   "‖x‖₂",
        W_SMALL, "log(σ)",
    )
end

function log_iteration(solver, nlp, stats)
    @sprintf(
        "%-6d%-+16.7e%-12.2e%-12.2e%-+8.1f%-12.2e%-+8.1f",
        stats.iter,
        stats.objective,
        stats.primal_feas,
        stats.dual_feas,
        log10(solver.substats.solver_specific[:tau]),
        norm(solver.substats.solution),
        log10(solver.substats.solver_specific[:sigma]),
    )
end
