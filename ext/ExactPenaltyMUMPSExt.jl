module ExactPenaltyMUMPSExt

using MUMPS, MPI
using ExactPenalty

using LinearAlgebra, SparseMatricesCOO

import ExactPenalty: construct_mumps_workspace, solve_system!, update_workspace!
import ExactPenalty: get_inertia, get_solution!, get_status
import ExactPenalty: set_dual_inertia!, set_primal_inertia!

function __init__()
  ENV["MUMPS_SEQ"] = "1"
  MPI.Init()
end

include("MUMPS/mumps_workspace.jl")

end