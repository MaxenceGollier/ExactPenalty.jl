struct PenaltyMumpsWorkspace{WP <: Mumps}
  M::WP
  x::V
  n::Int
  m::Int
end

function construct_mumps_workspace(H::M, u1::V, n, m) where{M <: K2, V <: AbstractVector}
  T = eltype(u1)
  icntl = default_icntl[:]
  icntl[10] = 10 # Perform at most 10 iterative refinement steps
  mumps = Mumps{T}(mumps_symmetric, icntl, default_cntl)
  associate_matrix!(mumps, H; unsafe=true)
  return PenaltyMumpsWorkspace(mumps, similar(u1), n, m)
end

function update_workspace!(solver_workspace::PenaltyMumpsWorkspace, B, A, σ, α)
  #TODO: see K2.jl
end

function update_workspace!(solver_workspace::PenaltyMumpsWorkspace, α)
  #TODO: see K2.jl
end

function solve_system!(workspace::PenaltyMumpsWorkspace, u::V) where{V <: AbstractVector}
  mumps_solve!(workspace.x, workspace.M, u)
end

function get_solution!(x::V, workspace::PenaltyMumpsWorkspace) where{V <: AbstractVector}
  x .= workspace.x
end

function get_status(workspace::PenaltyMumpsWorkspace)
  workspace.M.stats.solved && return :success
  return :failed
end