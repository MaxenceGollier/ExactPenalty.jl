function construct_minres_qlp_workspace(H::M, u1::V) where{M <: AbstractLinearOperator, V <: AbstractVector}
  return MinresQlpWorkspace(H, u1)
end

function update_workspace!(solver_workspace::KrylovWorkspace, H, B, A, σ, α)
  H.α = α
  H.σ = σ
end

function update_workspace!(solver_workspace::KrylovWorkspace, H, m, α)
  H.α = α
end

function solve_system!(workspace::KrylovWorkspace, H::M, u::V; kwargs...) where{M <: AbstractLinearOperator, V <: AbstractVector}
  krylov_solve!(workspace, H, u; kwargs...)
end

function get_solution!(x::V, workspace::KrylovWorkspace) where{V <: AbstractVector}
  x .= workspace.x
end

function get_status(workspace::KrylovWorkspace)
  workspace.stats.solved && return :success
  return :failed
end