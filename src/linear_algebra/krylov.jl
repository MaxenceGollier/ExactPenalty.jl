struct PenaltyKrylovWorkspace{WP <: KrylovWorkspace, OP <: OpK2}
  M::WP
  H::OP
  n::Int
  m::Int
end

function construct_minres_qlp_workspace(H::M, u1::V, n, m) where{M <: OpK2, V <: AbstractVector}
  return PenaltyKrylovWorkspace(MinresQlpWorkspace(H, u1), H, n, m)
end

function update_workspace!(solver_workspace::PenaltyKrylovWorkspace, B, A, σ, α)
  solver_workspace.H.B = B
  solver_workspace.H.A = A
  solver_workspace.H.α = α
  solver_workspace.H.σ = σ
end

function update_workspace!(solver_workspace::PenaltyKrylovWorkspace, α)
  solver_workspace.H.α = α
end

function solve_system!(workspace::PenaltyKrylovWorkspace, u::V) where{V <: AbstractVector}
  T = eltype(u)
  krylov_solve!(workspace.M, workspace.H, u, atol = eps(T)^0.8, rtol = eps(T)^0.8, Artol  = eps(T)^0.7)
end

function get_solution!(x::V, workspace::PenaltyKrylovWorkspace) where{V <: AbstractVector}
  x .= workspace.M.x
end

function get_status(workspace::PenaltyKrylovWorkspace)
  workspace.M.stats.solved && return :success
  return :failed
end