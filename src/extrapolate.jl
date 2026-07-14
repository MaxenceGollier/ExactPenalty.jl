function extrapolate!(
  x::V,
  solver::L2PenaltySolver{T,V,S,PB},
  τ₂::T,
  τ₁::T,
) where {T,V,S,N<:QuasiNewtonModel{T,V},PB<:L2PenalizedProblem{T,V,N}}
  return false
end

function extrapolate!(
  x::V,
  solver::L2PenaltySolver{T,V,S,PB},
  τ₂::T,
  τ₁::T,
) where {T,V,S,N<:AbstractNLPModel{T,V},PB<:L2PenalizedProblem{T,V,N}}

  # Retrieve workspace
  subsolver, substats = solver.subsolver, solver.substats
  ms_solver, ms_stats = subsolver.subsolver, subsolver.substats
  mk = subsolver.subpb
  φ, ψ = mk.model, mk.h
  nlp, h = mk.parent.model, mk.parent.h
  fk, hk = substats.solver_specific[:smooth_obj], substats.solver_specific[:nonsmooth_obj]
  xk, xkn, s, y = subsolver.xk, subsolver.xkn, subsolver.s, subsolver.y
  ∇fk = φ.data.c
  n, m = nlp.meta.nvar, nlp.meta.ncon
  α = ms_stats.solver_specific[:alpha]

  norm_y = norm(y, 2)
  norm_y < τ₁ && return false
  τ₁ = norm_y

  update_workspace!(
    ms_solver.workspace,
    φ.data.H,
    ψ.A,
    φ.data.σ,
    α,
  )

  # [ H + σI Aᵀ][x'] = -[0]
  # [   A    0 ][y'] = -[x] 
  @views @. ms_solver.u2[(n+1):(n+m)] = -ms_solver.x1[(n+1):(n+m)]
  solve_system!(ms_solver.workspace, ms_solver.u2)
  get_solution!(ms_solver.x2, ms_solver.workspace)

  @views px, py = ms_solver.x2[1:n], ms_solver.x2[(n+1):(n+m)]
  α_dot = norm_y/dot(y, py)

  px .*= α_dot
  solver.x .= x .+ (τ₂ - τ₁) * px

  return true
end
