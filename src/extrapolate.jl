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

  # Step 1. If the multipliers are smaller than the penalty parameter, the derivative w.r.t τ₁ is equal to 0.
  norm(y) < τ₁ && return false

  # Step 2. Compute multipliers that are such that ‖y‖ < τ₁
  φ.data.σ = substats.solver_specific[:sigma]
  solve!(ms_solver, mk, ms_stats; accept_descent = false)
  get_primal_dual_sol!(s, y, ms_solver)

  ms_stats.status != :first_order && return false

  # Step 3. Shift the model to the multipliers, check with an Armijo-condition
  xkn .= xk .+ s
  fkn, hkn = obj(nlp, xkn), h(xkn)
  mks = dot(∇fk, s) + ψ(s)

  Δobj = fk + hk - (fkn + hkn) + max(1, abs(fk + hk)) * 10 * eps()
  Δmod = fk + hk - (fk + mks) + max(1, abs(fk + hk)) * 10 * eps()

  ρk = Δmod < 0 ? 0 : Δobj / Δmod
  ρk <= 0 && return false

  set_solver_specific!(substats, :smooth_obj, fk)
  set_solver_specific!(substats, :nonsmooth_obj, hk)
  xk .= xkn
  shift!(mk, xk, y = y)

  # Step 3.5. If norm(y) < τ₁ the derivative is still 0
  norm(y) < τ₁ && return false

  # Step 4. Construct u = [0  y]
  @. ms_solver.u1[1:n] = 0
  @. ms_solver.u1[(n+1):(n+m)] = y

  update_workspace!(
    ms_solver.workspace,
    φ.data.H,
    ψ.A,
    φ.data.σ,
    ms_stats.solver_specific[:alpha],
  )

  # Step 5. Solve
  # [ H  Jᵀ ][z] = [u]
  # [ J -αI ][z] = [u]
  solve_system!(ms_solver.workspace, ms_solver.u1)
  get_solution!(ms_solver.x1, ms_solver.workspace)
  status = get_status(ms_solver.workspace)
  npos, nzero, nneg = get_inertia(ms_solver.workspace)
  check_inertia = npos == n && nzero == 0 && nneg == m

  (status != :success || !check_inertia) && return false

  # Step 6. Compute
  # d[x]/dτ = τ₁[z]/uᵀz
  # d[y]/dτ = τ₁[z]/uᵀz
  u1x1 = dot(ms_solver.u1, ms_solver.x1)
  abs(u1x1) < (norm(ms_solver.u1)*norm(ms_solver.x1)) * sqrt(eps(T)) && return false
  scale = τ₁ / dot(ms_solver.u1, ms_solver.x1)
  ms_solver.x1 .*= scale

  # Step 7. Update
  solver.x .= xk .+ ms_solver.x1[1:n] .* (τ₂ - τ₁)
  solver.y .= y .+ ms_solver.x1[n+1:end] .* (τ₂ - τ₁)
  norm(solver.y) > τ₂ && (solver.y .*= τ₂ / norm(solver.y))
  
  shift!(mk, solver.x, y = solver.y)
  set_solver_specific!(substats, :smooth_obj, obj(nlp, solver.x))

  return true
end
