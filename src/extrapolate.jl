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

  φ, ψ = solver.subsolver.subpb.model, solver.subsolver.subpb.h
  nlp, h = solver.subsolver.subpb.parent.model, solver.subsolver.subpb.parent.h

  c, norm_c = ψ.b, norm(ψ.b)

  # Implicitely update the Hessian of the subproblem
  y = solver.y .= (τ₁/norm_c) .* c

  # Prepare the linear solver
  linear_solver = solver.subsolver.subsolver.workspace
  u1, x1, x2 = solver.subsolver.subsolver.u1,
  solver.subsolver.subsolver.x1,
  solver.subsolver.subsolver.x2
  m, n = size(ψ.A)

  update_workspace!(linear_solver, φ.data.H, ψ.A, zero(T), norm_c/τ₁)

  @views u1[1:n] .= 0
  @views u1[(n+1):(n+m)] .= c ./ (-τ₁)

  # [ H     Aᵀ    ][x] = -[0]
  # [ A   -‖c‖/τI ][y] = -[c/τ] 
  solve_system!(linear_solver, u1)
  get_solution!(x1, linear_solver)
  status = get_status(linear_solver)
  npos, nzero, nneg = get_inertia(linear_solver)
  check_inertia = npos == n && nzero == 0 && nneg == m

  (status != :success || !check_inertia) && return false
  # TODO print warning

  # [ H     Aᵀ    ][x] = -[0]
  # [ A   -‖c‖/τI ][y] = -[c/√(τ*norm_c)]  
  @views u1[(n+1):(n+m)] .= c ./ (-sqrt(norm_c*τ₁))
  solve_system!(linear_solver, u1)
  get_solution!(x2, linear_solver)
  status = get_status(linear_solver)

  status != :success && return false
  # TODO print warning

  # u = √(τ/norm_c)*J(x)ᵀc/norm_c = J(x)ᵀy/√(τ*norm_c)
  @views mul!(u1[1:n], ψ.A', y, 1/sqrt(τ₁*norm_c), zero(T))

  dx_dτ = @view x1[1:n]

  @views dx_dτ .-= x2[1:n] .* (dot(x1[1:n], u1[1:n])/(1 + dot(x2[1:n], u1[1:n])))
  if (τ₂ - τ₁) * norm(dx_dτ)/norm(x) < 1
    x .= solver.x .+ dx_dτ .* (τ₂ - τ₁)
    return true
  else
    return false
  end

end
