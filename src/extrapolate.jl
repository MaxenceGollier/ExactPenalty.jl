function extrapolate!(x::V, solver::L2PenaltySolver{T, V}, τ₂::T, τ₁::T) where{T, V}

  c = solver.subsolver.ψ.b
  y = solver.subpb.y
  norm_c = norm(c)

  # Implicitely update the Hessian of the subproblem
  y .=  (τ₁/norm_c).*c 

  # Prepare the linear solver
  linear_solver = solver.subsolver.subsolver.workspace
  u1, x1, x2 = solver.subsolver.subsolver.u1, solver.subsolver.subsolver.x1, solver.subsolver.subsolver.x2
  m, n = size(solver.subsolver.subpb.h.A)

  update_workspace!(linear_solver, solver.subsolver.subpb.model.B, solver.subsolver.subpb.h.A, zero(T), norm_c/τ₁)

  @views u1[1:n] .= 0
  @views u1[(n + 1):(n + m)] .= c./(-τ₁)

  # [ H     Aᵀ    ][x] = -[0]
  # [ A   -‖c‖/τI ][y] = -[c/τ] 
  solve_system!(linear_solver, u1)
  get_solution!(x1, linear_solver)
  status = get_status(linear_solver)

  status != :success && return
    # TODO print warning

  # [ H     Aᵀ    ][x] = -[0]
  # [ A   -‖c‖/τI ][y] = -[c/√(τ*norm_c)]  
  @views u1[(n + 1):(n + m)] .= c./(-sqrt(norm_c*τ₁)) 
  solve_system!(linear_solver, u1)
  get_solution!(x2, linear_solver)
  status = get_status(linear_solver)

  status != :success && return
    # TODO print warning

  # u = √(τ/norm_c)*J(x)ᵀc/norm_c = J(x)ᵀy/√(τ*norm_c)
  @views mul!(u1[1:n], solver.subsolver.subpb.h.A', y, 1/sqrt(τ₁*norm_c), zero(T))

  dx_dτ = @view x1[1:n]

  @views dx_dτ .-= x2[1:n].*(dot(x1[1:n], u1[1:n])/(1 + dot(x2[1:n], u1[1:n])))

  x .= solver.x .+ dx_dτ.*(τ₂ - τ₁)

  return x
end