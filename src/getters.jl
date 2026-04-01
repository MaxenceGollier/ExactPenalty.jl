function get_ψ(solver::R2Solver{T}) where {T}
  return solver.ψ
end

function get_ψ(solver::R2NSolver{T}) where {T}
  return solver.subpb.h
end

function get_ψ(solver::L2PenaltySolver{T}) where {T}
  return get_ψ(solver.subsolver)
end

function get_cauchy_sigma(solver::L2PenaltySolver{T}) where {T}
  return isa(solver.subsolver, R2NSolver) ? solver.substats.solver_specific[:sigma_cauchy] : solver.substats.solver_specific[:sigma]
end

function get_cauchy_step(solver::R2Solver{T}) where {T}
  return solver.s
end

function get_cauchy_step(solver::R2NSolver{T}) where {T}
  return solver.s1
end

function get_cauchy_step(solver::L2PenaltySolver{T}) where {T}
  return get_cauchy_step(solver.subsolver)
end