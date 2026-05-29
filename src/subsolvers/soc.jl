function second_order_correction!(
  solver::MoreSorensenSolver{T,V},
  reg_nlp::ShiftedL2PenalizedProblem{T,V,M,H,P},
  stats::GenericExecutionStats{T,V,V},
) where {T,V,M,H,P}
  n = reg_nlp.model.meta.nvar
  m = length(reg_nlp.h.b)

  @. solver.u1[1:n]         = -reg_nlp.model.data.c
  @. solver.u1[(n+1):(n+m)] = - reg_nlp.h.b - reg_nlp.parent.h.b

  solve_system!(solver.workspace, solver.u1)

  status = get_status(solver.workspace)
  status == :success ? set_status!(stats, :first_order) : set_status!(stats, :exception)
end