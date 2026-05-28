abstract type PenaltyWorkspace end
abstract type PenaltyDirectWorkspace    <: PenaltyWorkspace end
abstract type PenaltyIterativeWorkspace <: PenaltyWorkspace end

function construct_workspace(H::M, u1::V, n::Int, m::Int; solver = :minres_qlp) where {M,V}
  if solver == :minres_qlp
    return construct_minres_qlp_workspace(H, u1, n, m)
  elseif solver == :ldlt
    return construct_ldlt_workspace(H, u1, n, m)
  elseif solver == :ma57
    return construct_ma57_workspace(H, u1, n, m)
  end
end

get_n_fact(workspace::PenaltyIterativeWorkspace) = 0
get_n_fact(workspace::PenaltyDirectWorkspace) = workspace._n_fact

function set_n_fact!(workspace::PenaltyIterativeWorkspace, n::Int)
  return
end

function set_n_fact!(workspace::PenaltyDirectWorkspace, n::Int)
  workspace._n_fact = n
end