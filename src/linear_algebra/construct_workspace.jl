function construct_workspace(H::M, u1::V, n::Int, m::Int; solver = :minres_qlp) where{M, V}
  if solver == :minres_qlp
    return construct_minres_qlp_workspace(H, u1, n, m)
  elseif solver == :ldlt
    return construct_ldlt_workspace(H, u1, n, m)
  end
end