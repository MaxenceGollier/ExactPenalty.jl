function construct_workspace(H::M, u1::V; solver = :minres_qlp) where{M, V}
  if solver == :minres_qlp
    @assert isa(H, AbstractLinearOperator)
    return construct_minres_qlp_workspace(H, u1)
  elseif solver == :ma57
    @assert isa(H, SparseMatrixCOO)
    return construct_ma57_workspace(H, u1)
  end
end