export TRMoreSorensenSparseSolver

mutable struct TRMoreSorensenSparseSolver{T <: Real, V <: AbstractVector{T}, W <: Ma57} <: AbstractPenalizedProblemSolver
  u1::V
  u2::V
  x1::V
  x2::V
  work::V
  residual::V
  workspace::W
end

function TRMoreSorensenSparseSolver(reg_nlp::AbstractRegularizedNLPModel{T, V}; ) where {T, V}
  x0 = reg_nlp.model.meta.x0
  n = reg_nlp.model.meta.nvar
  m = length(reg_nlp.h.b)
  u1 = similar(x0, n+m)
  u2 = zeros(eltype(x0), n+m)
  x1 = similar(u1)
  x2 = similar(u1)
  work = similar(u1, 4*(n+m))
  residual = zeros(u1, n + m)

  H1 = [reg_nlp.model.B+one(T)*I(n) coo_spzeros(T, n, m);]
  H2 = [reg_nlp.h.A (-one(T))*I]
  H = [H1; H2]
  ma_workspace = ma57_coord(n + m, H.rows, H.cols, H.vals, sqd = true)

  return TRMoreSorensenSparseSolver(u1, u2, x1, x2, work, residual, ma_workspace)
end

function SolverCore.solve!( #TODO add verbose and kwargs
  solver::TRMoreSorensenSparseSolver{T, V},
  reg_nlp::AbstractRegularizedNLPModel{T, V},
  stats::GenericExecutionStats{T, V, V};
  x = reg_nlp.model.meta.x0,
  σk = T(1),
  atol = eps(T)^(0.3),
  max_time = T(30),
  max_iter = 100,
) where {T <: Real, V <: AbstractVector{T}}
  start_time = time()
  set_time!(stats, 0.0)
  set_iter!(stats, 0)

  αmin = eps(T)^(0.6)
  α = zero(T)
  θ = 0.8
  n_refin = 20

  # Retrieve workspace
  m, n = size(A)
  Δ = reg_nlp.h.h.lambda

  u1, u2, x1, x2 = solver.u1, solver.u2, solver.x1, solver.x2
  work, residual = solver.work, solver.residual
  M = solver.workspace

  # Create problem
  @. u1[1:n] = -reg_nlp.model.∇f
  @. u1[(n + 1):(n + m)] = -reg_nlp.h.b
  _update_workspace!(M, reg_nlp.model.B, reg_nlp.h.A, reg_nlp.model.σ, α)

  # Check interior convergence
  ma57_factorize!(M)
  ma57_solve!(M, u1, x1, residual, work, n_refin)

  # Check positive definiteness of Q
  n_neg_eigvals = M.info.info[24]
  Q_pos_def = (M.info.rank - n_neg_eigvals == n)
  if !Q_pos_def
    # TODO
  end

  # Check full row rankness of A
  A_full_row_rank = (n_neg_eigvals == m)
  if !A_full_row_rank
    # This is equivalent to performing Golub-Riley iteration method on the Schur complement with regularization equal to αmin
    α = αmin
    _update_workspace!(M, m, α)
    ma57_factorize!(M)
    ma57_solve!(M, u1, x1, residual, work, n_refin)
  end

  # Return interior solution
  if norm(@view x1[n+1:n+m]) <= Δ + atol && norm(residual) <= eps(T)^0.5
    set_solution!(stats, @view x1[1:n])
    return
  end

  # Scalar root finding
  @views @. u2[(n + 1):(n + m)] = -x1[(n + 1):(n + m)]
  ma57_solve!(M, u2, x2, residual, work, n_refin)

  norm_x1 = norm(@view x1[(n + 1):(n + m)])
  while abs(norm_x1 - Δ) > atol && stats.iter < max_iter && stats.elapsed_time < max_time
    @views α₊ = α + norm_x1^2/dot(x1[(n + 1):(n + m)], x2[(n + 1):(n + m)])*(norm_x1 / Δ - 1)
    α = α₊ ≤ 0 ? θ*α : α₊
    α = α ≤ αmin ? αmin : α

    _update_workspace!(M, m, α)

    ma57_factorize!(M)
    ma57_solve!(M, u1, x1, residual, work, n_refin)
    @views @. u2[(n + 1):(n + m)] = -x1[(n + 1):(n + m)]
    ma57_solve!(M, u2, x2, residual, work, n_refin)

    norm_x1 = norm(@view x1[(n + 1):(n + m)])

    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time()-start_time)
    α == αmin && break
  end

  # The algorithm failed to find a root
  if abs(norm_x1 - Δ) > atol
    set_solution!(stats, x)
    isa(reg_nlp.model.B, AbstractQuasiNewtonOperator) && reset!(reg_nlp.model.B) # Reset quasi-Newton approximation for better conditioning in next iterations
    return
  end

  set_solution!(stats, @view x1[1:n])
  return

end

function _update_workspace!(M::Ma57, B::AbstractMatrix, A::AbstractMatrix, σ::T, α::T) where T
  nnz = length(M.vals)
  m, n = size(A)
  Bi, Ai = 1, 1
  @inbounds for i = 1:nnz
    if M.rows[i] <= n && M.cols[i] <= n # B zone
      #Update B
      @assert (M.rows[i] == B.rows[Bi] && M.cols[i] == B.cols[Bi])
      M.vals[i] = B.vals[Bi]
      if M.rows[i] == M.cols[i]
        M.vals[i] += σ
      end
      Bi += 1
    elseif M.rows[i] > n && M.cols[i] <= n
      @assert (M.rows[i] == A.rows[Ai] + n && M.cols[i] == A.cols[Ai]) 
      M.vals[i] = A.vals[Ai]
      Ai += 1
    else
      @assert (M.rows[i] == M.cols[i] && M.rows[i] > n)
      M.vals[i] = -α
    end
  end
end

function _update_workspace!(M::Ma57, m::Int, α::T) where T
  nnz = length(M.vals)
  M.vals[nnz - m + 1 : nnz] .= -α
end
