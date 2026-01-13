export TRTruncatedCGSparseSolver, solve!

mutable struct TRTruncatedCGSparseSolver{T <: Real, V <: AbstractVector{T}, W <: Ma57} <: AbstractPenalizedProblemSolver
  u1::V
  u2::V
  x1::V
  x2::V
  g::V
  work::V
  residual::V
  workspace::W
end

function TRTruncatedCGSparseSolver(reg_nlp::AbstractRegularizedNLPModel{T, V}; ) where {T, V}
  x0 = reg_nlp.model.meta.x0
  n = reg_nlp.model.meta.nvar
  m = length(reg_nlp.h.b)
  u1 = similar(x0, n)
  u2 = zeros(eltype(x0), m)
  g = similar(x0, m)
  x1 = similar(u1)
  x2 = similar(u1)

  work = similar(u1, 4*n)
  residual = similar(u1)
  H = SparseMatrixCOO(Matrix(reg_nlp.model.B)+one(T)*I(n)) # FIXME: Highly inefficient but sparsematricesCOO does not work for some reason......... 

  ma_workspace = ma57_coord(n, H.rows, H.cols, H.vals)

  return TRTruncatedCGSparseSolver(u1, u2, x1, x2, g, work, residual, ma_workspace)
end

function SolverCore.solve!( #TODO add verbose and kwargs
  solver::TRTruncatedCGSparseSolver{T, V},
  reg_nlp::AbstractRegularizedNLPModel{T, V},
  stats::GenericExecutionStats{T, V, V};
  x = reg_nlp.model.meta.x0,
  σk = T(1),
  atol = eps(T)^(0.6),
  max_time = T(30),
  max_iter = 100,
) where {T <: Real, V <: AbstractVector{T}}
println("----------------------------")
  atol = eps(T)^0.6
  start_time = time()
  set_time!(stats, 0.0)
  set_iter!(stats, 0)

  m,n = size(reg_nlp.h.A)

  M = solver.workspace
  resid = solver.residual
  work = solver.work
  n_refin = 5
  Δ = reg_nlp.h.h.lambda

  _update_workspace!(M, reg_nlp.model.B, reg_nlp.model.σ)
  ma57_factorize!(M)

  # Compute LHS
  g = solver.g # g = AQ^{-1}d + b
  g .= reg_nlp.h.b
  ma57_solve!(M, reg_nlp.model.∇f, solver.u1, resid, work, n_refin)
  mul!(g, reg_nlp.h.A, solver.u1, -one(T), one(T))

  # Compute residual
  residual = zeros(T, m)
  @. residual = -g

  yk = zeros(T, m)
  ykn = zeros(T, m)
  dk = zeros(T, m)

  dk .= residual

  u0 = zeros(T, n)
  u1 = zeros(T, n)
  u2 = zeros(T, m)

  atol += norm(residual)*atol

  while norm(residual) > eps(T)^0.6 && stats.iter < max_iter && stats.elapsed_time < max_time
    apply_H!(u2, dk, reg_nlp.h.A, M, u1, u0, resid, work, n_refin) # H dk
    dHd = dot(dk, u2)

    if dHd <= 0
      α = _find_tau_to_boundary(yk, dk, Δ)
      @. yk = yk + α*dk
      println("Converged on negative curvature with curvature $dHd to $(norm(yk))")
      println(reg_nlp.model.σ)
      set_status!(stats, :unbounded)
      return
    end

    α = dot(residual, residual)/dHd

    @. ykn = yk + α*dk

    if norm(ykn) >= Δ
      α = _find_tau_to_boundary(yk, dk, Δ)
      @. yk = yk + α*dk
      println("Converged on boundary with radius $Δ to $(norm(yk))")
      break
    end

    yk .= ykn

    rTr = dot(residual, residual)
    @. residual = residual - α*u2

    β = dot(residual, residual)/rTr
    @. dk = dk + β*residual

    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time()-start_time)
  end

  # Compute primal solution Q^{-1}(d+A^T y)
  x1 = zeros(T, n)
  x2 = zeros(T, n)

  x2 .= reg_nlp.model.∇f
  mul!(x2, reg_nlp.h.A', yk, one(T), -one(T))
  ma57_solve!(M, x2, x1, resid, work, n_refin)

  (norm(residual) < eps(T)^0.6) && println("Converged on residual with $(norm(residual))")

  set_solution!(stats, x1)

  if Δ*norm(reg_nlp.h.b) - obj(reg_nlp.model, x1; skip_sigma = true) - reg_nlp.h(x1)  < 0
    println("reverting to cauchy point...")
    set_solution!(stats, x)
  else
    println("it worked at least once")
    error("done")
  end

end

function _update_workspace!(M::Ma57, B::AbstractMatrix, σ::Real)
  nnz = length(M.vals)
  n = size(B, 1)
  @inbounds for i = 1:nnz
    #Update B
    M.vals[i] = B[M.rows[i], M.cols[i]]
    if M.rows[i] == M.cols[i]
      M.vals[i] += σ
    end
  end
end

function apply_H!(z::V, y::V, A::M1, Q::M2, x1::V, x2::V, resid::V, work::V, n_refin::Int) where{V, M1, M2 <: Ma57}
  mul!(x1, A', y)
  ma57_solve!(Q, x1, x2, resid, work, n_refin)
  mul!(z, A, x2)
end

"""
    find_tau_to_boundary(p, d, Δ)

Compute the step length τ ∈ (0, 1] such that ‖p + τ d‖ = Δ.

Used in truncated CG or Steihaug–Toint trust-region methods.
"""
function _find_tau_to_boundary(p::V, d::V, Δ::T) where{T, V}
    a = dot(d, d)
    b = dot(p, d)
    c = dot(p, p) - Δ^2

    if a <= 0
        return 0.0
    end

    disc = max(b^2 - a * c, zero(T))

    τ = (-b + sqrt(disc)) / a
    return τ
end