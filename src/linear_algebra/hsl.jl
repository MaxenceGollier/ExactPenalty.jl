struct Ma57_workspace{V}
  M::Ma57
  x::V
  residual::V
  work::V
end

function Ma57_workspace(M::Ma57{T}) where{T}
  x = Vector{T}(undef, M.n)
  residual = Vector{T}(undef, M.n)
  work = Vector{T}(undef, 4*M.n)
  return Ma57_workspace(M, x, residual, work)
end

function construct_ma57_workspace(H::M, u1::V) where{M <: SparseMatrixCOO, V <: AbstractVector}
  return Ma57_workspace(ma57_coord(H.n, H.rows, H.cols, H.vals, sqd = true))
end

function update_workspace!(workspace::Ma57_workspace, H, B::AbstractMatrix, A::AbstractMatrix, σ::T, α::T) where T
  M = workspace.M
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

function update_workspace!(workspace::Ma57_workspace, H, m::Int, α::T) where T
  M = workspace.M
  nnz = length(M.vals)
  M.vals[nnz - m + 1 : nnz] .= -α
end

function solve_system!(workspace::Ma57_workspace, H::M, u::V; kwargs...) where{M <: SparseMatrixCOO, V <: AbstractVector}
  ma57_factorize!(workspace.M)
  ma57_solve!(workspace.M, u, workspace.x, workspace.residual, workspace.work, 10)
end

function get_solution!(x::V, workspace::Ma57_workspace) where{V <: AbstractVector}
  x .= workspace.x
end

function get_status(workspace::Ma57_workspace)
  workspace.M.info.info[1] == 0 && return :success
  return :failed
end