mutable struct watchdog_checkpoint{T, V}
  xk::V
  ∇fk::V
  ck::V
  yk::V
  Jkvals::V
  Hkvals::V
  active::Bool
  iter::Int
  primal_feas::T
  dual_feas::T
  σk::T
end

function watchdog_checkpoint(nlp::ShiftedL2PenalizedProblem{T, V}) where{T, V}
  φ, ψ = nlp.model, nlp.h
  ∇f_model, b_model = φ.data.c, ψ.b
  xk, ∇fk = similar(∇f_model), similar(∇f_model)
  ck, yk = similar(b_model), similar(b_model)
  Jkvals = similar(ψ.A.vals)
  Hkvals = similar(φ.data.H.vals)
  return watchdog_checkpoint(
    xk,
    ∇fk,
    ck,
    yk,
    Jkvals,
    Hkvals,
    false,
    0,
    zero(T),
    zero(T),
    zero(T)
  )
end

function save!(checkpoint::watchdog_checkpoint, nlp::ShiftedL2PenalizedProblem, x, y, stats)
  φ, ψ = nlp.model, nlp.h
  
  checkpoint.xk .= x
  checkpoint.yk .= y
  checkpoint.∇fk .= φ.data.c
  checkpoint.Hkvals .= φ.data.H.vals
  checkpoint.σk = φ.data.σ
  checkpoint.ck .= ψ.b
  checkpoint.Jkvals .= ψ.A.vals
  checkpoint.primal_feas = stats.primal_feas
  checkpoint.dual_feas = stats.dual_feas
  checkpoint.iter = stats.iter
end

function fallback!(nlp::ShiftedL2PenalizedProblem, x, y, checkpoint::watchdog_checkpoint)
  φ, ψ = nlp.model, nlp.h

  x .= checkpoint.xk
  y .= checkpoint.yk
  φ.data.c .= checkpoint.∇fk
  φ.data.H.vals .= checkpoint.Hkvals
  φ.data.σ = checkpoint.σk
  ψ.b .= checkpoint.b
  ψ.A.vals .= checkpoint.Jkvals
end

function activate!(checkpoint::watchdog_checkpoint)
  checkpoint.active = true
end

function deactivate!(checkpoint::watchdog_checkpoint)
  checkpoint.active = false
end

is_active(checkpoint::watchdog_checkpoint) = checkpoint.active

function check_watchdog!(checkpoint::watchdog_checkpoint, stats)
  fail = (is_active(checkpoint) && (stats.iter - checkpoint.iter > 10) && 
            (stats.dual_feas > checkpoint.dual_feas || 
            stats.primal_feas > checkpoint.primal_feas))
  if !fail && stats.iter - checkpoint.iter <= 10
    deactivate!(checkpoint)
  end
  return fail
end