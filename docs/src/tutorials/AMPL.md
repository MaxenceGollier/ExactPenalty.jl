# AMPL tutorial

This tutorial shows how to solve an AMPL model with `ExactPenalty.jl` using
`AmplNLPReader.jl` (`AmplNLReader.jl` package name).

## 1. Create an `.nl` file with AMPL

Assume your AMPL model is in `mymodel.mod` and `mymodel.dat`.
Generate the `.nl` file from AMPL:

```bash
ampl -ogmymodel mymodel.mod mymodel.dat
```

This command creates `mymodel.nl`.

## 2. Read the AMPL model in Julia

```julia
using AmplNLReader

nlp = AmplModel("mymodel.nl")
```

`nlp` is an `NLPModel`, so it can be passed directly to `ExactPenalty.jl`.

## 3. Solve with ExactPenalty

```julia
using ExactPenalty

stats = L2Penalty(nlp; print_level = 1)

println("status: ", stats.status)
println("objective: ", stats.objective)
println("solution: ", stats.solution)
```

## 4. (Optional) Write a `.sol` file for AMPL

`AmplNLReader.jl` exposes `write_sol` to export primal-dual solutions:

```julia
using NLPModels

write_sol(
  nlp,
  string("Solved with status ", stats.status),
  stats.solution,
  stats.multipliers,
)
```

This writes a `.sol` file that AMPL can read.

## Notes

- If your model includes inequality constraints, convert them to equalities
  (for example with slack variables) before solving with `L2Penalty`.
- For advanced solver options, see the [Options Reference](options.md).