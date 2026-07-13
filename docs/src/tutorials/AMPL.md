# [AMPL tutorial](@id ampl-tutorial)

This tutorial shows how to solve a model written in [AMPL](https://ampl.com)
with `ExactPenalty.jl`, using
[AmplNLReader.jl](https://github.com/JuliaSmoothOptimizers/AmplNLReader.jl).

!!! warning "Inequality Constraints"
    `ExactPenalty.jl` solves problems of the form `minimize f(x) s.t. c(x) = 0`.
    If your AMPL model has inequality constraints, the solver will fail.

## 1. The AMPL model

We use the Hock–Schittkowski problem 6 (HS6), a small equality-constrained
problem.

```ampl
var x1 := -1.2;
var x2 := 1;

minimize obj: (1 - x1)^2;

subject to c1: 10 * (x2 - x1^2) = 0;
```

We save this in a `hs6.mod` file.

## 2. Generate the `.nl` file

AMPL compiles a model into a `.nl` file that solvers
read directly, without needing AMPL itself at solve time:

```console
$ ampl -oghs6 assets/hs6.mod
```

This produces file `hs6.nl`.

## 3. Read the model into Julia

```@example ampl
using AmplNLReader

nlp = AmplModel(joinpath(@__DIR__, "assets", "hs6.nl"))

nothing # hide
```

## 4. Solve with ExactPenalty

```@example ampl
using ExactPenalty

stats = L2Penalty(nlp; print_level = 1)

nothing # hide
```

```@example ampl
println("status    : ", stats.status)
println("objective : ", stats.objective)
println("solution  : ", stats.solution)
```

See the [options reference](../options.md) for the full list of keyword
arguments accepted by the solver.

## 5. Write a `.sol` file for AMPL (optional)

If you started from AMPL and want to hand the solution back to it,
`AmplNLReader.jl` exposes `write_sol` to export the primal-dual solution:

```@example ampl
write_sol(
  nlp,
  string("Solved with status ", stats.status),
  stats.solution,
  stats.multipliers,
)
```

AMPL can then read the resulting `.sol` file to recover the solution inside
your original model.