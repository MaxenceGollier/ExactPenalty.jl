using DataFrames
using JLD2
using Measures
using Printf
using SolverBenchmark
using Plots

const METHODS = (:exact, :lbfgs, :r2)

function load_stats(dir::AbstractString, stats, suffix = "")

  for method in METHODS
    file = joinpath(dir, "stats_$(method).jld2")

    if !isfile(file)
      error("Missing file: $file")
    end

    @info "Loading $file"
    dict = load(file)["stats"]
    for key in keys(dict)
      new_key = Symbol("$(key)$suffix")
      stats[new_key] = dict[key]
    end
  end

  return stats
end

function pairwise_plot(stats, keys)
  solved(df) = (df.status .== :first_order) # TODO: add infeasible problems
  costs = [df -> .!solved(df) * Inf + df.elapsed_time, 
            df -> .!solved(df) * Inf + df.neval_obj, df -> .!solved(df) * Inf + df.neval_grad]
  costnames = ["CPU Time", "# Objective Evals", "# Gradient Evals"]

  stats_subset = filter(kv -> kv[1] in keys, stats)

  # split the stat names. the first part is always l2penalty, then the method used, then the tolerance and finally the branch.
  parts_1 = Symbol.(split(string(keys[1]), "_"))
  parts_2 = Symbol.(split(string(keys[2]), "_"))
  @assert parts_1[3] == parts_2[3]

  models = Dict(:exact => "∇²f(x)", :lbfgs => "BFGS", :r2 => "σI")
  precision = Dict(:imprecise => "1e-3", :precise => "1e-9")

  suptitle = "\nHessian model: Bₖ(x) = " * models[parts_1[2]] * "; Tolerance: ϵ = " * precision[parts_1[3]]
  p = profile_solvers(stats_subset, costs, costnames; suptitle = suptitle, xlabel = "", ylabel = "")
  p.subplots[2][:legend_position] = :bottomright
  p.subplots[3][:legend_position] = :bottomright  
  p.series_list[1][:label] = Symbol.(split(string(p.series_list[1][:label]), "_"))[end] == :current ? :current : :main
  p.series_list[2][:label] = Symbol.(split(string(p.series_list[2][:label]), "_"))[end] == :current ? :current : :main
  p.series_list[3][:label] = Symbol.(split(string(p.series_list[3][:label]), "_"))[end] == :current ? :current : :main
  p.series_list[4][:label] = Symbol.(split(string(p.series_list[4][:label]), "_"))[end] == :current ? :current : :main
  p.series_list[5][:label] = Symbol.(split(string(p.series_list[5][:label]), "_"))[end] == :current ? :current : :main
  p.series_list[6][:label] = Symbol.(split(string(p.series_list[6][:label]), "_"))[end] == :current ? :current : :main

  p.series_list[1][:linecolor] = Symbol.(split(string(p.series_list[1][:label]), "_"))[end] == :current ? :blue : :red
  p.series_list[2][:linecolor] = Symbol.(split(string(p.series_list[2][:label]), "_"))[end] == :current ? :blue : :red
  p.series_list[3][:linecolor] = Symbol.(split(string(p.series_list[3][:label]), "_"))[end] == :current ? :blue : :red
  p.series_list[4][:linecolor] = Symbol.(split(string(p.series_list[4][:label]), "_"))[end] == :current ? :blue : :red
  p.series_list[5][:linecolor] = Symbol.(split(string(p.series_list[5][:label]), "_"))[end] == :current ? :blue : :red
  p.series_list[6][:linecolor] = Symbol.(split(string(p.series_list[6][:label]), "_"))[end] == :current ? :blue : :red

  return p
end

current_dir   = joinpath("artifacts", "current", "cutest-benchmark-stats")
reference_dir = joinpath("artifacts", "reference", "cutest-benchmark-stats")

stats = Dict{Symbol, DataFrame}()

@info "Loading current benchmark results"
load_stats(current_dir, stats, "_current")

@info "Loading reference benchmark results"
load_stats(reference_dir, stats, "_reference")

p = plot(
  pairwise_plot(stats, [:l2penalty_exact_imprecise_reference, :l2penalty_exact_imprecise_current]),
  pairwise_plot(stats, [:l2penalty_exact_precise_reference, :l2penalty_exact_precise_current]),
  pairwise_plot(stats, [:l2penalty_lbfgs_imprecise_reference, :l2penalty_lbfgs_imprecise_current]),
  pairwise_plot(stats, [:l2penalty_lbfgs_precise_reference, :l2penalty_lbfgs_precise_current]),
  pairwise_plot(stats, [:l2penalty_r2_imprecise_reference, :l2penalty_r2_imprecise_current]),
  pairwise_plot(stats, [:l2penalty_r2_precise_reference, :l2penalty_r2_precise_current]),
  layout = (3, 2), size = (1920, 1080))

savefig(p, "benchmark/result/benchmark_comparison.svg")