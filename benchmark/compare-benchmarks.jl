using JLD2
using Printf

const METHODS = ("exact", "lbfgs", "r2")

function load_stats(dir::AbstractString)
  stats = Dict{String, Any}()

  for method in METHODS
    file = joinpath(dir, "stats_$(method).jld2")

    if !isfile(file)
      error("Missing file: $file")
    end

    @info "Loading $file"
    stats[method] = load(file)
  end

  return stats
end

current_dir   = joinpath("artifacts", "current", "cutest-benchmark-stats")
reference_dir = joinpath("artifacts", "reference", "cutest-benchmark-stats")

@info "Loading current benchmark results"
current = load_stats(current_dir)

@info "Loading reference benchmark results"
reference = load_stats(reference_dir)