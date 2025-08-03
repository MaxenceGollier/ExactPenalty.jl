using ExactPenalty
using Documenter

DocMeta.setdocmeta!(ExactPenalty, :DocTestSetup, :(using ExactPenalty); recursive = true)

const page_rename = Dict("developer.md" => "Developer docs") # Without the numbers
const numbered_pages = [
    file for file in readdir(joinpath(@__DIR__, "src")) if
    file != "index.md" && splitext(file)[2] == ".md"
]

makedocs(;
    modules = [ExactPenalty],
    authors = "Maxence Gollier maxence-2.gollier@polymtl.ca",
    repo = "https://github.com/JuliaSmoothOptimizers/ExactPenalty.jl/blob/{commit}{path}#{line}",
    sitename = "ExactPenalty.jl",
    format = Documenter.HTML(; canonical = "https://JuliaSmoothOptimizers.github.io/ExactPenalty.jl"),
    pages = ["index.md"; numbered_pages],
)

deploydocs(; repo = "github.com/JuliaSmoothOptimizers/ExactPenalty.jl")
