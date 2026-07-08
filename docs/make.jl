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
  repo = "https://github.com/MaxenceGollier/ExactPenalty.jl/blob/{commit}{path}#{line}",
  sitename = "ExactPenalty.jl",
  format = Documenter.HTML(;
    canonical = "https://MaxenceGollier.github.io/ExactPenalty.jl",
  ),
  pages = [
    "Home" => "index.md",
    "Options" => "options.md",
    "Callbacks" => "callbacks.md",
    "Tutorials" => [
      "AMPL" => "tutorials/AMPL.md",
      "CUTEst" => "tutorials/CUTEst.md",
    ],
    "Developers" => [
      "Contributing" => "90-contributing.md",
      "Developing" => "91-developer.md",
    ]
  ],
)

deploydocs(; repo = "github.com/MaxenceGollier/ExactPenalty.jl")
