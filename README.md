# ExactPenalty

[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaSmoothOptimizers.github.io/ExactPenalty.jl/stable)
[![Development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaSmoothOptimizers.github.io/ExactPenalty.jl/dev)
[![Test workflow status](https://github.com/JuliaSmoothOptimizers/ExactPenalty.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/JuliaSmoothOptimizers/ExactPenalty.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaSmoothOptimizers/ExactPenalty.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaSmoothOptimizers/ExactPenalty.jl)
[![Docs workflow Status](https://github.com/JuliaSmoothOptimizers/ExactPenalty.jl/actions/workflows/Docs.yml/badge.svg?branch=main)](https://github.com/JuliaSmoothOptimizers/ExactPenalty.jl/actions/workflows/Docs.yml?query=branch%3Amain)
[![Build Status](https://api.cirrus-ci.com/github/JuliaSmoothOptimizers/ExactPenalty.jl.svg)](https://cirrus-ci.com/github/JuliaSmoothOptimizers/ExactPenalty.jl)
[![DOI](https://zenodo.org/badge/DOI/FIXME)](https://doi.org/FIXME)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![All Contributors](https://img.shields.io/github/all-contributors/JuliaSmoothOptimizers/ExactPenalty.jl?labelColor=5e1ec7&color=c0ffee&style=flat-square)](#contributors)
[![BestieTemplate](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/JuliaBesties/BestieTemplate.jl/main/docs/src/assets/badge.json)](https://github.com/JuliaBesties/BestieTemplate.jl)

ExactPenalty is a solver for the smooth, constrained, nonlinear problem,
```math
    \min f(x) \quad \text{s.t.} \ c(x) = 0.
```
The above is solved by computing an approximate solution of the non-smooth penalized problem
```math
    \min f(x) + \tau \| c(x) \|,
```
for some penalty parameter $$\tau > 0$$ and some norm $$\|\cdot\|$$.

## How to Cite

If you use ExactPenalty.jl in your work, please cite using the reference given in [CITATION.cff](https://github.com/JuliaSmoothOptimizers/ExactPenalty.jl/blob/main/CITATION.cff).

## Contributing

If you want to make contributions of any kind, please first that a look into our [contributing guide directly on GitHub](docs/src/90-contributing.md) or the [contributing page on the website](https://JuliaSmoothOptimizers.github.io/ExactPenalty.jl/dev/90-contributing/)

---

### Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
