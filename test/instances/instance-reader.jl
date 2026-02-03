function read_instance(file::String; type = Float64, Hessian_modifier = H -> H)
  lines = readlines(file)
  data = Dict{String, Any}()
  data["name"] = basename(file)

  i = 1
  while i <= length(lines)
    key = strip(lines[i])
    i += 1
    if key == "tau" || key == "sigma"
      data[key] = parse(type, strip(lines[i]))
      i += 1
    elseif key in ["nabla", "c"]
      data[key] = parse.(type, split(strip(lines[i])))
      i += 1
    elseif key in ["J", "Q"]
      matrix = []
      while i <= length(lines) && !(strip(lines[i]) in ["tau", "sigma", "nabla", "c", "J", "Q"])
        row = parse.(type, split(strip(lines[i])))
        push!(matrix, row)
        i += 1
      end
      data[key] = reduce(vcat, [reshape(r, 1, :) for r in matrix])

    else
      error("Unknown section: $key")
    end
  end
  x0 = zeros(type, length(data["nabla"]))
  model = R2NModel(Hessian_modifier((data["Q"] + data["Q"]')/2), data["nabla"], data["sigma"], x0)

  c! = let c = data["c"] 
    (b, x) -> 
    begin
      b .= c  
    end
  end
  J! = let J = data["J"] 
    (A, x) -> 
    begin
      A .= J  
    end
  end
  h = ShiftedCompositeNormL2(data["tau"], c!, J!, SparseMatrixCOO(data["J"]), data["c"])
  return RegularizedNLPModel(model, h)
end