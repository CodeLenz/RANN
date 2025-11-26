# Define a equação diferencial do problema e calcula o valor do resíduo no ponto
function EqDiff(Φ::Vector{Float64}, dΦ::Vector{Vector{Float64}}, dΦ2::Vector{Vector{Float64}},
                x::Vector{Float64})

    # Retorna o valor do resíduo no ponto
    return dΦ2[1][1] + dΦ2[2][2] + 2.0

end