# Define a equação diferencial do problema e calcula o valor do resíduo no ponto
function EqDiff(Φ::Vector{Float64}, dΦ::Vector{Vector{Float64}}, dΦ2::Vector{Vector{Float64}},
                x::Vector{Float64})

    # Retorna o valor do resíduo no ponto
    return dΦ2[1][1] + dΦ2[2][1] + 2.0

end

# Calcula a resposta analítica do problema
function Φ_Analitico(prob::String, XY::Vector{Float64})

    # Caso 1: Seção transversal circular
    if prob == "circular"

        # Importa os dados da seção
        R, _ = Geometria_Circular()

        # Calcula a resposta analítica
        Φ_analitico = 0.5 * (R^2 - XY[1]^2 - XY[2]^2)

        return Φ_analitico

    # Caso não seja selecionado nenhum problema, retorna vazio
    else

        return nothing

    end

end