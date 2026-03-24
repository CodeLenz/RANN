# Define a equação diferencial do problema e calcula o valor do resíduo no ponto
function EqDiff(Φ::Vector{Float64}, dΦ::Vector{Vector{Float64}}, dΦ2::Vector{Vector{Float64}},
                x::Vector{Float64})

    # Retorna o valor do resíduo no ponto
    return dΦ2[1][1] + dΦ2[2][1] + 2.0

end

# Calcula a resposta analítica do problema
function Φ_Analitico(prob::String, XY::Vector{Float64})

    # Para facilitar 
    x = XY[1]
    y = XY[2]

    # Caso 1: Seção transversal circular
    if prob == "Circular"

        # Importa os dados da seção
        R, a, b, _ = Geometria_Circular()

        # Calcula a resposta analítica
        Φ_analitico = 0.5 * (R^2 - (x - a)^2 - (y - b)^2)

        # Retorna resposta
        return Φ_analitico

    elseif prob == "Retangular"

        # Importa os dados da seção
        H, B, a, b, _ = Geometria_Retangular()

        # Loop por N termos da série, a princípio deixaremos 20
        N = 20

        Φ = 0.0
        for ki in 0:(N-1)

            k = 2 * ki + 1

            for li in 0:(N-1)

                    l = 2 * li + 1

                    num = (-1)^( ( (k + l) ÷ 2 ) - 1 )

                    Φ += ( num / (k * l * (k^2 * H^2 + l^2 * B^2)) ) * cos(k * π * x / B) * cos(l * π * y / H)

            end

        end
        
        # Solução analítica
        Φ_analitico = Φ * (32 * B^2 * H^2 / π^4)

        # Retorna resposta
        return Φ_analitico

    # Caso não seja selecionado nenhum problema, retorna vazio
    else

        return nothing

    end

end