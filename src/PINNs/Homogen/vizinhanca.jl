# Calcula as coordenadas da vizinhança de um ponto
# Centro, Leste, Oeste, Norte, Sul,
# Nordeste, Noroeste, Sudeste, Sudoeste
function Coord_Vizinhas(y::Vector{T}) where T

    # Extraí as componentes
    y1 = y[1]
    y2 = y[2]

    # Retorna a matriz de coordenadas da vizinhança do ponto
    y_viz = [[y1   y2  ], # Centro
             [y1+h y2  ], # Leste
             [y1-h y2  ], # Oeste
             [y1   y2+h], # Norte
             [y1   y2-h], # Sul
             [y1+h y2+h], # Nordeste
             [y1-h y2+h], # Noroeste
             [y1+h y2-h], # Sudeste
             [y1-h y2-h]] # Sudoeste

end

# Calcula o valor do deslocamento flutuante em todos
# os pontos da vizinhança
function Deslocamento_Viz(rede::Rede, pesos::Vector{<:AbstractMatrix{Float64}}, bias::Vector{<:AbstractVector{Float64}}, 
                          y_viz::Matrix{T}, prob::String)::Vector{T} where T

    # Número de pontos na vizinhança
    n_viz = size(y_viz, 1)

    # Aloca vetor para os deslocamentos flutuantes
    u_viz = Vector{T}(undef, n_viz)

    # Loop por cada ponto da vizinhança
    for i in 1:n_viz

        # Calcula o deslocamento flutuante no ponto i da vizinhança
        u_viz[i] = RNA_Fourier(rede, pesos, bias, y_viz[i], prob)

    end

    return u_viz

end
