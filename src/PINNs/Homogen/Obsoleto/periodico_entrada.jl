#
# Implementação da camada periódica com N modos de Fourier
# Transforma as coordenadas espaciais (y1, y2) da célula unitária
# [0,1]x[0,1]  em entradas periodicas para a rede, como se 
# fosse uma pré-entrada. A medida que aumentamos k (1..N)
# temos senos e cossenos com periodicidade [0,1] e frequências
# angulares 2π * k cada vez maiores. 
#
function Camada_Fourier(y::Vector{T}, N::Int) where T

    # Extrai as componentes
    y1 = y[1]
    y2 = y[2]

    #  Aloca um vetor 4N
    saida = Vector{T}(undef, 4 * N)

    # Loop por cada um dos modos 
    for k in 1:N

        # Ondas no sentido y1 (horizontal)
        saida[4 * (k - 1) + 1] = sin(2π * k * y1)
        saida[4 * (k - 1) + 2] = cos(2π * k * y1)

        # Ondas no sentido y2 (vertical)
        saida[4 * (k - 1) + 3] = sin(2π * k * y2)
        saida[4 * (k - 1) + 4] = cos(2π * k * y2)

    end

    # Retorna o vetor de 4N posições para ser 
    # usado como entrada da rede
    return saida
    
end