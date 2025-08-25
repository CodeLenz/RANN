# Função objetivo que depende das entradas da rede neural
# R^n -> R, onde n é o número total de pesos e bias da rede 
function Objetivo(rede::Rede, entradas::Matrix{Float64}, saidas_esperadas::Matrix{Float64}, 
                  n::Int64, x::Vector{T}) where T

    # Aloca as matrizes de pesos e bias a partir das variáveis de projeto
    pesos, bias = Atualiza_pesos_bias(rede, x)

    # Define o erro da rede
    perda = zero(T)

    # Loop pelos treinos
    for coluna = 1:n
 
        # Extraí as entradas da rede
        entrada_i = entradas[:, coluna]

        # Extraí as saídas esperadas
        saida_esperada_i = saidas_esperadas[:, coluna]

        # Roda a rede (forward loop)
        logits_saida_i = RNA(rede, pesos, bias, entrada_i)

        # Computa a função objetivo cross entropy e soma na perda total
        perda += cross_entropy(logits_saida_i, saida_esperada_i)

    end
    
    # Retorna a perda média da rede
    return perda / n

end
