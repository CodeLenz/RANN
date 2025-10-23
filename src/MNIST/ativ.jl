# Define função de ativação ReLU
# Rectified Linear Unit
# Retorna o valor de entrada se for positivo e zero se for negativo
function ReLU(x::T) where T

    # Retorna o valor máximo entre x e 0
    return max(0, x)

end

# Define função de ativação Softmax
# Aplicável para a última camada para classificação dos dados
# Função de densidade de probabilidade, total sempre 1
function Softmax(z::Vector{T}) where T

    # Desconta o valor máximo dos logits para limitar os expoentes e evitar overflow
    z_shift = z .- maximum(z) 

    # Calcula exponencial nos logits e divide pelo total para normalizar
    exps = exp.(z_shift)
    return exps / sum(exps)

end

# Define função objetivo Cross Entropy, já integrada com a função de ativação Softmax
# Aplicável para a última camada para classificação dos dados
# Compara os valores de saída esperados com os valores calculados pela rede
function cross_entropy(z::Vector{T}, y_verdadeiro::Vector{T}) where T

    # Desconta o valor máximo dos sinais para limitar os expoentes e evitar overflow
    z_shift = z .- maximum(z) 

    # Retorna a função objetivo
    return -dot(y_verdadeiro, z_shift) + log(sum(exp.(z_shift)))

end