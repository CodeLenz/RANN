# Define função de ativação ReLU
# Rectified Linear Unit
# Retorna o valor de entrada se for positivo e zero se for negativo
function ReLU(x::T)::T where T

    # Retorna o valor máximo entre x e 0
    return max(0.0, x)

end

# Define função de ativação GELU
# Gaussian Error Linear Unit
# Similar ao ReLU, mas com uma transição mais suave entre as partes linear e não linear
# Usando aqui uma aproximação comum
function GELU(x::T)::T where T

    # Retorna o valor de x multiplicado por uma função sigmoidal suave
    return 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))

end

# Define função de ativação Swish
# Interpolação entre função linear e ReLU, com uma transição suave
# f(x) = x / (1 + exp(-βx)), adotando β = 1
function Swish(x::T, β = 1.0)::T where T   

    # Retorna o valor de x multiplicado por uma função sigmoidal suave
    return x / (1 + exp(-β * x))

end