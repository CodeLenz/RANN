# Define função de ativação ReLU
# Rectified Linear Unit
# Retorna o valor de entrada se for positivo e zero se for negativo
function ReLU(x)

    # Retorna o valor máximo entre x e 0
    return max(0.0, x)

end

# Define a função de perda para a condição de contorno essencial/natural
function Fn_perda_contorno(y_pred::Vector{Float64}, y_esperado::Vector{Float64})

    return abs(y_pred[1]-y_esperado[1])

end

# Define a função de perda física para os pontos de treino
function Fn_perda_fisica(treino::Treino, u::Vector{Float64}, du::Vector{Float64}, du2::Vector{Float64})

    return abs(treino.m * du2[1] + treino.μ * du[1] + treino.k * u[1])

end