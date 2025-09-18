# Define função de ativação ReLU
# Rectified Linear Unit
# Retorna o valor de entrada se for positivo e zero se for negativo
function ReLU(x::T) where T

    # Retorna o valor máximo entre x e 0
    return max(zero(T), x)

end

# Define a função de perda para a condição de contorno essencial/natural
function Fn_perda_contorno(y_pred::Vector{Float64}, y_esperado::Vector{Float64})

    return norm(y_pred .- y_esperado)

end

# Define a função de perda física para os pontos de treino
function Fn_perda_fisica(treino::Treino, u::Vector{Float64}, du::Vector{Float64}, du2::Vector{Float64})

    return norm(treino.m .* du2 .+ treino.μ .* du .+ treino.k .* u)

end