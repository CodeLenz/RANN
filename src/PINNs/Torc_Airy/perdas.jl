# Define a função de perda para a condição inicial ou condição de contorno
function Fn_CC_CI(y_pred::Vector{Float64}, y_esperado::Vector{Float64})

    return abs(y_pred[1] - y_esperado[1])

end



# Define a função de perda física para os pontos de treino
function Fn_perda_fisica(u::Vector{Float64}, du_xy::Vector{Vector{Float64}}, du2_xy::Vector{Vector{Float64}},
                         x::Vector{Float64})

    # Acessa a equação diferencial e calcula o valor do resíduo no ponto
    res = EqDiff(u, du_xy, du2_xy, x)
                       
    # Retorna a norma
    # Como é escalar, estamos usando abs para evitar o norm2
    return abs(res)

end