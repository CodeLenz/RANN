# Define a função de perda física para os pontos de treino
function Fn_perda_fisica(u::Vector{Float64}, du_xy::Vector{Vector{Float64}}, du2_xy::Vector{Vector{Float64}},
                         x::Vector{Float64})

    # Calcula as deformações
    ϵ11 = ϵ11_macro + ( u_viz[2] - u_viz[3] ) / 2h
    ϵ22 = ϵ22_macro + ( u_viz[4] - u_viz[5] ) / 2h
    ϵ12 = ϵ12_macro + 0.5 * ( ( u_viz[4] - u_viz[5] ) / 2h + ( u_viz[2] - u_viz[3] ) / 2h)

    # Calcula as derivadas segundas dos deslocamentos
    d2u_dy12 = ( u_viz[2] - 2 * u_viz[1]  + u_viz[3] ) / h^2
    d2u_dy22 = ( u_viz[4] - 2 * u_viz[1]  + u_viz[5] ) / h^2
    d2u_dy1y2 = ( u_viz[6] - u_viz[7] - u_viz[8] + u_viz[9] ) / (4 * h^2)

    # Acessa a equação diferencial e calcula o valor do resíduo no ponto
    res = EqDiff(u, du_xy, du2_xy, x)
                       
    # Retorna a norma
    # Função quadrática para gradientes mais suaves
    return (res)^2

end