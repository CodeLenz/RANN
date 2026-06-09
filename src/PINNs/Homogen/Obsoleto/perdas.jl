# Define a função de perda física para os pontos de treino
function Fn_perda_fisica(u_viz::AbstractVector{T}, ϵ_macro::Matrix{Float64}, C::AbstractMatrix{T},
                         dC_dy1::AbstractMatrix{T}, dC_dy2::AbstractMatrix{T}) where T

    # Calcula as deformações
    ϵ11 = ϵ_macro[1,1] + ( u_viz[2] - u_viz[3] ) / 2h
    ϵ22 = ϵ_macro[2,2] + ( u_viz[4] - u_viz[5] ) / 2h
    ϵ12 = ϵ_macro[1,2] + 0.5 * ( ( u_viz[4] - u_viz[5] ) / 2h + ( u_viz[2] - u_viz[3] ) / 2h)

    # Calcula as derivadas segundas dos deslocamentos
    d2u_d2y1 = ( u_viz[2] .- 2 .* u_viz[1] .+ u_viz[3] ) ./ h^2
    d2u_d2y2 = ( u_viz[4] - 2 .* u_viz[1]  .+ u_viz[5] ) ./ h^2
    d2u_dy1y2 = ( u_viz[6] .- u_viz[7] .- u_viz[8] .+ u_viz[9] ) ./ (4 .* h^2)

    # Calcula os resíduos
    r1 = dC_dy1[1,1] * ϵ11 + dC_dy1[1,2] * ϵ22 + dC_dy2[3,3] * 2.0 * ϵ12
            + C[1,1] * d2u_d2y1[1] + C[1,2] * d2u_dy1y2[2] 
            + C[3,3] * (d2u_d2y2[1] + d2u_dy1y2[2]) 

    r2 = dC_dy1[3,3] * 2.0 * ϵ12 + dC_dy1[2,1] * ϵ11 + dC_dy2[2,2] * ϵ22
            + C[3,3] * (d2u_dy1y2[1] + d2u_d2y1[2]) + C[2,1] * d2u_dy1y2[1] 
            + C[2,2] * d2u_d2y2[2]

    # Calcula a perda física
    perda_fis = 0.5 * (norm(r1) + norm(r2))
                       
    # Retorna a perda física
    return perda_fis

end