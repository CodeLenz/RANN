#
# Função para obtenção das derivadas da rede neural em relação ao tempo
#
# Baseado na interessante proposta de Albert Chan em 
# https://www.hpmuseum.org/forum/thread-16299.html
#
#
# Detalhe que a perturbação não pode ser muito pequena, por causa da segunda derivada
#
function DerivadasC2!(RNA::Function, rede::Rede, pesos::Vector{Matrix{Float64}}, bias::Vector{Vector{Float64}},
                      u0::Vector{Float64}, du::Vector{Float64}, d2u::Vector{Float64}, t::Vector{Float64},
                      ϵ = 1E-3)

    #
    # Calcula a primeira e a segunda derivadas em relação ao tempo,
    # utilizando DFC complexas
    #
    
    # Valor escalar do tempo, para propormos uma perturbação complexa
    h = ϵ*sqrt(im)
    tf = t[1] + h
    tt = t[1] - h

    # Calcula a resposta para frente no tempo
    uf = RNA(rede, pesos, bias, [tf])

    # Calcula a resposta para trás no tempo
    ut = RNA(rede, pesos, bias, [tt])

    # Primeira derivada na cara dura
    du .= real( (uf.-ut)/(2*h) ) 
    
    # A segunda derivada na cara dura
    d2u .= real( (uf .- 2*u0 .+ ut)/(h^2) )
    
end

#
# f:: Função a derivar R^n -> R
#
# x:: ponto atual 
#
# ϵ:: perturbação 
#
function DerivadasPDE!(RNA::Function, rede::Rede, pesos::Vector{Matrix{Float64}}, bias::Vector{Vector{Float64}},
                      u0::Vector{Float64}, du_xy::Vector{Vector{Float64}}, d2u_xy::Vector{Vector{Float64}},  
                      x::Vector{Float64}, ϵ = 1E-3)    

    # Número de variáveis de projeto 
    nvp = length(x)

    # Valor escalar do tempo, para propormos uma perturbação complexa
    h = ϵ * sqrt(im)

    # Promove x para complexo 
    xc = zeros(ComplexF64, nvp)
 
    # Copia os dados atuais para o vetor complexo
    xc .= x

    # Loop em cada dimensão  
    for i in eachindex(xc)

       # Backup de x[i]
       bx = xc[i]

       # Perturbação para frente
       xc[i] = bx + h

       # Calcula para frente
       uf = RNA(rede, pesos, bias, xc)

       # Perturba para trás
       xc[i] = bx - h

       # Calcula para trás
       ut = RNA(rede, pesos, bias, xc)

       # Primeira derivada em relação a x[i]
       du_xy[i] .= real( (uf - ut) / (2 * h) ) 
    
       # Segunda derivada em relação a x[i]
       d2u_xy[i] .= real( (uf - 2 * u0 + ut) / h^2 )

       # Restaura 
       xc[i] = bx

    end

end