#
# Função para obtenção das derivadas da rede neural em relação ao tempo
#
# Muito cuidado com a perturbação, por causa da segunda derivada...
#
#
function Derivadas!(RNA::Function, rede::Rede, 
                    pesos::Vector{Matrix{Float64}}, bias::Vector{Vector{Float64}},
                    u0::Vector{Float64}, du::Vector{Float64}, d2u::Vector{Float64},
                    t::Vector{Float64}, ϵ = 1E-8)

    #
    # Calcula a primeira e a segunda derivadas em relação ao tempo,
    # utilizando DFC complexas
    #
    
    # Valor escalar do tempo, para propormos uma perturbação complexa
    tt = t[1] + ϵ*im

    # Calcula a resposta para frente no tempo
    uf = RNA(rede, pesos, bias, [tt])
       
    # A primeira derivada pode ser obtida com 
    du.= imag.(uf)/ϵ

    # A segunda derivada pode ser obtida com 
    d2u .= -2*(real.(uf).-u0)/(ϵ^2)
    
end
    

#
# Função para obtenção das derivadas da rede neural em relação ao tempo
#
# Baseado na interessante proposta de Albert Chan em 
# https://www.hpmuseum.org/forum/thread-16299.html
#
#
# Detalhe que a perturbação não pode ser muito pequena, por causa da segunda derivada
#
function DerivadasC2!(RNA::Function, rede::Rede, 
                    pesos::Vector{Matrix{Float64}}, bias::Vector{Vector{Float64}},
                    u0::Vector{Float64}, du::Vector{Float64}, d2u::Vector{Float64},
                    t::Vector{Float64}, ϵ = 1E-3)

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