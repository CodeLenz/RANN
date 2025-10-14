#
# Função para obtenção das derivadas da rede neural em relação ao tempo
#
function Derivadas!(RNA!::Function, rede::Rede, sinais::Vector{Vector{Float64}},
                    pesos::Vector{Matrix{Float64}}, bias::Vector{Vector{Float64}},
                    u0::Vector{Float64}, du::Vector{Float64}, d2u::Vector{Float64},
                    t::Vector{Float64}, ϵ = 1E-6)

    #
    # Calcula a primeira e a segunda derivadas em relação ao tempo,
    # utilizando DFC
    #
    #
    # Testar aproximação de segunda ordem para a primeira derivada
    #
    # -( f(x+2δ) - 4f(x+δ) + 3f(x) ) / (2δ)
    # 
    # e tentar construir a segunda derivada por 
    #
    # (f'(x+δ) - f'(x-δ)) / (2δ)
    #
    # Daí podemos fazer uma rotina para calcular a primeira derivada e 
    # usar essa rotina para calcular a segunda derivada....
    #
    #

    # Valor escalar do tempo, para propormos uma perturbação 
    tt = t[1]

    # Estimativa de perturbação (caso tt>0). Isso faz com que a perturbação 
    # seja proporcional ao valor da variável em relação a qual estamos derivando
    δ = tt*ϵ

    #
    # Se a perturbação para trás for negativa, só podemos fazer DF para frente. Do contrário
    # podemos fazer central 
    # 
    if tt-δ<=0

        # Calcula a resposta para frente no tempo
        RNA!(rede, sinais, pesos, bias, t .+ ϵ)
        uf = copy(sinais[end])

        # Calcula a resposta mais para frente no tempo
        RNA!(rede, sinais, pesos, bias, t .+ 2*ϵ)
        uff = copy(sinais[end])

        # Estimativa da primeira derivada
        du .= (uf .- u0)/ϵ

        # Estimativa da segunda derivada para frente
        d2u .=  (uff .- 2.0 * uf .+ u0) ./ (ϵ^2)

    else    

        # Calcula a resposta para frente no tempo
        RNA!(rede, sinais, pesos, bias, t .+ δ)
        uf = copy(sinais[end])

        # Calcula a resposta para trás no tempo
        RNA!(rede, sinais, pesos, bias, t .- δ)
        ut = copy(sinais[end])
        
        # Primeira derivada por DFC
        du .= (uf .- ut) ./ (2 * δ)

        # Segunda derivada dor DFC
        d2u .= (uf .- 2.0 * u0 .+ ut) ./ (δ^2)

    end

end
    

#
# Função para obtenção das derivadas da rede neural em relação ao tempo
#
# Calcula a primeira e a segunda usando alta ordem...
#
#
function Derivadas_O2!(RNA!::Function, rede::Rede, sinais::Vector{Vector{Float64}},
                       pesos::Vector{Matrix{Float64}}, bias::Vector{Vector{Float64}},
                       u0::Vector{Float64}, du::Vector{Float64}, d2u::Vector{Float64},
                       t::Vector{Float64}, ϵ = 1E-8)

    
        # Tenta calcular a perturbação em função do valor do tempo 
        δ = ϵ

        # Calcula a resposta para frente no tempo
        RNA!(rede, sinais, pesos, bias, t .+ δ)
        uf = copy(sinais[end])

        # Calcula a resposta para frente no tempo 2
        RNA!(rede, sinais, pesos, bias, t .+ 2*δ)
        uff = copy(sinais[end])

        # Calcula a resposta para trás no tempo
        RNA!(rede, sinais, pesos, bias, t .- δ)
        ut = copy(sinais[end])
    
        # Calcula a resposta para trás no tempo 2
        RNA!(rede, sinais, pesos, bias, t .- 2*δ)
        utt = copy(sinais[end])
        
        # Primeira derivada (third order - five point stencil)
        #du .= -(uff .- 4*uf .+ 3*u0) ./ (2*δ)
        # Primeira derivada (third order)
        du .= ( utt .- 8*ut .+ 8*uf - uff )/(12*δ)

        # Segunda derivada
        d2u .= (-uff .+ 16*uf .- 30*u0 .+ 16*ut .- utt) ./ (12*δ^2)

end


#
# Função para obtenção da primeira derivada da rede neural em relação ao tempo
#
# Calcula a derivada usando uma aproximação de segunda ordem 
#
#  
# -( f(x+2δ) - 4f(x+δ) + 3f(x) ) / (2δ)
# 
#
#
function Derivada!(RNA!::Function, rede::Rede, sinais::Vector{Vector{Float64}},
                   pesos::Vector{Matrix{Float64}}, bias::Vector{Vector{Float64}}, 
                   du::Vector{Float64}, t::Vector{Float64}, ϵ = 1E-8)
        
        # Calcula a resposta no tempo atual 
        RNA!(rede, sinais, pesos, bias, t )
        du .= 3*sinais[end]
        
        # Calcula a resposta para frente no tempo
        RNA!(rede, sinais, pesos, bias, t .+ ϵ)
        du .-= 4*sinais[end]

        # Calcula a resposta mais para frente no tempo
        RNA!(rede, sinais, pesos, bias, t .+ 2*ϵ)
        du .+= sinais[end]

        # Calcula a estimativa de segunda ordem 
        du .= -du ./ (2*ϵ)

end


#
# Calcula a segunda derivada por meio da diferença da primeira derivada 
#
function Derivada2!(RNA!::Function, rede::Rede, sinais::Vector{Vector{Float64}},
                    pesos::Vector{Matrix{Float64}}, bias::Vector{Vector{Float64}}, 
                    du2::Vector{Float64}, t::Vector{Float64}, ϵ = 1E-8)
        
        # Calcula a derivada em um ponto para frente
        duf = zeros(1)
        Derivada!(RNA!, rede, sinais, pesos, bias, duf, t.+ϵ, ϵ)

        # Calcula a derivada em um ponto para tras
        dut = zeros(1)
        Derivada!(RNA!, rede, sinais, pesos, bias, dut, t.-ϵ, ϵ)

        # Calcula a estimativa da segunda derivada
        du2 .=  ( duf .- dut )./(2*ϵ)

end
    