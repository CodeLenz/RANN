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
# Calcula a primeira e a segunda usando alta ordem...o valor da função 
# no ponto atual é calculado antes da chamada da rotina e deve ser informado
# via o parâmetro u0.
#
# A saída da rede pode ser obtida com sinais[end], após a chamada desta rotina
#
#
function Derivadas_O2!(RNA!::Function, rede::Rede, sinais::Vector{Vector{Float64}},
                       pesos::Vector{Matrix{Float64}}, bias::Vector{Vector{Float64}},
                       u0::Vector{Float64}, du::Vector{Float64}, d2u::Vector{Float64},
                       t::Vector{Float64}, ϵ = 1E-8)

    
        # 
        # Tenta calcular a perturbação em função do valor do tempo 
        #
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
        du .= ( utt .- 8*ut .+ 8*uf - uff )/(12*δ)

        # Segunda derivada
        # TODO: Em alguns casos esse valor está dando muito alto, pois o 
        # a parte de cima da fração está dando na casa de 1E0 ou 1E1. 
        # 
        # Em um primeiro momento eu imaginei que isso seria devido a estarmos 
        # em um ponto de mínimo ou de máximo em relação ao tempo, (velocidade 
        # praticamente nula). Com isso, poderíamos ter uma estimativa numérica 
        # bem ruim da segunda derivada, pois ela é uma função da variação da primeira 
        # derivada. 
        # Não tenho bem certeza se é isso, pois em uns testes usando a aproximação 
        #
        # f(x+δ) ≈ f(x) + df δ + 1/2 δ^2 d2f  e o fato de df ser nula nestes pontos 
        #  
        # d2f ≈ 2(f(x+δ) - f(x)) /  δ^2
        #
        # que acabou dando a mesma coisa do que a fórmula que estamos usando hoje.
        #
        # Portanto, o mais provável é que a rede não propague muito bem algumas 
        # perturbações e isso implique em erro na segunda derivada.
        #
        d2u .= (-uff .+ 16*uf .- 30*u0 .+ 16*ut .- utt) ./ (12*δ^2)

end


#
# Função para obtenção das derivadas da rede neural em relação ao tempo
#
# Calcula a primeira e a segunda usando alta ordem...o valor da função 
# no ponto atual é calculado antes da chamada da rotina e deve ser informado
# via o parâmetro u0.
#
# A saída da rede pode ser obtida com sinais[end], após a chamada desta rotina
#
#
function Derivadas_Richard!(RNA!::Function, rede::Rede, sinais::Vector{Vector{Float64}},
                            pesos::Vector{Matrix{Float64}}, bias::Vector{Vector{Float64}},
                            u0::Vector{Float64}, du::Vector{Float64}, d2u::Vector{Float64},
                            t::Vector{Float64}, ϵ = 1E-8)

    
        # 
        # Tenta calcular a perturbação em função do valor do tempo 
        #
        δ = max(ϵ, t[1]*1E-6)
    
        # Calcula a resposta para frente no tempo: δ 
        RNA!(rede, sinais, pesos, bias, t .+ δ)
        uf = copy(sinais[end])

        # Calcula a resposta para frente no tempo: δ/2
        RNA!(rede, sinais, pesos, bias, t .+ δ/2)
        uf2 = copy(sinais[end])

        # Calcula a resposta para trás no tempo: -δ
        RNA!(rede, sinais, pesos, bias, t .- δ)
        ut = copy(sinais[end])
    
        # Calcula a resposta para trás no tempo: -δ/2
        RNA!(rede, sinais, pesos, bias, t .- δ/2)
        ut2 = copy(sinais[end])
        
        # Primeira derivada para frente e primeira derivada para trás
        d1h  = (uf-ut)/(2*δ)
        d1h2 = (uf2-ut2)/(δ)

        # Primeira derivada 
        du .= (4*d1h2 .- d1h)./3

        # Segunda derivada para frente e segunda para trás
        d2h  = (uf .- 2*u0 .+ ut)/(δ^2)
        d2h2 = (uf2 .-2*u0 .+ ut2)/((δ/2)^2)

        # Segunda derivada 
        d2u.= (4*d2h2 .- d2h) ./ 3
        
end



#
# Função para obtenção das derivadas da rede neural em relação ao tempo
#
#
# A saída da rede pode ser obtida com sinais[end], após a chamada desta rotina
#
# Aqui estou realmente usando a interpolação de 5 pontos
#
#
function Derivadas_O3!(RNA!::Function, rede::Rede, sinais::Vector{Vector{Float64}},
                       pesos::Vector{Matrix{Float64}}, bias::Vector{Vector{Float64}},
                       u0::Vector{Float64}, du::Vector{Float64}, d2u::Vector{Float64},
                       t::Vector{Float64}, ϵ = 1E-8)

        δ = ϵ

        # Pontos em que vamos calcular os valores da função para aproximar a 
        # primeira e a segunda derivadas
        # A sequência será 
        #     1     2    3    4     5
        #   t-2δ   t-δ   t   t+δ   t+2δ
        #
        #
        # Assuminos que u3 é o valor calculado no ponto 3 (ponto atual)

       
        # Calcula a resposta para frente no tempo
        xf = t[1]+δ
        RNA!(rede, sinais, pesos, bias, t .+ δ)
        uf = copy(sinais[end])

        # Calcula a resposta para frente no tempo 2
        xff = t[1]+2*δ
        RNA!(rede, sinais, pesos, bias, t .+ 2*δ)
        uff = copy(sinais[end])

        # Posição atual 
        x0 = t[1]

        # Calcula a resposta para trás no tempo
        xt = t[1]-δ
        RNA!(rede, sinais, pesos, bias, t .- δ)
        ut = copy(sinais[end])
    
        # Calcula a resposta para trás no tempo 2
        xtt = t[1]-2*δ
        RNA!(rede, sinais, pesos, bias, t .- 2*δ)
        utt = copy(sinais[end])

        # Calcula os coeficientes da aproximação polinomial de quarta 
        # ordem 
        A = [1 xtt xtt^2 xtt^3 xtt^4 ; 
             1 xt  xt^2  xt^3  xt^4  ;
             1 x0  x0^2  x0^3  x0^4  ; 
             1 xf  xf^2  xf^3  xf^4  ;
             1 xff xff^2 xff^3 xff^4 ] 
        b = [utt ; ut; u0; uf; uff]     

        # Os coeficientes serão 
        coefs = A\b

        # Primeira derivada
        du .= coefs[2] + 2*coefs[3]*x0 + 3*coefs[4]*x0^2 + 4*coefs[5]*x0^3

        # Segunda derivada
        d2u .= 2*coefs[3] + 6*coefs[4]*x0 + 12*coefs[5]*x0^2

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
    