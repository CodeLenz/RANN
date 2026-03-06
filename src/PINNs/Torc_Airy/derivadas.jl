#
# Função para obtenção das derivadas da rede neural em relação ao tempo
#
# Baseado na interessante proposta de Albert Chan em 
# https://www.hpmuseum.org/forum/thread-16299.html
#
#
# Detalhe que a perturbação não pode ser muito pequena, por causa da segunda derivada
#
function DerivadasC2!(RNA::Function, rede::Rede, pesos::Vector{<:AbstractMatrix{Float64}}, bias::Vector{<:AbstractVector{Float64}},
                      u0::Vector{Float64}, du::Vector{Float64}, d2u::Vector{Float64}, t::Vector{Float64},
                      ϵ = 1E-3)

    # Valor escalar do tempo, para propormos uma perturbação complexa
    h = ϵ * sqrt(im)
    
    # SVector aloca diretamente na Stack, blindando a memória e acelerando o processo
    tf = SVector{1, ComplexF64}(t[1] + h)
    tt = SVector{1, ComplexF64}(t[1] - h)

    # Calcula a resposta para frente no tempo
    uf = RNA(rede, pesos, bias, tf)

    # Calcula a resposta para trás no tempo
    ut = RNA(rede, pesos, bias, tt)

    # Primeira derivada na cara dura
    du[1] = real( (uf[1] - ut[1]) / (2*h) ) 
    
    # A segunda derivada na cara dura
    d2u[1] = real( (uf[1] - 2*u0[1] + ut[1]) / h^2 )
    
end

#
# f:: Função a derivar R^n -> R
#
# x:: ponto atual 
#
# ϵ:: perturbação 
#
#
# f:: Função a derivar R^n -> R
#
function DerivadasPDE!(RNA::Function, rede::Rede, pesos::Vector{<:AbstractMatrix{Float64}}, bias::Vector{<:AbstractVector{Float64}},
                      u0::Vector{Float64}, du_xy::Vector{Vector{Float64}}, d2u_xy::Vector{Vector{Float64}},  
                      x::Vector{Float64}, ϵ = 1E-3)    

    # Valor escalar da perturbação complexa
    h = ϵ * sqrt(im)

    # Cria um MVector mutável 
    xc = MVector{2, ComplexF64}(x[1], x[2]) 
 
    # Loop em cada dimensão espacial (x, depois y)
    for i in 1:2

       # Backup da coordenada original
       bx = xc[i]

       # Perturbação para frente
       xc[i] = bx + h
       uf = RNA(rede, pesos, bias, xc)

       # Perturba para trás
       xc[i] = bx - h
       ut = RNA(rede, pesos, bias, xc)

       # Primeira derivada em relação a x[i]
       # Acessamos o índice [1] diretamente e usamos = em vez de .= 
       # para evitar o broadcast e acelerar ainda mais
       du_xy[i][1] = real( (uf[1] - ut[1]) / (2 * h) ) 
    
       # Segunda derivada em relação a x[i]
       d2u_xy[i][1] = real( (uf[1] - 2 * u0[1] + ut[1]) / h^2 )

       # Restaura a coordenada original para o próximo loop
       xc[i] = bx

    end

end
