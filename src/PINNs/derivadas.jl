#
# Função para obtenção da primeira derivada da rede neural em relação ao tempo
#
function Derivadas!(u::Function, rede::Rede,  pesos::Vector{Matrix{Float64}}, bias::Vector{Vector{Float64}},
                    u0::Vector{Float64}, du::Vector{Float64}, d2u::Vector{Float64},
                    t::Vector{Float64}, ϵ = 1E-8)

    # Calcula a resposta para frente no tempo
    uf = u(rede, pesos, bias, t .+ ϵ)

    # Calcula a resposta para trás no tempo
    ut = u(rede, pesos, bias, t .- ϵ)
    
    # Primeira derivada por DFC
    du[1] = (uf[1] - ut[1]) / (2 * ϵ)

    # Segunda derivada dor DFC
    d2u[1] = (uf[1] - 2.0 * u0[1] + ut[1]) / (ϵ^2)

end
    
