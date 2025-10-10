#
# Função para obtenção da primeira derivada da rede neural em relação ao tempo
#
function Derivadas!(RNA!::Function, rede::Rede, sinais::Vector{Vector{Float64}},
                    pesos::Vector{Matrix{Float64}}, bias::Vector{Vector{Float64}},
                    u0::Vector{Float64}, du::Vector{Float64}, d2u::Vector{Float64},
                    t::Vector{Float64}, ϵ = 1E-8)

    # Calcula a resposta para frente no tempo
    RNA!(rede, sinais, pesos, bias, t .+ ϵ)
    uf = copy(sinais[end])

    # Calcula a resposta para trás no tempo
    RNA!(rede, sinais, pesos, bias, t .- ϵ)
    ut = copy(sinais[end])
    
    # Primeira derivada por DFC
    du .= (uf .- ut) ./ (2 * ϵ)

    # Segunda derivada dor DFC
    d2u .= (uf .- 2.0 * u0 .+ ut) ./ (ϵ^2)

end
    
