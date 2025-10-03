# Função para obtenção da primeira derivada da rede neural
# Método das diferenças finitas
function Derivadas!(u::Function, rede::Rede,  pesos::Vector{Vector{Float64}}, bias::Vector{Vector{Float64}},
                    u0::Vector{Float64}, du::Vector{Float64}, d2u::Vector{Float64},
                    t::Vector{Float64}, ϵ = 1E-8)

    # Calcula a primeira e a segunda derivada em relação ao tempo utilizando DFC
    uf = u(rede, pesos, bias, t .+ ϵ)
    ut = u(rede, pesos, bias, t .- ϵ)
    
    # Primeira derivada
    du[1] = (uf[1] - ut[1]) / (2 * ϵ)

    # Segunda derivada 
    d2u[1] = (uf[1] - 2.0 * u0[1] + ut[1]) / (ϵ^2)

end
    

#=
# Função para obtenção da primeira derivada da rede neural
function PrimeiraDerivada!(u::Function, rede, x::Vector, du::Vector{Float64}, t::Vector{Float64},ϵ=1E-8)


    # Aproxima a primeira derivada utilizando DF
    uf = u(rede,x,t.+ϵ)
    ut = u(rede,x,t.-ϵ)
    du[1] = (uf-ut)/(2*ϵ)
    
    #=
    # Derivação automática
    Enzyme.autodiff(
                    set_runtime_activity(Reverse),
                    u,
                    Const(rede),
                    Const(x),
                    Duplicated(t, du)
                )
    =#

end

# Função para obtenção da segunda derivada da rede neural
function SegundaDerivada!(u::Function, rede, x::Vector, du0::Vector{Float64}, du2::Vector{Float64}, t::Vector{Float64}, ϵ=1E-6)

    # A primeira derivada no ponto atual é informada em du0

    # Com isso, podemos calcular a segunda derivada por DF

    # Calcula a primeira derivada para frente, aproveitando o vetor du2
    #=
    PrimeiraDerivada!(u, rede, x, du2, t.+ϵ)

    # Calcula du2 aproveitando a memória
    du2 .= (du2.-du0)./ϵ
    =#

end
=#