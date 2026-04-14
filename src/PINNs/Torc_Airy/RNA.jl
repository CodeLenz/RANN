
# Atualiza os vetores de pesos e bias utilizando o vetor de variáveis de projeto x
function Atualiza_pesos_bias(rede::Rede, x::AbstractVector)

    # Acessa os termos em Rede por apelidos 
    n_camadas = rede.n_camadas
    topologia = rede.topologia
    pesos_ranges = rede.pesos_ranges
    bias_ranges  = rede.bias_ranges
    
    # Cria vetores contendo VIEWS (@view) em vez de alocar novas matrizes.
    # O reshape organiza a view em formato de matriz sem copiar dados.
    pesos = [reshape(@view(x[pesos_ranges[i]]), topologia[i+1], topologia[i]) for i in 1:n_camadas]
    bias  = [@view(x[bias_ranges[i]]) for i in 1:n_camadas]

    # Retorna as matrizes (views) de pesos e bias
    return pesos, bias

end


# Forward da Rede neural otimizado e AD-Friendly
function RNA(rede::Rede, pesos::Vector{<:AbstractMatrix{Float64}}, bias::Vector{<:AbstractVector{Float64}}, 
             entrada_i::AbstractVector{T}, prob::String)::Vector{T} where T

    # Promove a entrada estática para um Vector padrão.
    a = Vector{T}(entrada_i)

    # Loop pelas camadas
    for c in 1:rede.n_camadas
        
        # Aliases
        W = pesos[c]
        b = bias[c]
        ϕ = rede.ativ[c]

        # Calcula a combinação linear
        z = W * a .+ b

        # Aplica a função de ativação
        for i in eachindex(z)
            z[i] = ϕ(z[i])
        end

        # Atualiza para a próxima camada
        a = z
    end

    return a

end

# Reforço "forte" das condições de contorno
function RNA_forte(rede::Rede, pesos::Vector{<:AbstractMatrix{Float64}}, bias::Vector{<:AbstractVector{Float64}}, 
                   entrada_i::AbstractVector{T}, prob::String)::Vector{T} where T

    # Calcula saída da rede neural
    ψ = RNA(rede, pesos, bias, entrada_i, prob)

    # Função de distância do contorno
    s = Symbol("Distancia_Contorno_"*prob)
    B = getfield(Main, s)(entrada_i)
    
    # Função representativa do contorno - por enquanto, é zero
    # TODO: generalizar
    g = zeros(T,rede.topologia[end])

    # Saída da rede neural ajustada
    # Loop explícito para evitar avisos do Enzyme
    # Reaproveita ψ
    for i in eachindex(ψ)
        ψ[i] = g[i] + B * ψ[i]
    end

    # retorna ψ (antigo u)
    return ψ
    
end

