
# Atualiza os vetores de pesos e bias utilizando o vetor de variáveis de projeto x.
# Evitamos @view/reshape sobre o vetor ativo porque o Enzyme não lida bem com aliasing
# de slices em reverse-mode. Copiamos os trechos do vetor antes de reorganizá-los.
function Atualiza_pesos_bias(rede::Rede, x::AbstractVector)

    n_camadas = rede.n_camadas
    topologia = rede.topologia

    pesos = Vector{Matrix{Float64}}(undef, n_camadas)
    bias  = Vector{Vector{Float64}}(undef, n_camadas)

    k = 1
    for i in 1:n_camadas
        n_pesos = rede.conexoes[i]
        pesos[i] = reshape(copy(x[k:(k + n_pesos - 1)]), topologia[i+1], topologia[i])
        k += n_pesos

        n_bias = topologia[i+1]
        bias[i] = copy(x[k:(k + n_bias - 1)])
        k += n_bias
    end

    return pesos, bias

end


# Forward da Rede neural otimizado e AD-Friendly
# Versão que recebe o vetor de parâmetros achatado da rede.
function RNA(rede::Rede, x::AbstractVector{Float64}, entrada_i::AbstractVector{T}, prob::String)::Vector{T} where T

    a = Vector{T}(entrada_i)

    for c in 1:rede.n_camadas
        W = reshape(@view(x[rede.pesos_ranges[c]]), rede.topologia[c+1], rede.topologia[c])
        b = @view(x[rede.bias_ranges[c]])
        ϕ = rede.ativ[c]

        z = W * a .+ b

        for i in eachindex(z)
            z[i] = ϕ(z[i])
        end

        a = z
    end

    return a
end

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

# Seleciona a distância de contorno de forma estática para manter a diferenciação
# em Enzyme compatível com o modo reverse.
function Distancia_Contorno(prob::String, XY::AbstractVector{T}) where {T}
    if prob == "Retangular"
        return Distancia_Contorno_Retangular(XY)
    elseif prob == "Circular"
        return Distancia_Contorno_Circular(XY)
    elseif prob == "L"
        return Distancia_Contorno_L(XY)
    else
        error("Geometria não suportada: $(prob)")
    end
end

# Reforço "forte" das condições de contorno. Versão direta sobre o vetor de parâmetros.
function RNA_forte(rede::Rede, x::AbstractVector{Float64}, entrada_i::AbstractVector{T}, prob::String)::Vector{T} where T

    ψ = RNA(rede, x, entrada_i, prob)
    B = Distancia_Contorno(prob, entrada_i)
    g = zeros(T, rede.topologia[end])

    for i in eachindex(ψ)
        ψ[i] = g[i] + B * ψ[i]
    end

    return ψ
end

function RNA_forte(rede::Rede, pesos::Vector{<:AbstractMatrix{Float64}}, bias::Vector{<:AbstractVector{Float64}}, 
                   entrada_i::AbstractVector{T}, prob::String)::Vector{T} where T

    # Calcula saída da rede neural
    ψ = RNA(rede, pesos, bias, entrada_i, prob)

    # Função de distância do contorno. A seleção direta evita o lookup dinâmico por
    # Symbol/getfield, que não é suportado pelo Enzyme em autodiff reverso.
    B = Distancia_Contorno(prob, entrada_i)

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

