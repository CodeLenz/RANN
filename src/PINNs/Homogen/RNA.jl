# =============================================================================
#  Estruturas de dados da rede neural
# =============================================================================
struct Camada{T<:AbstractFloat}
    W::Matrix{T}
    b::Vector{T}
    φ::Function
    dφ::Function
end

struct Rede{T<:AbstractFloat}
    camadas::Vector{Camada{T}}
end

# -----------------------------------------------------------------------------
#  Inicialização
# -----------------------------------------------------------------------------
function Inicializa_Rede(larguras::Vector{Int}, ativacoes::Vector, 
                         ::Type{T}=Float64) where {T<:AbstractFloat}
    
    # Inicializa as camadas
    camadas = Camada{T}[]
    
    # Loop pelas camadas 
    for i in 1:length(larguras) - 1

        # Número de entradas e de saídas da camada
        n_in, n_out = larguras[i], larguras[i + 1]
        
        # Inicializa a matriz de pesos da camada
        # Inicialização de LeCun
        W = randn(T, n_out, n_in) * sqrt(T(1) / n_in)

        # Bias começam zerados
        b = zeros(T, n_out)
        
        # Guarda ativação e sua derivada 
        φ, dφ = ativacoes[i]

        # Guarda a camada na rede
        push!(camadas, Camada{T}(W, b, φ, dφ))
    end
    
    # Devolve a rede 
    return Rede{T}(camadas)

end

# -----------------------------------------------------------------------------
# Forward otimizado
# Z_buffers é um buffer de memória
# -----------------------------------------------------------------------------
function Forward_Rede_InPlace!(W::Vector{Matrix{T}}, b::Vector{Vector{T}}, rede::Rede{T}, X::Matrix{T}, 
                               Z_buffers::Vector{Matrix{T}}, As::Vector{Matrix{T}}) where {T<:AbstractFloat}
    
    # A primeira ativação recebe os dados do lote (coluna por coluna)
    As[1] .= X
    
    for (i, camada) in enumerate(rede.camadas)
        
        # Multiplica W * A e guarda no rascunho temporário
        mul!(Z_buffers[i], W[i], As[i])
        
        # Soma o bias no próprio rascunho
        Z_buffers[i] .+= b[i]
        
        # Aplica a ativação e salva em As[i+1]
        As[i+1] .= camada.φ.(Z_buffers[i])
        
    end
    
end

# -----------------------------------------------------------------------------
#  Passo reverso 
# -----------------------------------------------------------------------------
function Backward_Rede(W::Vector{Matrix{T}}, rede::Rede{T}, As::Vector{Matrix{T}},
                       dL_dAL::Matrix{T}) where {T<:AbstractFloat}

    # Número de camadas
    L  = length(rede.camadas)

    # Inicializa os gradientes para cada camada
    ∇W = Vector{Matrix{T}}(undef, L)
    ∇b = Vector{Vector{T}}(undef, L)

    # sensibilidades da saída
    Δ = dL_dAL .* rede.camadas[L].dφ.(As[end])

    # Loop pelas camadas, aplicando a backpropagation
    for i in L:-1:1

        #  Gradientes por pushback
        ∇W[i] = Δ * As[i]'
        ∇b[i] = vec(sum(Δ, dims = 2))
        
        # Evita calcularmos na primeira camada, pois não tem 
        # uma camada anterior
        if i > 1
            Δ = (W[i]' * Δ) .* rede.camadas[i - 1].dφ.(As[i])
        end
    end
    
    # Devolve os arrays com os gradientes por camada
    return ∇W, ∇b

end