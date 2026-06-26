# =============================================================================
#  Rotinas para PINNs de Homogeneização
# =============================================================================

# -----------------------------------------------------------------------------
#  Geração de malha Quasi-Monte Carlo via sequência de Sobol
# -----------------------------------------------------------------------------
function Gera_Pontos_Sobol(N_pontos::Int, ::Type{T}=Float64) where {T<:AbstractFloat}

    # Definimos uma sequência de Sobol em R^2
    seq = SobolSeq(2)

    # Aloca a matriz de coordenadas para os pontos de Sobol
    pontos = Matrix{T}(undef, N_pontos, 2)

    # Loop pelos pontos 
    for i in 1:N_pontos

        # Pega o próximo ponto da sequência
        p = Sobol.next!(seq)

        # Guarda na matriz 
        pontos[i, 1] = T(p[1])
        pontos[i, 2] = T(p[2])

    end

    # Retorna a matriz de pontos
    return pontos

end

# -----------------------------------------------------------------------------
#  QMC randomizado: rotação de Cranley-Patterson (shift aleatório mod 1).
#  Mantém a baixa discrepância do Sobol e gera um conjunto NOVO a cada rodada.
# -----------------------------------------------------------------------------
function Gera_Pontos_QMC_Rotacionado(base::Matrix{T}, shift::Vector{T}) where {T<:AbstractFloat}
    P = similar(base)
    @inbounds for j in 1:size(base,2), i in 1:size(base,1)
        P[i,j] = mod(base[i,j] + shift[j], one(T))
    end
    return P
end

# -----------------------------------------------------------------------------
#  Camada que transforma entrada (y1,y2) em um sinal pediódico de 
#  dimensão 4N (número de modos)
# -----------------------------------------------------------------------------
function Camada_Periodica(y1::T, y2::T, N::Int) where {T<:AbstractFloat}
    reduce(vcat, [[sin(2π * k * y1), cos(2π * k * y1), sin(2π * k * y2), cos(2π * k * y2)] for k in 1:N])
end

# -----------------------------------------------------------------------------
#
#  Integral da Energia de Deformação (Versão Vetorizada)
#
#  Troca loop escalar por operações de fatiamento 
#
# -----------------------------------------------------------------------------
function Perda_Energia_Alvo(AL::Matrix{T}, pontos::Matrix{T}, ε_macro::Matrix{T}, prob::String,
                            mat_params::NamedTuple, λ_avg::T) where {T<:AbstractFloat}

    # Número de pontos Sobol no lote
    N_pts = size(pontos, 1)
    
    # Perturbação paramétrica para as diferenças finitas
    h = T(1e-4)

    # Recupera todas as coordenadas espaciais de uma só vez
    y1 = pontos[:, 1]
    y2 = pontos[:, 2]

    #
    # Vetorização da extração de deslocamentos
    #

    # Centro
    u1_C = AL[1, 1:5:end]
    u2_C = AL[2, 1:5:end]

    # Leste (y1​+h)
    u1_E = AL[1, 2:5:end] 
    u2_E = AL[2, 2:5:end]

    # Oeste (y1​−h)
    u1_W = AL[1, 3:5:end] 
    u2_W = AL[2, 3:5:end]

    # Norte (y2​+h)
    u1_N = AL[1, 4:5:end] 
    u2_N = AL[2, 4:5:end]

    # Sul (y2​−h)
    u1_S = AL[1, 5:5:end] 
    u2_S = AL[2, 5:5:end]
    
    #
    # Diferenças finitas centrais do gradiente de deslocamentos (em lote)
    #
    du1_dy1 = (u1_E .- u1_W) ./ (T(2.0) * h)
    du1_dy2 = (u1_N .- u1_S) ./ (T(2.0) * h)
    du2_dy1 = (u2_E .- u2_W) ./ (T(2.0) * h)
    du2_dy2 = (u2_N .- u2_S) ./ (T(2.0) * h)
    
    #
    # Deformações globais para todos os pontos 
    #
    ε11 = ε_macro[1,1] .+ du1_dy1
    ε22 = ε_macro[2,2] .+ du2_dy2
    ε12 = ε_macro[1,2] .+ T(0.5) .* (du1_dy2 .+ du2_dy1)

    #
    # Avaliação do material usando broadcast 
    # o Ref(mat_params) serve para avisar o compilador que mat_params 
    # é constante e não deve ser vetorizado, apenas y1 e y2.
    #
    props_simbolo = Symbol("Propriedades_Material_"*prob)
    props = getfield(Main, props_simbolo).(pontos[:, 1], pontos[:, 2], Ref(mat_params))
    
    # Separa os vetores de E e ν que agora estão em lote também
    E_C = first.(props)
    ν_C = last.(props)

    #
    # Construção termo a termo de C, com vetorização
    #
    fac = E_C ./ (T(1.0) .- ν_C.^2)
    C11 = fac
    C22 = fac
    C12 = fac .* ν_C
    C33 = fac .* (T(1.0) .- ν_C) ./ T(2.0)
    
    #
    # Acúmulo da densidade de energia (0.5 * ε^T * C * ε) expandida termo a termo
    #
    energia_pontual = T(0.5) .* (ε11 .* C11 .* ε11 .+ 
                                 ε22 .* C22 .* ε22 .+ 
                                 T(2.0) .* ε11 .* C12 .* ε22 .+ 
                                 T(4.0) .* ε12 .* C33 .* ε12)
    
    # Estimativa por Monte Carlo (Média do lote)
    L_energia = sum(energia_pontual) / N_pts
    
    # Restrição de corpo rígido (valor médio nulo)
    L_avg = (sum(u1_C) / N_pts)^2 + (sum(u2_C) / N_pts)^2

    perda = L_energia + λ_avg * L_avg
    
    # Retorna a perda total da PINN
    return perda, L_energia, L_avg

end






