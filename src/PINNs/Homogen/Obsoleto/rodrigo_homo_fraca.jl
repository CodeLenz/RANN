# =============================================================================
#  Programa de homogeneização usando PINNs
#  O programa minimiza a energia de deformação da célula unitária utilizando
#  integração Quasi-Monte Carlo (Sobol). 
#
#  Aproveitei para dar uma estudada em parametrização de tipos no Julia 
#  e também usar o zygote.pullback
#
#  Lenz maio de 2026
# =============================================================================
using LinearAlgebra
using Zygote
using Sobol

# -----------------------------------------------------------------------------
#  Funções de ativação parametrizadas para a saída ativada 'a'
# -----------------------------------------------------------------------------
TANH_GEN   = (tanh,     a -> one(a) - a^2)
LINEAR_GEN = (identity, a -> one(a))

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
#  Passo direto 
# -----------------------------------------------------------------------------
function Forward_Rede(rede::Rede{T}, X::Matrix{T}) where {T<:AbstractFloat}

    # Inicializa o vetor de matrizes das ativações
    As = [X]
    A  = X
    
    # Loop pelas camadas
    for camada in rede.camadas

        # Pré-ativação
        Z = camada.W * A .+ camada.b

        # Aplica a função de ativação 
        A = camada.φ.(Z)

        # Guarda a ativação no histórico
        push!(As, A)
    end
    
    # Retorna o vetor de matrizes de ativações
    return As

end

# -----------------------------------------------------------------------------
#  Passo reverso 
# -----------------------------------------------------------------------------
function Backward_Rede(rede::Rede{T}, As::Vector{Matrix{T}}, dL_dAL::Matrix{T}) where {T<:AbstractFloat}

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
            Δ = (rede.camadas[i].W' * Δ) .* rede.camadas[i - 1].dφ.(As[i])
        end
    end
    
    # Devolve os arrays com os gradientes por camada
    return ∇W, ∇b

end

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
        p = next!(seq)

        # Guarda na matriz 
        pontos[i, 1] = T(p[1])
        pontos[i, 2] = T(p[2])
    end

    # Retorna a matriz de pontos
    return pontos

end

# -----------------------------------------------------------------------------
#  Camada que transforma entrada (y1,y2) em um sinal pediódico de 
#  dimensão 4N (número de modos)
# -----------------------------------------------------------------------------
function Camada_Periodica(y1::T, y2::T, N::Int) where {T<:AbstractFloat}
    reduce(vcat, [[sin(2π * k * y1), cos(2π * k * y1), sin(2π * k * y2), cos(2π * k * y2)] for k in 1:N])
end

# -----------------------------------------------------------------------------
#  Rotina para devolver E, ν para um ponto (y_1,y_2)
#  em uma fibra de raio r0 em uma matriz 
# -----------------------------------------------------------------------------
function Propriedades_Material(y1::T, y2::T, params::NamedTuple) where {T<:AbstractFloat}

    # Calcula a distância em relação ao centro
    d = sqrt((y1 - params.yc1)^2 + (y2 - params.yc2)^2)
    
    # Se a distância for menor do que o raio da inclusão 
    # usamos as propriedades da fibra
    if d <= params.r0 
       E = params.E_f
       ν = params.ν_f
    else 
       E = params.E_m
       ν = params.ν_m
    end

    # Retorna as propriedades
    return E, ν

end

# -----------------------------------------------------------------------------
#  \C para EPT
# -----------------------------------------------------------------------------
function Matriz_Constitutiva(E::T, ν::T) where {T<:AbstractFloat}

    # termo comum 
    fac = E / (T(1.0) - ν^2)

    # Matriz EPT
    C = fac * [T(1.0)  ν      T(0.0);
               ν      T(1.0)  T(0.0);
               T(0.0)  T(0.0)  (T(1.0) - ν)/T(2.0)]

    # Devolve a matriz
    return C

end

# -----------------------------------------------------------------------------
#  Integral da Energia de Deformação
#
#  Essa rotina é chave para o pushback do Zygote. Ver a estratégia de 
#  vetorização.
#
# -----------------------------------------------------------------------------
function Perda_Energia_Alvo(AL::Matrix{T}, pontos::Matrix{T}, ε_macro::Matrix{T},
                            mat_params::NamedTuple, λ_avg::T) where {T<:AbstractFloat}

    # Número de pontos Sobol
    N_pts = size(pontos, 1)
    
    # Perturbação para o stencil
    h = T(1e-4)

    # Incializa as variáveis para calcular o objetivo
    # e a restrição (valor médio nulo)
    L_energia = T(0.0)
    sum_u1 = T(0.0)
    sum_u2 = T(0.0)

    #
    # Calcula a deformação em cada ponto de Sobol
    #
    for p in 1:N_pts

        # Recupera as coordenadas do ponto
        y1, y2 = pontos[p, 1], pontos[p, 2]

        # offset para a posição central do treino do ponto atual
        offset = (p - 1) * 5
        
        # Recupera os deslocamentos da saída da rede nos 5 pontos do stencil 
        # Centro
        u1_C  = AL[1, offset + 1]
        u2_C  = AL[2, offset + 1]

        # Leste (y1​+h)
        u1_E  = AL[1, offset + 2] 
        u2_E  = AL[2, offset + 2]

        # Oeste (y1​−h)
        u1_W  = AL[1, offset + 3] 
        u2_W  = AL[2, offset + 3]

        # Norte (y2​+h)
        u1_N  = AL[1, offset + 4] 
        u2_N  = AL[2, offset + 4]

        # Sul (y2​−h)
        u1_S  = AL[1, offset + 5] 
        u2_S  = AL[2, offset + 5]
        
        # Diferenças finitas centrais do gradiente de deslocamentos
        du1_dy1 = (u1_E - u1_W) / (T(2.0) * h)
        du1_dy2 = (u1_N - u1_S) / (T(2.0) * h)
        du2_dy1 = (u2_E - u2_W) / (T(2.0) * h)
        du2_dy2 = (u2_N - u2_S) / (T(2.0) * h)
        
        # Deformações globais
        ε11 = ε_macro[1,1] + du1_dy1
        ε22 = ε_macro[2,2] + du2_dy2
        ε12 = ε_macro[1,2] + T(0.5) * (du1_dy2 + du2_dy1)

        # Monta na notação de Voigt
        ε_vec = [ε11, ε22, T(2.0) * ε12]
        
        # Recupera as propriedades do material do ponto (central)
        E_C, ν_C = Propriedades_Material(y1, y2, mat_params)

        # Monta o tensor no ponto 
        C = Matriz_Constitutiva(E_C, ν_C)
        
        # Densidade de energia (0.5 * ε^T * C * ε)
        L_energia += T(0.5) * dot(ε_vec, C * ε_vec)
        
        # Soma os deslocamentos para já calcularmos a 
        # restrição
        sum_u1 += u1_C
        sum_u2 += u2_C

    end

    # Estimativas por Monte Carlo
    L_energia = L_energia / N_pts
    L_avg = (sum_u1 / N_pts)^2 + (sum_u2 / N_pts)^2
    
    # Retorna a perda total da PINN
    return L_energia +  λ_avg * L_avg

end

# -----------------------------------------------------------------------------
#  Calcula as deformações para a parte do pós-processamento do C^H
# -----------------------------------------------------------------------------
function Calcula_Deformacoes_pos_treino(y1::T, y2::T, ε_macro::Matrix{T}, rede::Rede{T},
                                        N_modos::Int; h=T(1e-5)) where {T<:AbstractFloat}

    # Perturba em y1
    u_x_mais  = vec(Forward_Rede(rede, reshape(Camada_Periodica(y1 + h, y2, N_modos), :, 1))[end])
    u_x_menos = vec(Forward_Rede(rede, reshape(Camada_Periodica(y1 - h, y2, N_modos), :, 1))[end])
    du_dy1 = (u_x_mais .- u_x_menos) ./ (T(2.0) * h)
    
    # Perturba em y2
    u_y_mais  = vec(Forward_Rede(rede, reshape(Camada_Periodica(y1, y2 + h, N_modos), :, 1))[end])
    u_y_menos = vec(Forward_Rede(rede, reshape(Camada_Periodica(y1, y2 - h, N_modos), :, 1))[end])
    du_dy2 = (u_y_mais .- u_y_menos) ./ (T(2.0) * h)
    
    # Calcula as deformações
    ε11 = ε_macro[1,1] + du_dy1[1]
    ε22 = ε_macro[2,2] + du_dy2[2]
    ε12 = ε_macro[1,2] + T(0.5) * (du_dy2[1] + du_dy1[2])
    
    # Retorna no formato de Voigt
    return [ε11, ε22, ε12]

end

# =============================================================================
#  Treino da rede
# =============================================================================

function Treina_Rede_PINN_Energia!(rede::Rede{T}, pontos::Matrix{T}, 
                                   ε_macro::Matrix{T}, N_modos::Int, 
                                   mat_params::NamedTuple; 
                                   η = T(0.005), epochs = 1000, λ_decay = T(1e-4),
                                   N_SHOW = 50, λ_avg = T(100.0),
                                   β1 = T(0.9), β2 = T(0.999), ϵ = T(1e-8),
                                   verbose = true) where {T<:AbstractFloat}

    # Aloca histórico da perda
    historico = T[]

    # Número de camadas da rede
    L = length(rede.camadas)

    # Número de pontos para o Sobol
    N_pts = size(pontos, 1)

    # Perturbação para DF
    h = T(1e-4)

    # Agora vem a parte "central" da vetorização da PINN. Montar todos os 
    # lotes de treino (pontos de Sobol) juntos. Isso vai gerar uma matriz 
    # GIGANTE 
    #
    #    X_all =  4N × 5*N_pontos 
    #
    # onde N é o número de modos e N_pontos é o número de pontos de Sobol.
    #
    # As linhas de uma coluna vão ter os valores de 
    # 
    #   sin(2\pi 1 y1)  
    #   cos(2\pi 1 y1)
    #   sin(2\pi 1 y2)
    #   cos(2\pi 1 y2)
    #         ... 
    #   sin(2\pi N y1)
    #   cos(2\pi N y1)
    #   sin(2\pi N y2)
    #   cos(2\pi N y2)
    #
    # e como temos o stencil de 5 pontos para calcular as derivadas, já aproveitamos e 
    # já geramos a coluna central (essa que está ali em cima) mais as 4 colunas associadas
    # às perturbações Leste, Oeste, Norte e Sul. Para acessar um ponto central temos que usar
    # um acesso parecido com o que fazemos em FEM, 
    #
    # coluna_pto_central_p = 5(p-1) + 1 
    # 
    # e os ptos em torno do central são 
    #
    # coluna_pto_LESTE_p = 5(p-1) + 2 
    # coluna_pto_OESTE_p = 5(p-1) + 3 
    # coluna_pto_NORTE_p = 5(p-1) + 4 
    # coluna_pto_SUL_p = 5(p-1) + 5 
    #
    #

    # Malha com  5 pontos: Centro, Leste, Oeste, Norte, Sul
    # Vamos montar com push! no loop abaixo 
    X_list = Matrix{T}[]

    # Coordenadas dos pontos do stencil, aplicando a camada periódica
    for p in 1:N_pts

        # Ponto central
        y1, y2 = pontos[p, 1], pontos[p, 2]

        # Lista com o stencil 
        coords = [(y1, y2), (y1+h, y2), (y1-h, y2), (y1, y2+h), (y1, y2-h)]

        # Para cada ponto do stencil aplicamos a camada periódica
        for pt in coords
            push!(X_list, reshape(Camada_Periodica(pt[1], pt[2], N_modos), :, 1))
        end

    end

    # Concatenamos horizontalmente para ficarmos com uma matriz só
    X_all = hcat(X_list...)

    # Inicializa os pesos e os bias para fazer as atualizações 
    # do AdamW
    mW = [zeros(T, size(c.W)) for c in rede.camadas]
    vW = [zeros(T, size(c.W)) for c in rede.camadas] 
    mb = [zeros(T, size(c.b)) for c in rede.camadas]
    vb = [zeros(T, size(c.b)) for c in rede.camadas]
    
    # Adicione este parâmetro na assinatura da função: λ_decay = T(1e-4)
    # λ_decay controla a força do decaimento de peso desacoplado.
    
    # Loop de treino
    for iter in 1:epochs

        # Propaga a rede para frente
        As = Forward_Rede(rede, X_all)

        # Saídas da ultima camada da rede
        AL = As[end]
        
        # Grande manha do Zygote!!!
        custo, back = Zygote.pullback(al -> Perda_Energia_Alvo(al, pontos, ε_macro, mat_params, λ_avg), AL)

        # Sensibilidade da perda em relação à ultima camada (saída da rede)
        dL_dAL = back(T(1.0))[1]

        # Já guardamos o custo aqui
        push!(historico, custo)
        
        # Calcula o resto dos gradientes "manualmente" (estes são os gradientes puros da função de perda)
        ∇W, ∇b = Backward_Rede(rede, As, dL_dAL)
        
        # Otimização AdamW camada a camada
        for i in 1:L

            # recupera os gradientes puros da camada 
            gW = ∇W[i]
            gb = ∇b[i]
            
            # Estimativa dos momentos SOMENTE com o gradiente da perda (desacoplado)
            mW[i] .= β1 .* mW[i] .+ (1 - β1) .* gW
            vW[i] .= β2 .* vW[i] .+ (1 - β2) .* (gW .^ 2)

            # Correção de viés para os pesos
            mhatW = mW[i] ./ (1 - β1^iter)
            vhatW = vW[i] ./ (1 - β2^iter)
            
            # Atualização AdamW: O passo é dado pela estimativa adaptativa MAIS o decaimento linear da matriz W
            rede.camadas[i].W .-= η .* (mhatW ./ (sqrt.(vhatW) .+ ϵ) .+ λ_decay .* rede.camadas[i].W)
            
            # Estimativa e correção para os bias
            mb[i] .= β1 .* mb[i] .+ (1 - β1) .* gb
            vb[i] .= β2 .* vb[i] .+ (1 - β2) .* (gb .^ 2)
            mhatb = mb[i] ./ (1 - β1^iter)
            vhatb = vb[i] ./ (1 - β2^iter)
            
            # O decaimento de peso em bias costuma ser evitado na literatura, mas vamos 
            # colocar a forma completa
            rede.camadas[i].b .-= η .* (mhatb ./ (sqrt.(vhatb) .+ ϵ) .+ λ_decay .* rede.camadas[i].b)

        end
        
        # Mostra o resultado atual 
        if verbose && (iter == 1 || iter % max(1, iter ÷ N_SHOW) == 0)
            println("Iteração ", iter, "    energia = ", custo)
        end
    end
    
    # Retorna o histórico do objetivo
    return historico

end

# -----------------------------------------------------------------------------
#  Integração final de propriedades
# -----------------------------------------------------------------------------
function Calcula_Tensor_Homogeneizado(redes::Vector{Rede{T}}, modos::Vector{Matrix{T}}, 
                                      N_modos::Int, mat_params::NamedTuple, N_eval::Int=50) where {T<:AbstractFloat}


    # Aloca a matriz 
    CH = zeros(T, 3, 3)

    # Gera N_eval pontos entre 0 e 1 para calcular a integral 
    # na célula 
    ys  = range(T(0.0), T(1.0), length=N_eval)
    pts = [(y1, y2) for y1 in ys for y2 in ys]

    # Número de pontos total 
    N_pts = length(pts)

    # Deformações 
    deformacoes = [Vector{Vector{T}}(undef, N_pts) for _ in 1:3]
    
    # Loop pelas redes (k) e pelos pontos (p)
    for k in 1:3
        for p in 1:N_pts

            # Coordenadas do ponto 
            y1, y2 = pts[p][1], pts[p][2]

            # Deformação do ponto 
            ε_vec = Calcula_Deformacoes_pos_treino(y1, y2, modos[k], redes[k], N_modos)

            # Monta em formato de Voigt
            deformacoes[k][p] = [ε_vec[1], ε_vec[2], T(2.0)*ε_vec[3]]
        end
    end

    # Aloca as tensões 
    tensoes = [Vector{T}(undef, 3) for _ in 1:3]

    # Loop pelas redes (k) e pelos pontos (p)
    for p in 1:N_pts

        # Recupera coordenadas do ponto 
        y1, y2 = pts[p][1], pts[p][2]

        # Propriedades do material no ponto 
        E, ν = Propriedades_Material(y1, y2, mat_params)

        # Matriz constitutiva
        C_local = Matriz_Constitutiva(E, ν)
        
        # Tensões para cada rede 
        for i in 1:3
            tensoes[i] = C_local * deformacoes[i][p]
        end
        
        # Acumula a matriz homogeneizada
        for i in 1:3, j in 1:3
            CH[i, j] += dot(deformacoes[i][p], tensoes[j])
        end
    end
    
    # Calcula a média 
    CH ./= N_pts

    # E retorna o valor médio 
    return CH

end

# =============================================================================
#  Validação
# =============================================================================
function Testa_Homogenizacao_DRM()

    # Propriedades dos materiais da célula
    # no caso, fibra e matriz
    # raio e centro da fibra
    mat_params = (
        E_m = 1.0, ν_m = 0.3,
        E_f = 10.0, ν_f = 0.3,
        r0 = 0.25, yc1 = 0.5, yc2 = 0.5
    )

    # Número de modos para a camada periódica
    N_modos_fourier = 4

    # Modos fundamentais para cada caso de "carga" 
    # da homogeneização
    ε_1 = [1.0f0 0.0f0; 0.0f0 0.0f0]
    ε_2 = [0.0f0 0.0f0; 0.0f0 1.0f0]
    ε_3 = [0.0f0 0.5f0; 0.5f0 0.0f0]
    modos = [ε_1, ε_2, ε_3]

    # Amostragem QMC
    N_colocacao = 1000 
    pontos = Gera_Pontos_Sobol(N_colocacao, Float32)

    # Vetor de redes para guardar cada rede (são 3) que vamos treinar
    redes_treinadas = Rede{Float32}[]

    # Vetor de vetores para guardar os históricos (são 3)
    historicos_treinados = Vector{Float32}[]

    #
    # Loop pelas redes
    #
    for k in 1:3

        # Avisa que vamos treinar a rede k 
        println("\n Treinando Modo ", k)

        # Inicializa a rede
        rede = Inicializa_Rede([16, 20, 20, 2], [TANH_GEN, TANH_GEN, LINEAR_GEN], Float32)
        
        # Treina a rede
        hist = Treina_Rede_PINN_Energia!(rede, pontos, modos[k], N_modos_fourier, mat_params;
                                  η = 0.005, epochs = 1000, N_SHOW = 100, λ_avg = 100.0f0)

        # Guarda a rede no vetor de redes para fazermos o pós-processamento depois 
        push!(redes_treinadas, rede)

        # Guarda os valores do treino da rede
        push!(historicos_treinados, hist)

    end

    # Agora vamos calcular o tensor homogeneizado
    println("\n Calculando Tensor Homogeneizado ")
    CH = Calcula_Tensor_Homogeneizado(redes_treinadas, modos, N_modos_fourier, mat_params, 50)
    display(CH)
    
    # Retorna a matriz homogeneizada e os históricos
    return CH, historicos_treinados

end

# Testa ....
CH, historicos_treinados = Testa_Homogenizacao_DRM()