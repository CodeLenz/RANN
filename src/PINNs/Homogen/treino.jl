# =============================================================================
#  Treino da rede
# =============================================================================

function Treina_Rede_PINN_Energia!(rede::Rede{T}, pontos::Matrix{T}, ε_macro::Matrix{T},
                                   N_modos::Int, mat_params::NamedTuple; 
                                   η = T(0.005), epochs = 1000, λ_avg = T(100.0),
                                   verbose = true) where {T<:AbstractFloat}

    # Aloca histórico da perda
    historico, historico_energia, historico_avg = T[], T[], T[]

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
   
    # Pré-aloca o histórico de ativações para o backward
    As = [Matrix{T}(undef, size(rede.camadas[1].W, 2), N_pts * 5)]
    for c in rede.camadas
        push!(As, Matrix{T}(undef, size(c.W, 1), N_pts * 5))
    end

    # Pré-aloca os rascunhos de Z apenas para  usar no forward
    Z_buffers = [Matrix{T}(undef, size(c.W, 1), N_pts * 5) for c in rede.camadas]

    # Define wrappers para a função objetivo e gradiente, para facilitar a passagem de argumentos
    # Função objetivo
    # Zygote.pullback calcula a função custo, e também realiza o pullback em relação a AL
    # A segunda chamada (sem gradiente) recupera os termos de monitoramento de perda sem custo de AD
    obj_fn(AL) = begin

        custo, back = Zygote.pullback(al -> Perda_Energia_Alvo(al, pontos, ε_macro, mat_params, λ_avg)[1], AL)
        dL_dAL = back(T(1.0))[1]
        _, L_energia, L_avg = Perda_Energia_Alvo(AL, pontos, ε_macro, mat_params, λ_avg)

        return custo, L_energia, L_avg, dL_dAL

    end
    
    # Chama o otimizador AdamW (in-place)
    AdamW!(obj_fn, rede, X_all, Z_buffers, As, historico, historico_energia, historico_avg, η, epochs)
    
    # Retorna o histórico do objetivo
    return historico, historico_energia, historico_avg

end