# =============================================================================
#  Treino da rede
# =============================================================================

function Treina_Rede_PINN_Energia!(rede::Rede{T}, pontos::Matrix{T}, ε_macro::Matrix{T},
                                   N_modos::Int, mat_params::NamedTuple; 
                                   η = T(0.005), epochs_ADAM = 1000, epochs_LBFGS = 1000, λ_avg = T(100.0),
                                   verbose = true) where {T<:AbstractFloat}

    # Aloca histórico da perda
    historico_ADAM, historico_energia_ADAM, historico_avg_ADAM = T[], T[], T[]
    historico_LBFGS, historico_energia_LBFGS, historico_avg_LBFGS = T[], T[], T[]

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

    # Pré-aloca buffers de W e b para reconstrução dos pesos e bias no L-BFGS
    W_buffer = [similar(c.W) for c in rede.camadas]
    b_buffer = [similar(c.b) for c in rede.camadas]

    # Define wrappers para a função objetivo e gradiente, para facilitar a passagem de argumentos
    # Função objetivo
    obj_fn(W, b) = begin

        # Propaga a rede para frente In-Place (Zero Alocação para Z e A)
        Forward_Rede_InPlace!(W, b, rede, X_all, Z_buffers, As)

        # Saídas da ultima camada da rede
        AL = As[end]

        # Zygote.pullback calcula a função custo, e também realiza o pullback em relação a AL
        # Nesse caso, gradiente é a sensibilidade da perda em relação à ultima camada (saída da rede)
        # A segunda chamada (sem gradiente) recupera os termos de monitoramento de perda sem custo de AD
        custo, back = Zygote.pullback(al -> Perda_Energia_Alvo(al, pontos, ε_macro, mat_params, λ_avg)[1], AL)
        dL_dAL = back(T(1.0))[1]
        _, L_energia, L_avg = Perda_Energia_Alvo(AL, pontos, ε_macro, mat_params, λ_avg)

        # Calcula o resto dos gradientes "manualmente" (estes são os gradientes puros da função de perda)
        ∇W, ∇b = Backward_Rede(W, rede, As, dL_dAL)

        return custo, L_energia, L_avg, ∇W, ∇b

    end

    # Wrapper da função objetivo para o L-BGFS
    # Recebe vetor flat de pesos e bias
    obj_fn_LBFGS(Θ) = begin

        # Reconstrói os pesos e bias a partir do vetor flat
        offset = 0
        for i in eachindex(W_buffer)
            n_W = length(W_buffer[i])
            W_buffer[i] .= reshape(Θ[offset + 1 : offset + n_W], size(W_buffer[i]))
            offset += n_W
        end
        for i in eachindex(b_buffer)
            n_b = length(b_buffer[i])
            b_buffer[i] .= reshape(Θ[offset + 1 : offset + n_b], size(b_buffer[i]))
            offset += n_b
        end

        # Chama a função objetivo normal para calcular o custo e os gradientes
        custo, L_energia, L_avg, ∇W, ∇b = obj_fn(W_buffer, b_buffer)

        # Achata os gradientes de volta para vetor flat
        ∇L = vcat([vec(gW) for gW in ∇W]..., [vec(gb) for gb in ∇b]...)

        return custo, L_energia, L_avg, ∇L

    end
    
    println("\n*********************************")
    println("Iniciando Treino com AdamW...")
    println("*********************************")

    # Chama o otimizador AdamW (in-place)
    AdamW!(obj_fn, rede, historico_ADAM, historico_energia_ADAM, historico_avg_ADAM, η, epochs_ADAM; verbose = false)

    println("\n*********************************")
    println("Continuando Treino com L-BFGS...")
    println("*********************************")

    # Chama o otimizador L-BFGS (in-place)
    L_BFGS!(obj_fn_LBFGS, rede, historico_LBFGS, historico_energia_LBFGS, historico_avg_LBFGS, epochs_LBFGS; verbose = false)
    
    # Retorna o histórico do objetivo
    return historico_ADAM, historico_energia_ADAM, historico_avg_ADAM, 
        historico_LBFGS, historico_energia_LBFGS, historico_avg_LBFGS

end