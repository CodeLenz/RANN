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
    historico_energia = T[]
    historico_avg = T[]

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
   
    # Pré-aloca o histórico de ativações para o backward
    As = [Matrix{T}(undef, size(rede.camadas[1].W, 2), N_pts * 5)]
    for c in rede.camadas
        push!(As, Matrix{T}(undef, size(c.W, 1), N_pts * 5))
    end

    # Pré-aloca os rascunhos de Z apenas para  usar no forward
    Z_buffers = [Matrix{T}(undef, size(c.W, 1), N_pts * 5) for c in rede.camadas]

    # Inicializa os pesos e os bias para fazer as atualizações do AdamW
    mW = [zeros(T, size(c.W)) for c in rede.camadas]
    vW = [zeros(T, size(c.W)) for c in rede.camadas] 
    mb = [zeros(T, size(c.b)) for c in rede.camadas]
    vb = [zeros(T, size(c.b)) for c in rede.camadas]
    
    # Loop de treino
    for iter in 1:epochs

        # Propaga a rede para frente In-Place (Zero Alocação para Z e A)
        Forward_Rede_InPlace!(rede, X_all, Z_buffers, As)

        # Saídas da ultima camada da rede
        AL = As[end]
         
        # Grande manha do Zygote!!!
        result = Zygote.withgradient(al -> Perda_Energia_Alvo(al, pontos, ε_macro, mat_params, λ_avg), AL)

        # Custo total e termos de energia e média
        custo, L_energia, L_avg = result.val

        # Sensibilidade da perda em relação à ultima camada (saída da rede)
        dL_dAL = result.grad[1]

        # Já guardamos o custo aqui
        push!(historico, custo)
        push!(historico_energia, L_energia)
        push!(historico_avg, L_avg)

        # Calcula o resto dos gradientes "manualmente" (estes são os gradientes puros da função de perda)
        ∇W, ∇b = Backward_Rede(rede, As, dL_dAL)

        # Otimização AdamW camada a camada com mutação In-Place
        for i in 1:L

            # recupera os gradientes puros da camada 
            gW = ∇W[i]
            gb = ∇b[i]
            
            # Estimativa dos momentos SOMENTE com o gradiente da perda (desacoplado)
            @. mW[i] = β1 * mW[i] + (1 - β1) * gW
            @. vW[i] = β2 * vW[i] + (1 - β2) * (gW ^ 2)

            # Correção de viés (termos escalares)
            bias1 = 1 - β1^iter
            bias2 = 1 - β2^iter
            
            # Atualização AdamW In-Place: passo adaptativo + decaimento de peso
            @. rede.camadas[i].W -= η * ((mW[i] / bias1) / (sqrt(vW[i] / bias2) + ϵ) + λ_decay * rede.camadas[i].W)
            
            # Estimativa e correção para os bias
            @. mb[i] = β1 * mb[i] + (1 - β1) * gb
            @. vb[i] = β2 * vb[i] + (1 - β2) * (gb ^ 2)
            
            # O decaimento de peso em bias costuma ser evitado na literatura, mas vamos 
            # colocar a forma completa In-Place
            @. rede.camadas[i].b -= η * ((mb[i] / bias1) / (sqrt(vb[i] / bias2) + ϵ) + λ_decay * rede.camadas[i].b)

        end
        
        # Mostra o resultado atual 
        if verbose && (iter == 1 || iter % max(1, iter ÷ N_SHOW) == 0)
            println("Iteração ", iter, "    energia = ", custo)
        end
    end
    
    # Retorna o histórico do objetivo
    return historico, historico_energia, historico_avg

end