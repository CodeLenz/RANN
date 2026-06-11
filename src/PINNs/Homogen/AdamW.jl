# Otimizador ADAM
function AdamW!(obj_fn::Function, rede::Rede{T}, X::Matrix{T}, Z_buffers::Vector{Matrix{T}}, 
                As::Vector{Matrix{T}}, historico::Vector{T}, historico_energia::Vector{T},
                historico_avg::Vector{T}, η::T, epochs::Int, 
                λ_decay = T(1e-4), N_SHOW = 50, β1 = T(0.9), β2 = T(0.999), ϵ = T(1e-8),
                verbose = true) where {T<:AbstractFloat}

    # Número de camadas da rede
    L = length(rede.camadas)

    # Inicializa os pesos e os bias para fazer as atualizações do AdamW
    mW = [zeros(T, size(c.W)) for c in rede.camadas]
    vW = [zeros(T, size(c.W)) for c in rede.camadas] 
    mb = [zeros(T, size(c.b)) for c in rede.camadas]
    vb = [zeros(T, size(c.b)) for c in rede.camadas]
              
    # Loop de treino
    for iter in 1:epochs

        # Propaga a rede para frente In-Place (Zero Alocação para Z e A)
        Forward_Rede_InPlace!(rede, X, Z_buffers, As)

        # Saídas da ultima camada da rede
        AL = As[end]

        # Chama função objetivo e recupera termos de custo e gradiente 
        # Nesse caso, gradiente é a sensibilidade da perda em relação à ultima camada (saída da rede)
        custo, L_energia, L_avg, dL_dAL = obj_fn(AL)

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

end