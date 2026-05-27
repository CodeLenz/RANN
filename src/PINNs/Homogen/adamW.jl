# Função de otimização da rede neural
# Método Adam
# Parâmetros de otimização: 
# α (Learning rate, default = 0.01) 
# β1 e β2 (Taxas de decaimento, default 0.9 e 0.999)
# ϵ (default 10^-8) 
# nepoch (Número de épocas, default = 10)
# δ (critério de convergência, default = 1E-8)
function AdamW(obj_fn::Function, grad_fn!::Function, x::Vector{Float64}, rede::Rede, treino::NamedTuple,
               nepoch::Int64, prob::String; α = 1E-3, β1 = 0.9, β2 = 0.999, ϵ = 1E-8, 
               w_decay = 0.0, conv = 1E-8, otimizador = "AdamW", subsample = false)
              
    # Intervalo para gerar as respostas (acompanhamento)
    intervalo_monitor = 1000

    # Aloca objetivo
    obj_treino = 0.0

    # Aloca um vetor para monitorar objetivo
    vetor_obj_treino = zeros(nepoch)

    # Aloca a resposta estimada para os pontos de teste
    u_test_pred = zeros(1, size(treino.teste, 2))

    # Aloca vetor gradiente - vazio, valores serão inputados posteriormente
    G = Vector{Float64}(undef, length(x))

    # Aloca conjunto de pontos de treino para o AdamW
    if subsample

        pontos_treino_fisica = Matrix{Float64}(undef, size(treino.fisica, 1), round(Int, size(treino.fisica, 2) * 0.3))

    else

        pontos_treino_fisica = Matrix{Float64}(undef, size(treino.fisica))

    end

    # Aloca vetores de primeiro (m) e segundo (v) momento do otimizador Adam
    m = zeros(Float64, rede.n_projeto)
    v = zeros(Float64, rede.n_projeto)

    # Loop de otimização pelas épocas
    @showprogress "Otimizando com AdamW..." for epoch = 1:nepoch

        # Avalia se irá usar o conjunto completo ou um subconjunto de pontos de treino para esta época
        if subsample
            
            # Seleciona um subconjunto de pontos de colocação para um intervalo de épocas
            # Subamostra 30% dos pontos de treino físico a cada 100 épocas
            if epoch % 100 == 0 || epoch == 1
            
                # Define amostral de treino para esta época
                pontos_treino_fisica .= subsample_fisica(treino.fisica, round(Int, size(treino.fisica, 2) * 0.3)) 

            end

        else

            # Usa o conjunto completo de pontos de treino físico
            pontos_treino_fisica .= treino.fisica

        end

        # Calcula o objetivo da rede para o treino 
        obj_treino = obj_fn(x, pontos_treino_fisica)

        # Testa se a otimização convergiu ao longo das iterações
        if obj_treino <= conv

            println("Convergiu em $epoch épocas")
            break

        end
        
        # Realiza a derivação automática com o Enzyme
        grad_fn!(G, x, pontos_treino_fisica)

        # Calcula o objetivo da rede para o treino 
        obj_treino  = obj_fn(x, pontos_treino_fisica)

        # Armazena o objetivo
        vetor_obj_treino[epoch] = obj_treino

        # AdamW
        # Atualiza as variáveis de projeto antes dos passos tradicionais do Adam
        x .= x .- α * w_decay * x

        # Atualiza os momentos        
        # Primeiro momento (média)
        m .= β1 * m .+ (1 - β1) * G

        # Segundo momento (variância)
        v .= β2 * v .+ (1 - β2) * G .* G 

        # Corrige o bias dos momentos devido a inicialização com zeros
        # Diretamente pela correção do alpha
        α_t = α * sqrt(1 - β2 ^ epoch) / (1 - β1 ^ epoch)

        # Atualiza as variáveis de projeto
        x .= x .- α_t * m ./ (v.^(1/2) .+ ϵ)

        # A cada <intervalo_monitor> epochs vamos monitorar o comportamento da rede 
        if (epoch % intervalo_monitor == 0) || (epoch == nepoch)

            # Obtém a resposta da rede neural para os pontos de teste e gera gráficos para monitoramento
            u_test_pred = Resposta_Teste(rede, x, treino, vetor_obj_treino, epoch, 
                                         prob, otimizador, intervalo_monitor)

        end

    end

    # Retorna as variáveis de projeto e a função objetivo ao longo do tempo
    return x, vetor_obj_treino, u_test_pred

end