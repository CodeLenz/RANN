# Função de otimização da rede neural
# Método Adam
# Parâmetros de otimização: 
# α (Learning rate, default = 0.01) 
# β1 e β2 (Taxas de decaimento, default 0.9 e 0.999)
# ϵ (default 10^-8) 
# nepoch (Número de épocas, default = 10)
# δ (critério de convergência, default = 1E-8)
function AdamW(rede::Rede, treino::Treino, nepoch::Int64; α = 1E-3, β1 = 0.9, β2 = 0.999,
              ϵ = 1E-8, w_decay = 0.0, conv = 1E-8)
              
    # Aloca objetivos
    obj_treino = 0.0
    perda_inicial_u = 0.0
    perda_inicial_du = 0.0
    perda_fisica = 0.0

    dobj_treino = 0.0 
    # Acessa os termos em Treino por apelidos
    t_inicial = treino.t_inicial
    u_inicial = treino.u_inicial
    du_inicial = treino.du_inicial
    t_fisica =  treino.t_fisica

    # Copia rede.x para uma outra memória
    x = copy(rede.x)

    # Obtém o número de dados de treino
    n_fisica = size(t_fisica, 2)

    # Aloca um array para monitorar o objetivo
    vetor_obj_treino = zeros(nepoch)
    vetor_perda_inicial_u = zeros(nepoch)
    vetor_perda_inicial_du = zeros(nepoch)
    vetor_perda_fisica = zeros(nepoch)

    # Aloca a resposta estimada para os pontos de teste
    u_test_pred = zeros(1, size(treino.u_an, 2))

    # Aloca vetor gradiente - vazio, valores serão inputados posteriormente
    G = Vector{Float64}(undef,length(x))

    # Aloca vetores de primeiro (m) e segundo (v) momento do otimizador Adam
    m = zeros(Float64, rede.n_projeto)
    v = zeros(Float64, rede.n_projeto)

    # Loop de otimização pelas épocas
    @showprogress "Otimizando..." for epoch = 1:nepoch

        # Calcula o objetivo da rede para o treino 
        obj_treino = ObjetivoFloat(rede, treino, t_inicial, u_inicial, du_inicial, n_fisica, t_fisica, epoch, x)

        # Testa se a otimização convergiu ao longo das iterações
        if obj_treino <= conv

            println("Convergiu em $epoch épocas")
            break

        end
        
        # Zera o vetor gradiente para novo cálculo
        fill!(G, 0.0)

        # Derivada automática em relação a x (pesos e bias) com o Enzyme
        Enzyme.autodiff(
            Enzyme.Reverse,
            ObjetivoFloat,
            Const(rede),
            Const(treino),
            Const(t_inicial),
            Const(u_inicial),
            Const(du_inicial),
            Const(n_fisica),
            Const(t_fisica),
            Const(epoch),
            Duplicated(x, G)
            )

        # Calcula o objetivo da rede para o treino 
        obj_treino, perda_inicial_u, perda_inicial_du, perda_fisica  = 
            Objetivo(rede, treino, t_inicial, u_inicial, du_inicial, n_fisica, t_fisica, epoch, x)

        # Armazena o objetivo
        vetor_obj_treino[epoch] = obj_treino
        vetor_perda_inicial_u[epoch] = perda_inicial_u
        vetor_perda_inicial_du[epoch] = perda_inicial_du
        vetor_perda_fisica[epoch] = perda_fisica

        # AdamW
        # Atualiza as variáveis de projeto antes dos passos tradicionais do Adam
        x .= x .- α*w_decay*x

        #
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

        # A cada 1000 epochs vamos monitorar o comportamento da rede 
        if (epoch % 1000 == 0) || (epoch == nepoch)

            # Obtém a resposta da rede neural para os pontos de teste
            u_test_pred = Deslocamento_Teste(rede, x, treino.u_an, treino.t_teste, vetor_obj_treino, vetor_perda_inicial_u,
                                             vetor_perda_inicial_du, vetor_perda_fisica, epoch)

        end

    end

    # Retorna as variáveis de projeto e a função objetivo ao longo do tempo
    return x, vetor_obj_treino, u_test_pred

end