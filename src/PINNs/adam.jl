# Função de otimização da rede neural
# Método Adam
# Parâmetros de otimização: 
# α (Learning rate, default = 0.01) 
# β1 e β2 (Taxas de decaimento, default 0.9 e 0.999)
# ϵ (default 10^-8) 
# nepoch (Número de épocas, default = 10)
# δ (critério de convergência, default = 1E-8)
function Adam(rede::Rede, treino::Treino; α = 1E-3, β1 = 0.9, β2 = 0.999,
              ϵ = 1E-8, nepoch = 15_000, conv = 1E-8)
              
    # Aloca objetivo
    obj_treino = 0.0

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

    # Aloca vetor gradiente - vazio, valores serão inputados posteriormente
    G = Vector{Float64}(undef,length(x))

    # Aloca vetores de primeiro (m) e segundo (v) momento do otimizador Adam
    m = zeros(Float64, rede.n_projeto)
    v = zeros(Float64, rede.n_projeto)

    # Loop de otimização pelas épocas
    @showprogress "Otimizando..." for t = 1:nepoch

        # Calcula o objetivo da rede para o treino 
        obj_treino = Objetivo(rede, treino, t_inicial, u_inicial, du_inicial, n_fisica, t_fisica, x, t)

        # Armazena o objetivo
        vetor_obj_treino[t] = obj_treino

        # Testa se a otimização convergiu ao longo das iterações
        if obj_treino <= conv

            println("Convergiu em $t épocas")
            break

        end
        
        # Zera o vetor gradiente para novo cálculo
        fill!(G, 0.0)

        # Derivada automática em relação a x (pesos e bias) com o Enzyme
        Enzyme.autodiff(
            Enzyme.Reverse,
            Objetivo,
            Const(rede),
            Const(treino),
            Const(t_inicial),
            Const(u_inicial),
            Const(du_inicial),
            Const(n_fisica),
            Const(t_fisica),
            Duplicated(x, G),
            Const(t)
            )

        # Atualiza os momentos
        m .= β1 * m .+ (1 - β1) * G
        v .= β2 * v .+ (1 - β2) * G .* G 

        # Corrige o bias dos momentos devido a inicialização com zeros
        # Diretamente pela correção do alpha
        α_t = α * sqrt(1 - β2^t) / (1 - β1^t)

        # Atualiza as variáveis de projeto
        x .= x .- α_t * m ./ (v.^(1/2) .+ ϵ)

        # A cada 100 epochs vamos monitorar o comportamento da rede 
        if t % 100 == 0

            @show "Atingimos a iteração $t"

            # Agora vamos calcular a resposta em cada tempo 
            u_test_pred = zeros(1, size(treino.u_an,2))

            # Desenrola os pesos e bias 
            pesos, bias = Atualiza_pesos_bias(rede, x)

            # Obtém a resposta da rede neural para os pontos de teste
            for j=1:size(treino.u_an,2)
                t = treino.t_teste[1,j]
                u_test_pred[:, j] = RNA(rede, pesos, bias, [t])
            end

            # Grava em um arquivo para monitoramento 
            writedlm("teste_rede_$(t).txt",u_test_pred)

        end

    end

    # Retorna as variáveis de projeto e a função objetivo ao longo do tempo
    return x, vetor_obj_treino

end