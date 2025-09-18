# Função de otimização da rede neural
# Método Adam
# Parâmetros de otimização: 
# α (Learning rate, default = 0.01) 
# β1 e β2 (Taxas de decaimento, default 0.9 e 0.999)
# ϵ (default 10^-8) 
# nepoch (Número de épocas, default = 10)
# δ (critério de convergência, default = 1E-8)
# mbs (mini-batch size, default = 100)
function Adam(rede::Rede, treino::Treino, α = 0.01, β1 = 0.9, β2 = 0.999,
              ϵ = 1E-8, nepoch = 15_000, δ = 1E-8)
              
    # Aloca objetivo
    obj_treino = 0.0

    # Acessa os termos em Treino por apelidos
    t_contorno_ess = treino.t_contorno_ess
    u_contorno_ess = treino.u_contorno_ess
    t_contorno_nat = treino.t_contorno_nat
    du_contorno_nat = treino.du_contorno_nat
    t_fisica =  treino.t_fisica
    t_teste = treino.t_teste
    x = rede.x

    # Obtém o número de dados de treino e teste
    n_fisica = size(t_fisica, 2)

    # Aloca um array para monitorar o objetivo
    vetor_obj_treino = zeros(nepoch)

    # Aloca vetor gradiente
    G = zeros(length(x))

    # Aloca vetores de primeiro (m) e segundo (v) momento do otimizador Adam
    m = zeros(Float64, rede.n_projeto)
    v = zeros(Float64, rede.n_projeto)

    # Loop de otimização pelas épocas
    @showprogress "Otimizando..." for t = 1:nepoch

        # Calcula o objetivo da rede para o treino 
        obj_treino = Objetivo(rede, treino, t_contorno_ess, u_contorno_ess, t_contorno_nat, du_contorno_nat, n_fisica, t_fisica, x)

        # Armazena o objetivo
        vetor_obj_treino[t] = obj_treino

        # Testa se a otimização convergiu ao longo das iterações
        if obj_treino <= δ

            println("Convergiu em $t épocas")
            break

        end
            
        # Define função objetivo em função das variáveis de projeto para a diferenciação
        f(rede, treino, t_contorno_ess, u_contorno_ess, t_contorno_nat, du_contorno_nat, n_fisica, t_fisica, x) = 
            Objetivo(rede, treino, t_contorno_ess, u_contorno_ess, t_contorno_nat, du_contorno_nat, n_fisica, t_fisica, x)

        # Zera o vetor gradiente para novo cálculo
        G .= 0.0

        # Derivada automática em relação a x (pesos e bias) com o Enzyme
        Enzyme.autodiff(
            Enzyme.Reverse,
            f,
            Const(rede),
            Const(treino),
            Const(t_contorno_ess),
            Const(u_contorno_ess),
            Const(t_contorno_nat),
            Const(du_contorno_nat),
            Const(n_fisica),
            Const(t_fisica),
            Duplicated(x, G)
            )

        # Atualiza os momentos
        m .= β1 * m .+ (1 - β1) * G
        v .= β2 * v .+ (1 - β2) * G .* G 

        # Corrige o bias dos momentos devido a inicialização com zeros
        # Diretamente pela correção do alpha
        α_t = α * sqrt(1 - β2^t) / (1 - β1^t)

        # Atualiza as variáveis de projeto
        x = x .- α_t * m ./ (v.^(1/2) .+ ϵ)

    end

    # Retorna as variáveis de projeto e a função objetivo ao longo do tempo
    return x, vetor_obj_treino

end