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
              ϵ = 1E-8, nepoch = 8, δ = 1E-8, mbs = 100)
              
    # Aloca objetivo
    obj_treino = 0.0
    obj_teste = 0.0

    # Acessa os termos em Treino por apelidos
    entradas_treino         = treino.entradas_treino
    saidas_esperadas_treino = treino.saidas_esperadas_treino
    entradas_teste          =  treino.entradas_teste
    saidas_esperadas_teste = treino.saidas_esperadas_teste
    x = rede.x

    # Obtém o número de dados de treino e teste
    n_treinos = size(entradas_treino, 2)
    n_testes = size(entradas_teste, 2)

    # Calcula o número de iterações em uma época
    iter = Int(ceil(n_treinos / mbs))

    # Aloca um array para monitorar o objetivo
    vetor_obj_treino = zeros(nepoch * iter)
    vetor_obj_teste = zeros(nepoch)

    # Aloca vetores de primeiro (m) e segundo (v) momento do otimizador Adam
    m = zeros(Float64, rede.n_projeto)
    v = zeros(Float64, rede.n_projeto)

    # Loop de otimização pelas épocas
    @showprogress "Otimizando..." for t = 1:nepoch

        # Gera um seed novo para cada época
        # Garante reproducibilidade no código mas mantém aleatoriedade entre épocas
        Random.seed!(1234+t)

        # Calcula o objetivo da rede para o teste
        obj_teste = Objetivo(rede, entradas_teste, saidas_esperadas_teste, n_testes, x)

        # Armazena o objetivo
        vetor_obj_teste[t] = obj_teste

        # Altera a ordem das colunas das entradas aleatoriamente (shuffle) 
        # Necessário para que os mini batches não tenham nenhum viés e a otimização seja estocástica
        permut = shuffle(1:n_treinos)
        entradas_treino_shuffle = entradas_treino[:, permut]
        saidas_esperadas_treino_shuffle = saidas_esperadas_treino[:, permut]

        # Loop pelas iterações entre os mini batches
        for i = 1:iter

            # Calcula índice para buscar as colunas do mini batch
            id = (i - 1) * mbs + 1

            # Seleciona dados de treino para o mini batch
            entradas_treino_iter = entradas_treino_shuffle[:, id:(i*mbs)]
            saidas_esperadas_treino_iter = saidas_esperadas_treino_shuffle[:, id:(i*mbs)]

            # Calcula o objetivo da rede para o treino 
            obj_treino = Objetivo(rede, entradas_treino_iter, saidas_esperadas_treino_iter, mbs, x)

            # Armazena o objetivo
            vetor_obj_treino[(t-1) * iter + i] = obj_treino

            # Testa se a otimização convergiu ao longo das iterações
            if obj_treino <= δ

                println("Convergiu em $i iterações")
                break

            end
            
            # Define função objetivo em função das variáveis de projeto para a diferenciação
            f(rede, entradas_treino_iter, saidas_esperadas_treino_iter, mbs, x) = Objetivo(rede, entradas_treino_iter, saidas_esperadas_treino_iter, mbs, x)

            # Aloca vetor gradiente
            G = zeros(length(x))

            # Derivada automática em relação a x (pesos e bias) com o Enzyme
            #f, G = value_and_gradient(f, AutoEnzyme(; function_annotation=Enzyme.Duplicated), x)
            Enzyme.autodiff(
                Enzyme.set_runtime_activity(Enzyme.Reverse),
                f,
                Const(rede),
                Const(entradas_treino_iter),
                Const(saidas_esperadas_treino_iter),
                Const(mbs),
                Duplicated(x, G),
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

    end

    # Retorna as variáveis de projeto e a função objetivo ao longo do tempo
    return x, vetor_obj_treino, vetor_obj_teste

end