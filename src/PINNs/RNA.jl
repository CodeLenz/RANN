# Atualiza os vetores de vetores de pesos e bias utilizando o vetor de variáveis de projeto x
function Atualiza_pesos_bias(rede::Rede, x::Vector{T}) where T

    # Acessa os termos em Rede por apelidos 
    n_camadas = rede.n_camadas
    conexoes = rede.conexoes
    topologia = rede.topologia
    
    # Pesos: Vetor de vetores ("matriz") com os pesos de cada conexão da rede neural
    # Cada vetor representa uma camada oculta
    # Copia os primeiros valores de x na matriz de pesos através das conexões

    # Inicializa a matriz de pesos pela primeira camada, incluindo as variáveis de projeto
    pesos  = [ x[1:conexoes[1]] ]

    # Inicializa contador k que busca os pesos dentro de x
    k = 1 + conexoes[1]
    
    # Loop pelas linhas da matriz (camadas da rede)
    for i = 2:n_camadas

        # Seleciona as variáveis de projeto da camada e aloca em um vetor
        pesos_linha = x[k:(k+conexoes[i]-1)]

        # Atualiza o contador para a próxima camada
        k = k + conexoes[i]

        # Concatena os pesos da camada i no vetor geral de pesos
        pesos = vcat(pesos, [pesos_linha])

    end

    # Bias: Vetor de vetores ("matriz") com os bias de cada neurônio da rede neural
    # Cada linha representa uma camada oculta

    # Passa os ultimos n_neuronios valores de x para a matriz 
    # de bias, usando topologia como guia para saber quantos
    # valores gravamos por linha

    # Inicializa a matriz de bias pela primeira camada, incluindo as variáveis de projeto
    bias = [ x[k:(k+topologia[2]-1)] ]

    # Atualiza o contador
    k = k + topologia[2]

    # Loop pelas linhas da matriz (camadas da rede)
    for i = 2:n_camadas

        # Seleciona as variáveis de projeto da camada e aloca em um vetor
        bias_linha = x[k:(k+topologia[i+1]-1)]

        # Atualiza o contador; já está contando desde os pesos
        k = k + topologia[i+1]

        # Concatena os bias da camada i no vetor geral de pesos
        bias = vcat(bias, [bias_linha])

    end

    # Retorna matrizes de pesos e bias
    return pesos, bias

end


# Rede neural
function RNA(rede::Rede, pesos::Vector{Vector{T}}, bias::Vector{Vector{T}}, 
             entrada_i::Vector{T}) where T

    # Acessa os termos em Rede por apelidos 
    topologia = rede.topologia
    n_camadas = rede.n_camadas
    ativ      = rede.ativ

    # Inicializa sinais    
    # Incluí o vetor de entradas na primeira linha de sinais
    sinais = [entrada_i]

    # Loop pelas camadas
    for c = 2:(n_camadas+1)

        # Recupera a camada anterior de sinais
        camada_anterior = sinais[end]

        # Recupera número de neurônios da camada anterior e da nova camada
        n_in = topologia[c - 1]
        n_out = topologia[c]

        # Calcula os sinais da camada
        # Loop em i pelos neurônios das camadas
        # Loop em j pelos neurônios da camada anterior
        # Somatório do produto entre entradas da camada anterior e pesos + bias
        # Passa pela função de ativação
        camada_sinais = [ativ[c - 1](
                                    sum(
                                        pesos[c - 1][(i - 1) * n_in + j] * camada_anterior[j]  for j in 1:n_in
                                        ) 
                                    + bias[c - 1][i]
                                    ) for i in 1:n_out
                        ]

        # Concatena os sinais da camada na matriz geral de sinais (vetor de vetores)
        sinais = vcat(sinais, [camada_sinais])

    end

    # Aloca a última linha de sinais no vetor de saídas
    # Como não aplicamos a ativação Softmax na última camada, ainda é um logit
    logits_saida = sinais[end]

    # Retorna os logits de saída
    return logits_saida

end




