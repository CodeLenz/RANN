# Atualiza os vetores de vetores de pesos e bias utilizando o vetor de variáveis de projeto x
function Atualiza_pesos_bias(rede::Rede, x::Vector{Float64})

    # Acessa os termos em Rede por apelidos 
    n_camadas = rede.n_camadas
    conexoes = rede.conexoes
    topologia = rede.topologia
    
    # Pesos: Vetor de vetores ("matriz") com os pesos de cada conexão da rede neural
    # Cada vetor representa uma camada oculta
    # Copia os primeiros valores de x na matriz de pesos através das conexões
    pesos = Vector{Matrix{Float64}}(undef, n_camadas)

    # Bias: Vetor de vetores ("matriz") com os bias de cada neurônio da rede neural
    # Cada linha representa uma camada oculta

    # Passa os ultimos n_neuronios valores de x para a matriz 
    # de bias, usando topologia como guia para saber quantos
    # valores gravamos por linha
    bias = Vector{Vector{Float64}}(undef,n_camadas)

    # Inicializa contador k que busca os pesos e biases dentro de x
    k = 1

    # Loop pelas linhas da matriz (camadas da rede)
    for i = 1:n_camadas

        # Seleciona as variáveis de projeto da camada e aloca na matriz de pesos
        pesos[i] = reshape(x[k:(k+conexoes[i]-1)], topologia[i+1], topologia[i])

        # Atualiza o contador para os biases
        k = k + conexoes[i]

        # Seleciona as variáveis de projeto e aloca no vetor de biases
        bias[i] = x[k:(k+topologia[i+1]-1)]

        # Atualiza o contador para a próxima camada
        k = k + topologia[i+1]

    end

    # Retorna matrizes de pesos e bias
    return pesos, bias

end


# Rede neural
function RNA(rede::Rede, pesos::Vector{Matrix{Float64}}, bias::Vector{Vector{Float64}}, 
             entrada_i::Vector{Float64})::Vector{Float64}

    # Acessa os termos em Rede por apelidos 
    topologia = rede.topologia
    n_camadas = rede.n_camadas
    ativ      = rede.ativ

    sinais = Vector{Vector{Float64}}(undef, n_camadas+1)

    # Inicializa sinais    
    # Incluí o vetor de entradas na primeira linha de sinais
    sinais[1] = copy(entrada_i)

    # Loop pelas camadas
    for c = 2:(n_camadas+1)

        # Recupera a camada anterior de sinais
        camada_anterior = sinais[c-1]

        # Recupera número de neurônios da camada anterior e da nova camada
        n_in = topologia[c - 1]
        n_out = topologia[c]

        # Vamos tentar otimizar o código, transformando os pesos em uma matriz por 
        # camada
        W = pesos[c-1] 
        b = bias[c-1]
        ϕ = ativ[c-1]

        # Faz o forward (agora é só um produto de matriz por vetor)
        z = W * camada_anterior .+ b

        # Armazena
        sinais[c] = ϕ.(z)

    end

    # Aloca a última linha de sinais no vetor de saídas
    # Como não aplicamos a ativação Softmax na última camada, ainda é um logit
    logits_saida = sinais[end]

    # Retorna os logits de saída
    return logits_saida

end




