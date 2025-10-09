# Atualiza os vetores de vetores de pesos e bias utilizando o vetor de variáveis de projeto x
function Atualiza_pesos_bias(rede::Rede, x::Vector{Float64})

    # Acessa os termos em Rede por apelidos 
    n_camadas = rede.n_camadas
    #conexoes = rede.conexoes
    topologia = rede.topologia
    pesos_ranges = rede.pesos_ranges
    bias_ranges  = rede.bias_ranges
    
    # Pesos: Vetor de vetores ("matriz") com os pesos de cada conexão da rede neural
    # Cada vetor representa uma camada oculta
    # Copia os primeiros valores de x na matriz de pesos através das conexões

    pesos = [Matrix{Float64}(undef,topologia[i+1], topologia[i]) for i in 1:n_camadas]

    #pesos = Vector{Matrix{Float64}}(undef, n_camadas)

    # Bias: Vetor de vetores ("matriz") com os bias de cada neurônio da rede neural
    # Cada linha representa uma camada oculta

    # Passa os ultimos n_neuronios valores de x para a matriz 
    # de bias, usando topologia como guia para saber quantos
    # valores gravamos por linha

    bias = [Vector{Float64}(undef,topologia[i+1]) for i in 1:n_camadas]

    # bias = Vector{Vector{Float64}}(undef,n_camadas)

    #
    # Agora só usamos os valores pré-calculados de acessos e 
    # também usamos @views para evitar alocação de memória
    #
    @inbounds for i in eachindex(pesos)
        copyto!(pesos[i], reshape(@view(x[pesos_ranges[i]]), topologia[i+1], topologia[i]))
        copyto!(bias[i], @view(x[bias_ranges[i]]))
    end

    # Vamos garantir que nunca vamos acessar fora da memória alocada
    #=
    @inbounds begin
        k = 1
        for i in eachindex(pesos)
            
            # Teste com o uso de view (para não alocar memória)
            w_view = @view x[k : k + conexoes[i] - 1]

            # Reshape para uma matriz de pesos...
            pesos[i] = reshape(w_view, topologia[i+1], topologia[i])

            # Offset para a próxima camada em x
            k += conexoes[i]

            # Vamos testar uma view aqui também
            b_view = @view x[k : k + topologia[i+1] - 1]
            bias[i] = b_view

            # Mais um offset
            k += topologia[i+1]
        end
    end
    =#

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

    # Cria e pré-aloca o vetor de vetores
    sinais = [Vector{Float64}(undef,topologia[i]) for i in 1:n_camadas+1]

    #sinais = Vector{Vector{Float64}}(undef, n_camadas+1)

    # Inicializa sinais    
    # Incluí o vetor de entradas na primeira linha de sinais
    sinais[1] = copy(entrada_i)

    # Loop pelas camadas
    @inbounds for c = 2:(n_camadas+1)

        # Recupera a camada anterior de sinais
        camada_anterior = sinais[c-1]

        # Recupera número de neurônios da camada anterior e da nova camada
        #n_in = topologia[c - 1]
        #n_out = topologia[c]

        # Vamos tentar otimizar o código, transformando os pesos em uma matriz por 
        # camada
        W = pesos[c-1] 
        b = bias[c-1]
        ϕ = ativ[c-1]

        # Faz o forward (agora é só um produto de matriz por vetor)
        # vou aproveitar a memória que já está alocada em sinais[c]
        sinais[c] .= W * camada_anterior .+ b

        # Armazena
        sinais[c] = ϕ.(sinais[c])

    end

    # Aloca a última linha de sinais no vetor de saídas
    # Como não aplicamos a ativação Softmax na última camada, ainda é um logit
    logits_saida = sinais[end]

    # Retorna os logits de saída
    return logits_saida

end




