# Atualiza os vetores de vetores de pesos e bias utilizando o vetor de variáveis de projeto x
function Atualiza_pesos_bias(rede::Rede, x::Vector{Float64})

    # Acessa os termos em Rede por apelidos 
    n_camadas = rede.n_camadas
    #conexoes = rede.conexoes
    topologia = rede.topologia
    pesos_ranges = rede.pesos_ranges
    bias_ranges  = rede.bias_ranges
    
    # Aloca o vetor de Pesos: Vetor de matrizes com os pesos de cada camada da rede
    pesos = [Matrix{Float64}(undef,topologia[i+1], topologia[i]) for i in 1:n_camadas]

    # Aloca o vetor de biases: vetor de vetores com os bias dos neurônios de cada camada
    bias = [Vector{Float64}(undef,topologia[i+1]) for i in 1:n_camadas]

    #
    # Agora só usamos os valores pré-calculados de acessos e 
    # também usamos @views para evitar alocação de memória
    #
    @inbounds for i in eachindex(pesos)
        copyto!(pesos[i], reshape(@view(x[pesos_ranges[i]]), topologia[i+1], topologia[i]))
        copyto!(bias[i], @view(x[bias_ranges[i]]))
    end

    # Retorna matrizes de pesos e bias
    return pesos, bias

end

#
# Forward da Rede neural
#
function RNA(rede::Rede, 
             pesos::Vector{Matrix{Float64}}, bias::Vector{Vector{Float64}}, 
             entrada_i::Vector{Float64})::Vector{Float64}

    # Acessa os termos em Rede por apelidos 
    n_camadas = rede.n_camadas
    topologia = rede.topologia
    ativ      = rede.ativ

    # Pré-aloca a memória para sinais, que será utilizada várias vezes nesta rotina
    # a cada chamada de RNA
    sinais = [Vector{Float64}(undef,tt) for tt in topologia] 

    # Inclui o vetor de entradas na primeira linha de sinais
    copyto!(sinais[1], copy(entrada_i))

    # Loop pelas camadas
    for c = 2:(n_camadas+1)

        # Recupera a camada anterior de sinais
        camada_anterior = sinais[c-1]

        # Aliases para a matriz de pesos, vetor de bias e funções de ativação 
        W = pesos[c-1] 
        #b = bias[c-1]
        ϕ = ativ[c-1]

        # Copia os bias para sinais[c]
        copyto!(sinais[c],bias[c-1])

        # Calcula W*camada_anterior + b usando o mul! de 5 parâmetros
        mul!(sinais[c],W,camada_anterior,1.0,1.0)

        # Faz o forward (agora é só um produto de matriz por vetor + vetor de bias)
        # vou aproveitar a memória que já está alocada em sinais[c]
        #copyto!(sinais[c] , W * camada_anterior .+ b)

        # Aplica a função de ativação e armazena na mesma área de memória
        copyto!(sinais[c] , ϕ.(sinais[c]))

    end

    # Retorna os dados da ultima camada de sinais (saídas)
    return sinais[end]

end




