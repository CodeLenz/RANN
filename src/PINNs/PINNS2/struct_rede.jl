# Struct Rede
# Contém as definições básicas da rede neural
struct Rede

    # Topologia da rede
    topologia::Vector{Int64}

    # Funções de ativação da rede
    ativ::Tuple

    # Número de entradas
    n_entradas::Int64

    # Número de saídas
    n_saidas::Int64

    # Número de camadas da rede
    n_camadas::Int64

    # Número de neurônios total da rede
    n_neuronios::Int64

    # Conexões da rede
    conexoes::Vector{Int64}

    # Número máximo de conexões em uma camada da rede
    # É o número máximo de colunas na matriz de pesos 
    n_maximo_pesos::Int64

    # Número total de conexões
    n_total_conect::Int64

    # Número máximo de neurônios em uma camada
    n_max_neuronio::Int64 

    # Número de variáveis de projeto da rede, que é a soma do número de pesos 
    # com o número de bias
    n_projeto::Int64

    # Vetor de variáveis de projeto
    x::Vector{Float64}

    # Vou colocar os acessos aos pesos e bias pré-calculados 
    # aqui na rede
    pesos_ranges::Vector{UnitRange{Int}}
    bias_ranges::Vector{UnitRange{Int}}

    # Função que inicializa todas as variáveis calculadas na struct
    # Topologia e funções de ativações são os parâmetros informados pelo usuário
    function Rede(topologia, ativ)

        # Validações dos dados de entrada topologia e ativ
        if length(topologia) < 2

            error("Ao menos uma camada é necessária")

        end

        # Valida se o número de funções de ativação corresponde ao número de camadas
        if length(ativ) != (length(topologia) - 1)
        
            error("Número de funções de ativação propostas está incorreto")

        end

        # Número de entradas
        n_entradas = topologia[1]

        # Número de saídas
        n_saidas = topologia[end]

        # Número de camadas
        n_camadas = length(topologia) - 1

        # Número de neurônios total da rede
        n_neuronios = sum(topologia[2:end]) 

        # Chama função que calcula vetor de conexões, número máximo de pesos
        # e número total de conexões
        conexoes, n_maximo_pesos, n_total_conect = Conexoes(topologia, n_camadas)

        # Número máximo de neurônios em uma camada
        n_max_neuronio = maximum(topologia)

        # Calcula o número de variáveis de projeto 
        # n_total_conect + número de neurônios (número de bias)
        n_projeto = n_total_conect + n_neuronios

        # Inicializa o vetor de variáveis de projeto (pesos e bias)
        x = randn(n_projeto) #IniciaXHe(n_projeto, n_camadas, topologia, conexoes)

        # Pre-computa os acessos 
        pesos_ranges, bias_ranges = Pre_computa_acessos(topologia,conexoes)  

        # Cria o tipo e passa todos os dados da rede
        new(topologia, ativ, n_entradas, n_saidas, n_camadas, n_neuronios, conexoes, 
            n_maximo_pesos, n_total_conect, n_max_neuronio, n_projeto, x, 
            pesos_ranges, bias_ranges)

    end

end

# Rotina que retorna um vetor com o número de conexões entre cada camada da rede,
# o número máximo de conexões entre camadas e o total de conexões
function Conexoes(topologia::Vector{Int64}, n_camadas::Int64)

    # Define vetor com o número de conexões entre as camadas
    # Vetor deve ser de tipo inteiro
    conexoes = zeros(Int64, n_camadas)
    
    # Calcula o número de conexões entre as camadas
    for i in LinearIndices(conexoes)

        conexoes[i] =  topologia[i] * topologia[i+1]

    end
    
    # O número máximo de conexões (camada com maior número)
    # e o número total também são retornados
    return conexoes, maximum(conexoes), sum(conexoes)

end

# Inicializa o vetor de variáveis de projeto X conforme a inicialização He
function IniciaXHe(n_projeto::Int64, n_camadas::Int64, topologia::Vector{Int64}, conexoes::Vector{Int64})

    # Aloca vetor de variáveis de projeto
    x = zeros(n_projeto)

    # Pesos: Os pesos iniciais seguem uma distribuição normal na camada, com variância de sqrt(2 / n_camadas_input)
    # Método busca diminuir a variância entre os pesos, evitando valores muito grandes ou muito pequenos

    # Inicializa contador para as conexões
    k = 1

    # Loop pelas camadas
    for i = 1:n_camadas

        # Variância de He
        std = sqrt(2 / topologia[i])

        # Pesos: Define valores aleatórios conforme distribuição normal (randn) e escala multiplicando pela variância
        x[k:(k+conexoes[i]-1)] = randn(conexoes[i]) * std

        # Atualiza o contador para os biases
        k = k + conexoes[i]

        # Biases: Define valores bem pequenos para iniciar
        # Início neutro, sem empurrar neurônios para valores muito negativos (morrem) ou positivos (acendem)
        x[k:(k+topologia[i+1]-1)] .= 0.01

        # Atualiza o contador para a próxima camada
        k = k + topologia[i+1]

    end

    # Retorna o vetor inicial
    return x

end


#
# Rotina que pré-computa os acessos aos pesos e bias 
#
function Pre_computa_acessos(topologia,conexoes)

    # Número de conexoes
    n = length(conexoes)

    # Aloca os queridos...
    pesos_ranges = Vector{UnitRange{Int}}(undef, n)
    bias_ranges  = Vector{UnitRange{Int}}(undef, n)

    # Vamos lá...
    k = 1
    for i in 1:n
        pesos_ranges[i] = k : k + conexoes[i] - 1
        k += conexoes[i]
        bias_ranges[i]  = k : k + topologia[i+1] - 1
        k += topologia[i+1]
    end

    # Retorna os valores pré-calculados
    return pesos_ranges, bias_ranges

end
