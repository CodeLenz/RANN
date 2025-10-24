# Função objetivo que depende das entradas da rede neural
# R^n -> R, onde n é o número total de pesos e bias da rede
# λ1 e λ2 são hiperparâmetros para ponderação dos termos da função objetivo
function Objetivo(rede::Rede, treino::Treino, t_inicial::Vector{Float64}, u_inicial::Vector{Float64},
                  du_inicial::Vector{Float64}, n_fisica::Int64, t_fisica::Matrix{Float64}, epoch::Int64,
                  x::Vector{Float64}, λ1 = 1.0E-1, λ2 = 1.0E-3)

    # Alias
    topologia = rede.topologia
   
    #
    # Uma técnica interessante é gerar pontos aleatórios na faixa
    # (0,1) com o mesmo número de colunas de t_fisica
    #
    # CUIDADO...aqui temos que ver a faixa de tempo pois estou assumindo 
    # (0,1)
    #
    #t_fisica_rand = similar(t_fisica)
    #t_fisica_rand[1,:] .= rand(size(t_fisica,2))
   
    # Aloca as matrizes de pesos e bias a partir das variáveis de projeto
    pesos, bias = Atualiza_pesos_bias(rede, x)

    # Define os componentes de erro da rede
    perda_inicial_u = 0.0
    perda_inicial_du = 0.0
    perda_fisica = 0.0

    # Aloca as derivadas em relação ao tempo 
    du = zeros(1)
    d2u = zeros(1)

    # Aloca vetor de saída da rede
    u0 = zeros(topologia[end])

    #
    # Condições iniciais 
    #
    # Calcula o valor do deslocamento no tempo t0
    u0 .= RNA(rede,  pesos, bias, t_inicial)
    
    # Calcula a perda relativa a primeira condição inicial: u(t0)
    perda_inicial_u += Fn_perda_inicial(u0, u_inicial)

    # Calcula a primeira e a segunda derivada ao mesmo tempo
    DerivadasC2!(RNA, rede, pesos, bias, u0, du, d2u, t_inicial)
    
    # Calcula a perda da velocidade inicial 
    perda_inicial_du += Fn_perda_inicial(du, du_inicial)

    #
    # Perda do resíduo da EDO
    #
        
    # Perda física, associada ao atendimento da equação diferencial nos pontos de treino    
    # Loop pelos pontos de treino
    for coluna in axes(t_fisica,2) # 1:n_fisica
 
        # Extrai as entradas da rede
        #t_i = t_fisica_rand[:, coluna]
        t_i = t_fisica[:, coluna]

        # Valores 
        u0 .= RNA(rede, pesos, bias, t_i)
    
        # Obtém a primeira e segunda derivada - velocidade e aceleração
        DerivadasC2!(RNA, rede, pesos, bias, u0, du, d2u, t_i)
        
        # Calcula a perda
        perda_fisica += Fn_perda_fisica(treino, u0, du, d2u)

    end

    # Calcula a perda física média
    perda_fisica /= n_fisica

    # Soma as componentes de perda
    # TODO: utilizar fator_fis somente no ADAM
    fator_fis = min(epoch / 500, 1.0)
    perda = perda_inicial_u + λ1 * perda_inicial_du + λ2 * fator_fis * perda_fisica

    # Retorna a perda total ponderada pelos hiperparâmetros λ
    return perda, perda_inicial_u, perda_inicial_du, perda_fisica

end

# Cria um wrapper para enganar o Enzyme
# Retorna apenas o valor float que queremos diferenciar da função objetivo, visto que ele não consegue
# diferenciar mais de um parâmetro (tupla)
function ObjetivoFloat(rede::Rede, treino::Treino, t_inicial::Vector{Float64}, u_inicial::Vector{Float64},
                       du_inicial::Vector{Float64}, n_fisica::Int64, t_fisica::Matrix{Float64}, epoch::Int64,
                       x::Vector{Float64})

    perda, _ = Objetivo(rede, treino, t_inicial, u_inicial, du_inicial, n_fisica, t_fisica, epoch, x)

    return perda

end