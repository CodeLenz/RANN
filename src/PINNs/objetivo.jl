# Função objetivo que depende das entradas da rede neural
# R^n -> R, onde n é o número total de pesos e bias da rede
# λ1 e λ2 são hiperparâmetros para ponderação dos termos da função objetivo
function Objetivo(rede::Rede, treino::Treino, t_inicial::Vector{Float64}, u_inicial::Vector{Float64},
                  du_inicial::Vector{Float64}, n_fisica::Int64, t_fisica::Matrix{Float64},
                  x::Vector{Float64},  λ1 = 1.0, λ2 = 1.0E-2)

    # Alias
    topologia = rede.topologia
   
    # Aloca as matrizes de pesos e bias a partir das variáveis de projeto
    pesos, bias = Atualiza_pesos_bias(rede, x)

    # Define os componentes de erro da rede
    perda_inicial_u = 0.0
    perda_inicial_du = 0.0
    perda_fisica = 0.0

    # Aloca as derivadas em relação ao tempo 
    du = zeros(1)
    d2u = zeros(1)

    # Aloca sinais aqui fora
    sinais = [zeros(Float64,tt) for tt in topologia] 

    # Aloca vetor de saída da rede
    u0 = zeros(topologia[end])

    #
    # Condições iniciais 
    #
    # Calcula o valor do deslocamento no tempo t0
    RNA!(rede, sinais, pesos, bias, t_inicial)
    u0 .= sinais[end]

    # Calcula a perda relativa a primeira condição inicial: u(t0)
    perda_inicial_u += Fn_perda_inicial(u0, u_inicial)

    # Calcula as derivadas em t_inicial LER OS COMENTÁRIOS NA 
    # ROTINA QUE ESTÁ SENDO UTILIZADA E NAS QUE ESTÃO COMENTADAS.
    # LEMBRAR QUE TAMBÉM ESTAMOS CALCULANDO AS DERIVADAS NESTA MESMA 
    # ROTINA em um loop mais para baixo
    #
    # Calcula a primeira e a segunda derivadas ao mesmo tempo, usando 
    # DFC. 
    #Derivadas!(RNA!, rede, sinais, pesos, bias, u0, du, d2u, t_inicial)

    # Usando aproximação polinomial...em teste
    Derivadas_O3!(RNA!, rede, sinais, pesos, bias, u0, du, d2u, t_inicial)

    # Calcula a primeira e a segunda derivada ao mesmo tempo, usando aproximações 
    # de DF de alta ordem. Mais caro do que Derivadas!
    # Derivadas_O2!(RNA!, rede, sinais, pesos, bias, u0, du, d2u, t_inicial)
    # Vamos fazer um teste aqui

    # Calcula somente a primeira derivada 
    # Essa abordagem, junto com Derivada2! é mais cada do que as anteriores
    #Derivada!(RNA!,rede,sinais,pesos,bias,du,t_inicial)

    # Segunda derivada, usando a rotina para calcular a primeira derivada. 
    # Essa abordagem é mais cara do que as anteriores
    #Derivada2!(RNA!,rede,sinais,pesos,bias,d2u,t_inicial)

    # Calcula a perda da velocidade inicial 
    perda_inicial_du += Fn_perda_inicial(du, du_inicial)

    #
    # Perda do resíduo da EDO
    #
        
    # Perda física, associada ao atendimento da equação diferencial nos pontos de treino    
    # Loop pelos pontos de treino
    for coluna in axes(t_fisica,2) # 1:n_fisica
 
        # Extrai as entradas da rede
        t_i = t_fisica[:, coluna]

        # Valores 
        RNA!(rede, sinais, pesos, bias, t_i)

        # Reaproveita o u0 que já foi alocado fora do loop 
        u0 .= sinais[end]
     
        # Obtém a primeira e segunda derivada - velocidade e aceleração
        # Derivadas!(RNA!, rede, sinais, pesos, bias, u0, du, d2u, t_i)
        Derivadas_O2!(RNA!, rede, sinais, pesos, bias, u0, du, d2u, t_i)

        # Primeira derivada 
        #Derivada!(RNA!,rede,sinais,pesos,bias,du,t_i)

        # Segunda derivada 
        #Derivada2!(RNA!,rede,sinais,pesos,bias,d2u,t_i)

        # Calcula a perda
        perda_fisica += Fn_perda_fisica(treino, u0, du, d2u)

    end

    # Calcula a perda física média
    perda_fisica /= n_fisica

    # Retorna a perda total ponderada pelos hiperparâmetros λ
    return perda_inicial_u + λ1 * perda_inicial_du + λ2 * perda_fisica

end
