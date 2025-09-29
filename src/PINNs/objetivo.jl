# Função objetivo que depende das entradas da rede neural
# R^n -> R, onde n é o número total de pesos e bias da rede 
function Objetivo(rede::Rede, treino::Treino, t_inicial::Vector{Float64}, u_inicial::Vector{Float64},
                  du_inicial::Vector{Float64}, 
                  n_fisica::Int64, t_fisica::Matrix{Float64}, x::Vector{Float64},
                  λ1 = 1E-1, λ2 = 1E-4)

    # Aloca as matrizes de pesos e bias a partir das variáveis de projeto
    pesos, bias = Atualiza_pesos_bias(rede, x)

    # Parametriz a rede neural como uma função "apenas" de t
    #u_i(t) = RNA(rede, x, t)

    # Define os componentes de erro da rede
    perda_inicial_u = 0.0
    perda_inicial_du = 0.0
    perda_fisica = 0.0

    # Aloca as derivadas em relação ao tempo 
    du = zeros(1)
    d2u = zeros(1)

    # Calcula o valor
    u_inicial_calculado = RNA(rede,pesos,bias,t_inicial)

    #
    # Calcula a perda relativa a primeira condição inicial: u(t0)
    #
    perda_inicial_u += Fn_perda_contorno([u_inicial_calculado], u_inicial)
    
    # Primeira derivada - velocidade
    # Aloca a primeira derivada e calcula valor inplace
    # du_inicial_pred = zeros(1)

    # Calcula as derivadas em t_inicial
    Derivadas!(RNA,rede,pesos,bias,u_inicial_calculado,du,d2u,t_inicial)

    #PrimeiraDerivada!(RNA, rede, x, du_inicial_pred, t_inicial)

    # Calcula a perda
    perda_inicial_du += Fn_perda_contorno(du, du_inicial)
        
    # Perda física, associada ao atendimento da equação diferencial nos pontos de treino    
    # Loop pelos pontos de treino
    for coluna = 1:n_fisica
 
        # Extraí as entradas da rede
        t_i = t_fisica[:, coluna]

        # Valores 
        u_i_valores = RNA(rede,pesos,bias,t_i)

        # Obtém a primeira e segunda derivada - velocidade e aceleração
        #du_i =  [0.0]
        #du2_i = [0.0]

        # Calcula a primeira derivada 
        Derivadas!(RNA,rede,pesos,bias,u_i_valores,du,d2u,t_i)
        #PrimeiraDerivada!(RNA, rede, x, du_i, t_i)

        # Calcula a segunda derivada aproveitando o valor que já temos da primeira derivada
        #SegundaDerivada!(RNA, rede, x, du_i, du2_i, t_i)    

        # Calcula a perda
        perda_fisica += Fn_perda_fisica(treino, [u_i_valores], du, d2u)

    end

    # Calcula a perda física média
    perda_fisica /= n_fisica

    # Retorna a perda total ponderada pelos hiperparâmetros λ
    return perda_inicial_u + λ1 * perda_inicial_du + λ2 * perda_fisica

end
