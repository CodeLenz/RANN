# Função objetivo que depende das entradas da rede neural
# R^n -> R, onde n é o número total de pesos e bias da rede 
function Objetivo(rede::Rede, treino::Treino, t_inicial::Vector{Float64}, u_inicial::Vector{Float64},
                  du_inicial::Vector{Float64}, 
                  n_fisica::Int64, t_fisica::Matrix{Float64}, x::Vector{T},
                  λ1 = 1E-1, λ2 = 1E-4) where T

    # Aloca as matrizes de pesos e bias a partir das variáveis de projeto
    pesos, bias = Atualiza_pesos_bias(rede, x)

    # Define os componentes de erro da rede
    perda_inicial_u = zero(T)
    perda_inicial_du = zero(T)
    perda_fisica = zero(T)

    # Perda associada a condição de contorno essencial
    # Função de t para cálculo das derivadas
    #u_inicial_pred(t) = RNA(rede, pesos, bias, t)

    # Calcula o valor
    u_inicial_calculado = RNA(rede,pesos,bias,t_inicial)

    #
    # Calcula a perda relativa a primeira condição inicial: u(t0)
    #
    perda_inicial_u += Fn_perda_contorno([u_inicial_calculado], u_inicial)
    
    # Primeira derivada - velocidade
    # Aloca a primeira derivada e calcula valor inplace
    du_inicial_pred = zeros(1)

    PrimeiraDerivada!(RNA, rede, pesos, bias, du_inicial_pred, t_inicial)

    # Calcula a perda
    perda_inicial_du += Fn_perda_contorno(du_inicial_pred, du_inicial)
        
    # Perda física, associada ao atendimento da equação diferencial nos pontos de treino    
    # Loop pelos pontos de treino
    for coluna = 1:n_fisica
 
        # Extraí as entradas da rede
        t_i = t_fisica[:, coluna]

        # Roda a rede neural
        # Função de t
        u_i(t) = RNA(rede, pesos, bias, t)

        # Valores 
        u_i_valores = RNA(rede,pesos,bias,t_i)

        # Obtém a primeira e segunda derivada - velocidade e aceleração
        du_i, du2_i = [0.0], [0.0]
        #SegundaDerivada!(u_i, rede, pesos, bias, du_i, du2_i, t_i)    

        # Calcula a perda
        perda_fisica += Fn_perda_fisica(treino, [u_i_valores], du_i, du2_i)

    end

    # Calcula a perda física média
    perda_fisica /= n_fisica

    # Retorna a perda total ponderada pelos hiperparâmetros λ
    return perda_inicial_u + λ1 * perda_inicial_du + λ2 * perda_fisica

end
