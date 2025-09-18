# Função objetivo que depende das entradas da rede neural
# R^n -> R, onde n é o número total de pesos e bias da rede 
function Objetivo(rede::Rede, treino::Treino, t_contorno_ess::Vector{Float64}, u_contorno_ess::Float64,
                  t_contorno_nat::Vector{Float64}, du_contorno_nat::Vector{Float64}, 
                  n_fisica::Int64, t_fisica::Matrix{Float64}, x::Vector{T},
                  λ1 = 1E-1, λ2 = 1E-4) where T

    # Aloca as matrizes de pesos e bias a partir das variáveis de projeto
    pesos, bias = Atualiza_pesos_bias(rede, x)

    # Define os componentes de erro da rede
    perda_contorno_ess = zero(T)
    perda_contorno_nat = zero(T)
    perda_fisica = zero(T)

    # Perda associada a condição de contorno essencial
    # Roda a rede neural
    u_contorno_ess_pred = RNA(rede, pesos, bias, t_contorno_ess)

    # Calcula a perda
    perda_contorno_ess += Fn_perda_contorno(u_contorno_ess_pred, u_contorno_ess)

    # Perda associada a condição de contorno natural
    # Roda a rede neural
    u_contorno_nat_pred = RNA(rede, pesos, bias, t_contorno_ess)

    # Primeira derivada - velocidade
    # Aloca a primeira derivada e calcula valor inplace
    du_contorno_nat_pred = zeros(1)
    PrimeiraDerivada!(u_contorno_nat_pred, du_contorno_nat_pred, t_contorno_nat)

    # Calcula a perda
    perda_contorno_ess += Fn_perda_contorno(du_contorno_nat_pred, du_contorno_nat)
        
    # Perda física, associada ao atendimento da equação diferencial nos pontos de treino    
    # Loop pelos pontos de treino
    for coluna = 1:n_fisica
 
        # Extraí as entradas da rede
        t_i = t_fisica[:, coluna]

        # Roda a rede neural
        u_i = RNA(rede, pesos, bias, t_i)

        # Obtém a primeira e segunda derivada - velocidade e aceleração
        du_i, du2_i = [0.0], [0.0]
        SegundaDerivada!(u_i, du_i, du2_i, t_i)    

        # Calcula a perda
        perda_fisica += Fn_perda_fisica(treino, u_i, du_i, du2_i)

    end

    # Calcula a perda física média
    perda_fisica /= n

    # Retorna a perda total ponderada pelos hiperparâmetros λ
    return perda_contorno_ess + λ1 * perda_contorno_nat + λ2 * perda_fisica

end
