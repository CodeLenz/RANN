# Função objetivo que depende das entradas da rede neural
# R^n -> R, onde n é o número total de pesos e bias da rede
# λ1 e λ2 são hiperparâmetros para ponderação dos termos da função objetivo
function Objetivo(rede::Rede, pontos_treino_fisica::Matrix{Float64}, x::Vector{Float64}, prob::String)::Float64
   
   	# Aloca as matrizes de pesos e bias a partir das variáveis de projeto
   	pesos, bias = Atualiza_pesos_bias(rede, x)

   	# Define os componentes de erro da rede
   	# Perda física
   	perda = 0.0

   	# Aloca as derivadas em relação as variáveis (x, y) 
	# Primeira derivada: du/dx du/dy
   	du_xy  = [zeros(1), zeros(1)]

	# Segunda derivada: d2u/dx2 d2u/dy2
   	d2u_xy = [zeros(1), zeros(1)]

   	# Aloca vetor de saída da rede
   	u0 = zeros(rede.topologia[end])

    #
    # Perda física (resíduo da equação diferencial)
    #
    # Perda física, associada ao atendimento da equação diferencial nos pontos de treino    
    # Loop pelos pontos de treino
    for coluna in axes(pontos_treino_fisica, 2)
 
        # Extrai as entradas da rede
        x_i = pontos_treino_fisica[:, coluna]

        # Valores 
		# Não estou usando .= para evitar o warning do Enzyme
        u0 = RNA_forte(rede, pesos, bias, x_i, prob)

        # Testa por NaN
        if any(isnan.(u0)) 
           error("Nan em u0 físico ") 
        end 

        # Obtém a primeira e segunda derivada 
        DerivadasPDE!(RNA_forte, rede, pesos, bias, u0, du_xy, d2u_xy, x_i, prob)

        # Testa por NaN
        if any(isnan.(du_xy[1])) ||  any(isnan.(du_xy[2]))
           error("Nan em du física") 
        end

        # Testa por NaN
        if any(isnan.(d2u_xy[1])) || any(isnan.(d2u_xy[2]))
           error("Nan em d2u física") 
        end

        # Calcula a perda
        perda += Fn_perda_fisica(u0, du_xy, d2u_xy, x_i)

    end

    # Calcula a perda física média
    perda /= size(pontos_treino_fisica, 2)

	# Checa termos de perda para nan
    if isnan.(perda)
       
       error("NaN nas perdas $perda") 
	   
    end

    # Retorna a perda total ponderada pelos hiperparâmetros λ
    return perda

end