# Função objetivo que depende das entradas da rede neural
# R^n -> R, onde n é o número total de pesos e bias da rede
# λ1 e λ2 são hiperparâmetros para ponderação dos termos da função objetivo
function Objetivo(rede::Rede, treino::NamedTuple, epoch::Int64, x::Vector{Float64},
                  λ1 = 1.0E-1, λ2 = 1.0E-1, λ3 = 1.0E-2)
   
   	# Aloca as matrizes de pesos e bias a partir das variáveis de projeto
   	pesos, bias = Atualiza_pesos_bias(rede, x)

   	# Define os componentes de erro da rede
   	# Perda física, Perda de contorno, Perda inicial em t e dt
   	perda = zeros(4)

   	# Aloca as derivadas em relação as variáveis (x, y) 
	# Primeira derivada: du/dx du/dy
   	du_xy = [zeros(1) for _ in 1:2]

	# Segunda derivada: d2u/dx2 d2u/dxdy; d2u/dydx d2u/dy2
   	d2u_xy = [zeros(2) for _ in 1:2]

	# Aloca as derivadas em relação a variável (t)
	du_t = zeros(1)
	d2u_t = zeros(1)

   	# Aloca vetor de saída da rede
   	u0 = zeros(rede.topologia[end])

   	# 
   	# Condições de contorno
   	#
	# Verifica a existência das condições de contorno
	if !isnothing(treino.contorno)

		# Loop pelas condições de contorno
   		for coluna in axes(treino.contorno, 2)
 
    		# Extrai as entradas da rede
      		cc_i = treino.contorno[1:(end-1), coluna]

      		# Extrai o valor esperado no contorno
      		u0_esperado = treino.contorno[end, coluna]

      		# Valores 
      		u0 .= RNA(rede, pesos, bias, cc_i)

      		# Testa por NaN
      		if any(isnan.(u0)) 
        		error("Nan em u0 físico") 
      		end 

      		# Calcula a perda
      		perda[1] += Fn_CC_CI(u0, [u0_esperado])

		end

		# Calcula a perda de contorno média
    	perda[1] /= size(treino.contorno, 2)

   	end

   	#
   	# Condições iniciais 
   	#
	# Verifica a existência das condições iniciais
	if !isnothing(treino.t_inicial)

		# Extrai os valores das condições iniciais
		t_inicial = treino.t_inicial
		u_inicial = treino.u_inicial
		du_inicial = treino.du_inicial

   		# Calcula o valor do deslocamento no tempo t0
   		u0 .= RNA(rede, pesos, bias, [t_inicial])
      
   		# Testa por NaN
   		if any(isnan.(u0)) 
      		error("Nan em u0 ") 
   		end 

		# Verifica a existência da condição inicial da variável
		if !isnothing(u_inicial)

    		# Calcula a perda relativa a primeira condição inicial: u(t0)
    		perda[2] += Fn_CC_CI(u0, [u_inicial])

		end

    	# Calcula a primeira e a segunda derivada ao mesmo tempo
    	DerivadasC2!(RNA, rede, pesos, bias, u0, du_t, d2u_t, t_inicial)
    
    	# Testa por NaN
    	if any(isnan.(du_t)) 
       		error("Nan em du CI") 
    	end

    	# Testa por NaN
    	if any(isnan.(d2u_t)) 
       		error("Nan em d2u CI") 
    	end

		# Verifica a existência da condição inicial da primeira derivada
		if !isnothing(du_inicial)

			# Calcula a perda da condição inicial da primeira derivada 
    		perda[3] += Fn_CC_CI(du_t, [du_inicial])

		end

	end

    #
    # Perda física (resíduo da equação diferencial)
    #
    # Perda física, associada ao atendimento da equação diferencial nos pontos de treino    
    # Loop pelos pontos de treino
    for coluna in axes(treino.fisica, 2)
 
        # Extrai as entradas da rede
        x_i = treino.fisica[:, coluna]

        # Valores 
        u0 .= RNA(rede, pesos, bias, x_i)

        # Testa por NaN
        if any(isnan.(u0)) 
           error("Nan em u0 físico ") 
        end 

        # Obtém a primeira e segunda derivada - velocidade e aceleração
        DerivadasPDE!(RNA, rede, pesos, bias, u0, du_xy, d2u_xy, x_i)

        # Testa por NaN
        if any(isnan.(du_xy[1])) |  any(isnan.(du_xy[2]))
           error("Nan em du física") 
        end

        # Testa por NaN
        if any(isnan.(d2u_xy[1])) | any(isnan.(d2u_xy[2]))
           error("Nan em d2u física") 
        end

        # Calcula a perda
        perda[4] += Fn_perda_fisica(u0, du_xy, d2u_xy, x_i)

    end

    # Calcula a perda física média
    perda[4] /= size(treino.fisica, 2)

    # Soma as componentes de perda para valor do objetivo
    # TODO: utilizar fator_fis somente no ADAM
    fator_fis = max(epoch / 500, 1.0)
    obj = perda[1] + λ1 * perda[2] + λ2 * perda[3] + λ3 * fator_fis * perda[4]

	# Checa termos de perda para nan
    if any(isnan.(perda)) | isnan.(obj)
       
       error("NaN nas perdas $perda, $obj") 
    end


    # Retorna a perda total ponderada pelos hiperparâmetros λ
    return obj, perda

end

# Cria um wrapper para enganar o Enzyme
# Retorna apenas o valor float que queremos diferenciar da função objetivo, visto que ele não consegue
# diferenciar mais de um parâmetro (tupla)
function ObjetivoFloat(rede::Rede, treino::NamedTuple, epoch::Int64, x::Vector{Float64})::Float64  

    obj, _ = Objetivo(rede, treino, epoch, x)

    return obj

end