# Função objetivo que depende das entradas da rede neural
# R^n -> R, onde n é o número total de pesos e bias da rede
# λ1 e λ2 são hiperparâmetros para ponderação dos termos da função objetivo
function Objetivo(rede::Rede, treino::NamedTuple, epoch::Int64, x::Vector{Float64},
                  λ1 = 1.0E-1, λ2 = 1.0E-2)


   # Aloca as matrizes de pesos e bias a partir das variáveis de projeto
   pesos, bias = Atualiza_pesos_bias(rede, x)

   # Define fator física
   fator_fis = 1.0

   # Define os componentes de erro da rede
   # Perda física, Perda de contorno, Perda inicial em t e dt
   perda = zeros(3)

   # Aloca as derivadas em relação ao tempo 
   du = zeros(1)
   d2u = zeros(1)

   # Aloca vetor de saída da rede
   u0 = zeros(rede.topologia[end])

   #
   # Condições iniciais 
   #

   # Extrai os valores das condições iniciais
	t_inicial = treino.t_inicial
	u_inicial = treino.u_inicial
	du_inicial = treino.du_inicial

   # Calcula o valor do deslocamento no tempo t0
   u0 = RNA(rede, pesos, bias, t_inicial)
    
   # Testa por NaN
   if any(isnan.(u0)) 
      error("Nan em u0 ") 
   end 

   # Calcula a perda relativa a primeira condição inicial: u(t0)
   perda[1] += Fn_perda_inicial(u0, u_inicial)

   # Calcula a primeira e a segunda derivada ao mesmo tempo
   DerivadasC2!(RNA, rede, pesos, bias, u0, du, d2u, t_inicial)
    
   # Testa por NaN
   if any(isnan.(du)) 
      error("Nan em du CI") 
   end

   # Testa por NaN
   if any(isnan.(d2u)) 
      error("Nan em d2u CI") 
   end

   # Calcula a perda da velocidade inicial 
   perda[2] += Fn_perda_inicial(du, du_inicial)

   #
   # Perda do resíduo da EDO
   #
        
   # Perda física, associada ao atendimento da equação diferencial nos pontos de treino    
   # Loop pelos pontos de treino
   for coluna in axes(treino.t_fisica, 2)
 
      # Extrai as entradas da rede
      t_i = treino.t_fisica[:, coluna]

      # Valores 
      u0 = RNA(rede, pesos, bias, t_i)

      # Testa por NaN
      if any(isnan.(u0)) 
         error("Nan em u0 físico ") 
      end 


      # Obtém a primeira e segunda derivada - velocidade e aceleração
      DerivadasC2!(RNA, rede, pesos, bias, u0, du, d2u, t_i)
    
      # Testa por NaN
      if any(isnan.(du)) 
         error("Nan em du física") 
      end

      # Testa por NaN
      if any(isnan.(d2u)) 
         error("Nan em d2u física") 
      end

      # Calcula a perda
      perda[3] += Fn_perda_fisica(treino, u0, du, d2u)

   end

   # Calcula a perda física média
   perda[3] /= size(treino.t_fisica, 2)

   # Soma as componentes de perda
   # TODO: utilizar fator_fis somente no ADAM
   fator_fis = min(epoch / 500, 1.0)

   # Soma as componentes de perda para valor do objetivo
   obj =  perda[1] + λ1 * perda[2] + λ2 * fator_fis * perda[3]

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